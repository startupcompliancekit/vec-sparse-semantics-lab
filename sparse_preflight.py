"""Read-only numeric ingestion checks for ordinary dense/CSR/CSC H5AD X.

The reference order is source-dtype densification, then float32 conversion.
This checks one narrow input property, not the complete VEC contract.
"""
import argparse
import json
import sys
import warnings

import h5py
import numpy as np
from scipy import sparse

REFERENCE_COMMIT = "cc3859ab2a2305efc4544815d3f44f660efa77d4"


class NotAssessed(ValueError):
    """The supported layout or working-array allowance was exceeded."""


def _dataset(group, name):
    # Only ordinary datasets in this file are in the supported scope.
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise NotAssessed("Required field is absent or is not a local hard link")
    obj = group[name]
    if not isinstance(obj, h5py.Dataset) or obj.is_virtual or obj.external:
        raise NotAssessed("Only ordinary local HDF5 datasets are supported")
    return obj


def _real(dtype):
    if dtype.kind not in "biuf" or dtype.itemsize > 8:
        raise NotAssessed("Only real numeric values up to 64 bits are supported")


def _stored(stats, values):
    stats["stored_entries"] += values.size
    stats["stored_nonfinite"] += int(np.count_nonzero(~np.isfinite(values)))
    stats["stored_negative"] += int(np.count_nonzero(values < 0))
    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(np.float32)
    stats["stored_nonfinite_after_float32"] += int(
        np.count_nonzero(~np.isfinite(converted)))


def _logical(stats, dense):
    # Do not sort, sum_duplicates(), or cast the sparse data before this step.
    # Source-dtype accumulation has already happened in toarray().
    with np.errstate(over="ignore", invalid="ignore"):
        converted = dense.astype(np.float32, copy=False)
    stats["logical_nonfinite_after_float32"] += int(
        np.count_nonzero(~np.isfinite(converted)))
    stats["logical_negative_after_float32"] += int(
        np.count_nonzero(converted < 0))


def inspect(path, buffer_mib=64):
    """Return numeric diagnostics without changing the input or naming it.

    buffer_mib bounds a conservative estimate of temporary array payload per
    block, not process RSS or the HDF5 library's caches. A single major-axis
    slice that does not fit is declined, never partially declared a pass.
    """
    if isinstance(buffer_mib, bool) or not isinstance(buffer_mib, int) or buffer_mib < 1:
        raise NotAssessed("buffer_mib must be a positive integer")
    allowance = buffer_mib * 1024 * 1024
    stats = dict(verdict="NOT_ASSESSED", reference_commit=REFERENCE_COMMIT,
                 storage=None, shape=None, source_dtype=None, blocks=0,
                 duplicate_stored_entries=0, stored_entries=0,
                 stored_nonfinite=0, stored_negative=0,
                 stored_nonfinite_after_float32=0,
                 logical_nonfinite_after_float32=0,
                 logical_negative_after_float32=0,
                 max_estimated_block_bytes=0, buffer_mib=buffer_mib)
    with h5py.File(path, "r") as f:
        if not isinstance(f.get("X", getlink=True), h5py.HardLink):
            raise NotAssessed("X must exist as a local hard link")
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            x = _dataset(f, "X")
            if x.ndim != 2 or min(x.shape) < 1:
                raise NotAssessed("X must have two nonempty dimensions")
            _real(x.dtype)
            stats.update(storage="dense", shape=list(x.shape), source_dtype=str(x.dtype))
            row_bytes = x.shape[1] * (x.dtype.itemsize + 12)
            if row_bytes > allowance:
                raise NotAssessed("One dense row exceeds the working-array allowance")
            step = min(256, max(1, allowance // row_bytes))
            for start in range(0, x.shape[0], step):
                end = min(start + step, x.shape[0])
                values = x[start:end, :]
                _stored(stats, values)
                _logical(stats, values)
                stats["blocks"] += 1
                stats["max_estimated_block_bytes"] = max(
                    stats["max_estimated_block_bytes"], (end - start) * row_bytes)
        elif isinstance(x, h5py.Group):
            encoding = x.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode("ascii", errors="replace")
            if encoding not in ("csr_matrix", "csc_matrix"):
                raise NotAssessed("X group must use csr_matrix or csc_matrix encoding")
            shape = np.asarray(x.attrs.get("shape", []))
            if shape.shape != (2,) or shape.dtype.kind not in "iu" or np.any(shape < 1):
                raise NotAssessed("Sparse X needs a positive integer shape of length two")
            shape = tuple(int(n) for n in shape)
            data, indices, indptr = (_dataset(x, n) for n in ("data", "indices", "indptr"))
            _real(data.dtype)
            if any(a.ndim != 1 for a in (data, indices, indptr)):
                raise NotAssessed("Sparse storage arrays must be one-dimensional")
            if indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
                raise NotAssessed("Sparse indices and pointers must be integers")
            is_csr = encoding == "csr_matrix"
            major, minor = shape if is_csr else shape[::-1]
            if len(indptr) != major + 1 or len(indices) != len(data):
                raise NotAssessed("Sparse array lengths do not match shape and stored values")
            if int(indptr[0]) != 0 or int(indptr[-1]) != len(data):
                raise NotAssessed("Sparse pointers must start at zero and end at stored length")
            stats.update(storage=encoding, shape=list(shape), source_dtype=str(data.dtype))
            base_cost = minor * (data.dtype.itemsize + 12)
            if base_cost > allowance:
                raise NotAssessed("One sparse major-axis slice exceeds the working-array allowance")
            start = 0
            while start < major:
                stop = min(major, start + min(256, max(1, allowance // base_cost)))
                pointers = indptr[start:stop + 1]
                if np.any(pointers[1:] < pointers[:-1]) or np.any(pointers > len(data)):
                    raise NotAssessed("Sparse pointers are not monotone or are out of bounds")
                begin = int(pointers[0])
                # Sparse storage and duplicate-count workspace also count.
                while True:
                    count = int(pointers[-1]) - begin
                    estimate = ((stop - start) * base_cost
                                + count * (data.dtype.itemsize + indices.dtype.itemsize * 6 + 8)
                                + len(pointers) * 32)
                    if estimate <= allowance:
                        break
                    if stop - start == 1:
                        raise NotAssessed("One sparse slice's stored entries exceed the working-array allowance")
                    stop = start + max(1, (stop - start) // 2)
                    pointers = pointers[:stop - start + 1]
                end = int(pointers[-1])
                values = data[begin:end]
                ix = indices[begin:end]
                if np.any(ix < 0) or np.any(ix >= minor):
                    raise NotAssessed("Sparse indices are out of bounds")
                # int64 avoids unsigned subtraction or platform-width surprises.
                ptr = pointers.astype(np.int64) - begin
                for a, z in zip(ptr[:-1], ptr[1:]):
                    stats["duplicate_stored_entries"] += int(z - a - np.unique(ix[a:z]).size)
                _stored(stats, values)
                cls = sparse.csr_matrix if is_csr else sparse.csc_matrix
                block_shape = (stop-start, minor) if is_csr else (minor, stop-start)
                matrix = cls((values, ix, ptr), shape=block_shape)
                with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    dense = matrix.toarray()
                    _logical(stats, dense)
                stats["blocks"] += 1
                stats["max_estimated_block_bytes"] = max(stats["max_estimated_block_bytes"], estimate)
                start = stop
        else:
            raise NotAssessed("Unsupported X object")
    invalid = stats["logical_nonfinite_after_float32"] or stats["logical_negative_after_float32"]
    stats["verdict"] = "NUMERIC_FAIL" if invalid else "NUMERIC_PASS"
    stats["scope"] = "X numeric ingestion only; not a board validator, normalization check, or score"
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", help="Existing H5AD file, opened read-only")
    parser.add_argument("--buffer-mib", type=int, default=64,
                        help="Estimated temporary array allowance per block (default: 64)")
    args = parser.parse_args(argv)
    try:
        result = inspect(args.h5ad, args.buffer_mib)
        code = 1 if result["verdict"] == "NUMERIC_FAIL" else 0
    except NotAssessed as exc:
        result, code = {"verdict": "NOT_ASSESSED", "reason": str(exc)}, 2
    except (OSError, ValueError, TypeError, OverflowError, RuntimeError):
        # Library exceptions can contain private filenames; do not echo them.
        result, code = {"verdict": "NOT_ASSESSED", "reason": "File could not be read using supported numeric storage"}, 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
