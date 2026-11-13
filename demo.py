"""Build invented H5AD examples and compare with direct AnnData/SciPy ingestion."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from sparse_preflight import inspect


def duplicate(values, dtype, layout="csr"):
    """Two-by-three matrix; first major-axis slice repeats index zero."""
    values = np.array(values, dtype=dtype)
    cls = sparse.csr_matrix if layout == "csr" else sparse.csc_matrix
    ptr = [0, len(values), len(values)] if layout == "csr" else [0, len(values), len(values), len(values)]
    return cls((values, np.zeros(len(values), dtype=np.int32), np.array(ptr, dtype=np.int32)), shape=(2, 3))


def cases():
    return [
        ("dense-safe", np.array([[0, 1, 2], [3, 0, 1]], dtype=np.float32)),
        ("csr-safe-duplicates", duplicate([0.25, 0.75], "float32")),
        ("csc-safe-duplicates", duplicate([0.25, 0.75], "float64", "csc")),
        ("csr-float32-sum-overflow", duplicate([2e38, 2e38], "float32")),
        ("csc-float64-cast-overflow", duplicate([2e38, 2e38], "float64", "csc")),
        ("csr-int8-wrap", duplicate([100, 100], "int8")),
        ("csr-float64-cancellation", duplicate([4e38, -4e38], "float64")),
        ("csr-rounding-order", duplicate([1e8, 1, -1e8], "float64")),
        ("dense-cast-overflow", np.array([[4e38, 0, 0], [0, 0, 0]], dtype=np.float64)),
        ("dense-negative", np.array([[-1, 0, 0], [0, 0, 0]], dtype=np.float32)),
    ]


def oracle(x):
    """Direct full-array computation of the documented numeric operation order."""
    with np.errstate(over="ignore", invalid="ignore"):
        dense = x.toarray() if sparse.issparse(x) else np.asarray(x)
        converted = dense.astype(np.float32, copy=False)
    nonfinite = int(np.count_nonzero(~np.isfinite(converted)))
    negative = int(np.count_nonzero(converted < 0))
    first = float(converted.flat[0])
    return {"nonfinite": nonfinite, "negative": negative,
            "first_value": first if np.isfinite(first) else "nonfinite",
            "verdict": "NUMERIC_FAIL" if nonfinite or negative else "NUMERIC_PASS"}


def generate(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for name, x in cases():
        path = root / (name + ".h5ad")
        obj = ad.AnnData(X=x)
        obj.var_names = ["TOY_A", "TOY_B", "TOY_C"]
        obj.obs_names = ["toy_0", "toy_1"]
        obj.write_h5ad(path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        actual = inspect(path, buffer_mib=1)
        reference = oracle(ad.read_h5ad(path).X)
        unchanged = before == hashlib.sha256(path.read_bytes()).hexdigest()
        agrees = (actual["verdict"] == reference["verdict"]
                  and actual["logical_nonfinite_after_float32"] == reference["nonfinite"]
                  and actual["logical_negative_after_float32"] == reference["negative"])
        if not agrees or not unchanged:
            raise AssertionError("Demonstration disagreed with direct ingestion or changed its input")
        results.append({"case": name, "numeric_verdict": actual["verdict"],
                        "duplicate_stored_entries": actual["duplicate_stored_entries"],
                        "stored_nonfinite_after_float32": actual["stored_nonfinite_after_float32"],
                        "stored_negative": actual["stored_negative"],
                        "logical_nonfinite_after_float32": actual["logical_nonfinite_after_float32"],
                        "logical_negative_after_float32": actual["logical_negative_after_float32"],
                        "first_logical_value": reference["first_value"],
                        "direct_ingestion_agrees": agrees, "input_bytes_unchanged": unchanged})
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory; an existing directory is refused")
    args = parser.parse_args()
    for item in generate(args.out):
        print(item["case"], item["numeric_verdict"], "direct-check=OK", "unchanged=YES")
