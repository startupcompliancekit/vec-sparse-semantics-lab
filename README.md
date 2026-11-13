# VEC Sparse Semantics Lab

**Finite sparse entries do not guarantee finite expression values.** If two stored entries refer to the same cell and gene, converting the matrix to a dense array adds them. That addition can overflow before a later dtype conversion.

This small companion for the [Virtual Embryo Challenge](https://virtualembryo.ai/challenge/community) checks the numeric expression matrix that results from the public scorer's ingestion order. It includes a runnable, synthetic H5AD walkthrough for participants preparing sparse predictions.

The checker opens one file read-only, visits bounded blocks of `.X`, and distinguishes **stored entries** from **logical values after densification and float32 conversion**. It does not modify, repair, upload, score, or download anything.

## Try the ten-case walkthrough

Python 3.11 or later is recommended. From a fresh checkout:

```sh
python3 -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-demo.txt
python demo.py --out demo-output
python -m unittest -v
```

The demo creates ten small, invented, AnnData-readable `.h5ad` files and `demo-output/results.json`. It refuses an existing destination directory. Use another new name to repeat it.

**These two-cell, three-gene fixtures are teaching examples, not biological data or valid Challenge submissions. Do not upload them.** They use only fabricated identifiers and values.

For every example, the demo reloads the file through AnnData and directly computes dense conversion followed by float32 conversion with NumPy/SciPy. It checks that the blockwise preflight agrees, and that the input's SHA-256 did not change. This verifies numeric ingestion, not the full official scorer or portal.

| Example | Stored-entry trap | Logical result | Numeric verdict |
|---|---|---|---|
| Dense safe | None | Finite, nonnegative | PASS |
| CSR safe duplicates | Two entries refer to one coordinate | `0.25 + 0.75 = 1` | PASS |
| CSC safe duplicates | Same issue, column-compressed storage | `0.25 + 0.75 = 1` | PASS |
| CSR float32 sum overflow | Each `2e38` entry is finite even after float32 casting | Their source-dtype sum is infinite | FAIL |
| CSC float64 cast overflow | Each `2e38` entry fits float32 | The float64 sum `4e38` overflows when cast | FAIL |
| CSR int8 wrap | Both stored entries are positive: `100, 100` | Source-dtype accumulation wraps to `-56` | FAIL |
| CSR float64 cancellation | Each of `4e38, -4e38` overflows if cast individually | Addition in float64 produces zero first | PASS |
| CSR rounding order | Casting entries before summation changes rounding | Float64 `1e8 + 1 - 1e8` becomes `1` | PASS |
| Dense cast overflow | Source float64 is finite | `4e38` becomes nonfinite float32 | FAIL |
| Dense negative | Finite does not mean nonnegative | A negative value remains negative | FAIL |

The cancellation and rounding cases demonstrate operation order. They do **not** recommend storing negative expression, using cancellation as preprocessing, or treating these invented values as log-normalized biology.

## Inspect a prediction

The checker itself needs only NumPy, SciPy and h5py:

```sh
python -m pip install -r requirements.txt
python sparse_preflight.py prediction.h5ad
python sparse_preflight.py prediction.h5ad --buffer-mib 32
```

For a concrete failure from the demo:

```sh
python sparse_preflight.py demo-output/csr-float32-sum-overflow.h5ad
```

The last command intentionally exits with status **1**. Its JSON includes:

```json
{
  "verdict": "NUMERIC_FAIL",
  "duplicate_stored_entries": 1,
  "stored_nonfinite_after_float32": 0,
  "logical_nonfinite_after_float32": 1,
  "logical_negative_after_float32": 0
}
```

This is the key difference: checking the stored values individually finds zero nonfinite entries, while the matrix the scorer receives contains one.

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | `NUMERIC_PASS` | Every checked logical `.X` value is finite and nonnegative after the stated conversion |
| 1 | `NUMERIC_FAIL` | At least one resulting logical `.X` value is nonfinite or negative |
| 2 | `NOT_ASSESSED` | File/layout/dtype/shape or the configured working-array allowance prevented a complete check |

A refused or incomplete inspection is never presented as a pass. JSON is printed to standard output; the checker has no output-file option and includes no input filename or absolute path in its result. If you save a report yourself, choose a separate destination, never the input H5AD file.

## Why this order matters for VEC

In the public scorer at [commit `cc3859a`](https://github.com/aristoteleo/veckit/blob/cc3859ab2a2305efc4544815d3f44f660efa77d4/score_h5ad.py#L51), `_as_float32` first densifies sparse input, then casts to float32 and rejects nonfinite values. `_prediction` also rejects negative expression. This companion follows that numeric order only. It does not import or run Challenge source code.

Sparse CSR and CSC arrays can legally store multiple values at the same logical coordinate. SciPy's dense conversion adds those values. Reading `X.data`, casting it, or inspecting its extrema does not necessarily describe the resulting logical matrix. Integer accumulation and floating-point summation both depend on the original dtype; changing dtype or sorting/coalescing first can change the answer.

The implementation reads consecutive CSR rows or CSC columns, preserving the original within-slice order. It densifies each block in its source dtype using SciPy, then casts that block. It does not call `sum_duplicates`, sort entries, or change the sparse input first. The dense case also preserves the cast-before-check order.

If this check fails, inspect the builder that produced the file. Constructing repeated coordinates deliberately, applying an elementwise transform directly to duplicate entries, using a narrow integer dtype, or emitting extreme values can each cause trouble. Preserve the original file and investigate the upstream calculation. This tool deliberately does not choose a repair: changing dtype or coalescing can change the predicted values.

## How this fits existing community tools

The catalog was surveyed on 13 November 2026, including all 15 accepted descriptions and 46 linked pending resources. This is a narrow complement to that work:

* [VEC Data Inspector](https://github.com/cadentann/virtual-embryo-community-kit/tree/main/modules/data_inspector) provides broad structural and stored-entry QC. Its documented sparse extrema refer to stored entries, without coalescing duplicate coordinates.
* The [float32 finiteness patch for vec-submit-check](https://github.com/gh-dv-openclaw/vec-submit-check/pull/1) checks stored-value cast finiteness and explicitly leaves noncanonical duplicate aggregation outside its scope.
* [VEC Diff](https://github.com/random-guy-05/vec-diff) compares two prediction files. [VEC Slim](https://github.com/random-guy-05/vec-slim) rewrites a compact prediction while checking semantic fingerprints.
* [Reliability Lab 18](https://github.com/random-guy-05/vec-reliability-lab/blob/main/resources/en/18-semantic-array-hash.md) teaches memory-layout-independent hashing and notes sparse duplicate handling as a separate production concern.

This project supplies a one-file, read-only numeric preflight and ten executable H5AD examples specifically for the duplicate-aggregation/conversion boundary. It does not replace those tools or claim to be the first general-purpose sparse-matrix inspector.

## Scope and limits

* Only `.X` is checked. No gene-panel/order check, cell-count rule, coordinates, AnnData metadata validation, normalization inference, eligibility decision or scientific metric is provided. A `NUMERIC_PASS` is **not** approval to submit. Use the current official format check and rules too.
* Ordinary dense datasets and CSR/CSC groups with local HDF5 datasets are supported. Complex data, unsupported encodings, external/virtual datasets and invalid sparse structure are declined. Unusual dtypes that the installed SciPy cannot densify are also declined.
* The default 64 MiB allowance limits an estimate of temporary array payload per block. It is **not a hard process-RSS limit**; Python, HDF5 caches and library overhead are additional. Blocks contain at most 256 major-axis slices. A single slice whose width or stored-entry count exceeds the allowance is declined; increasing it is an explicit user choice.
* The checker reports the behavior of the installed numeric libraries. Overflow and floating-point summation are not mathematical real arithmetic. Pins describe the tested versions; they do not guarantee behavior across every platform.
* No Challenge dataset, target answer, credential, account, API key or network connection is needed. Public examples and saved demonstration results contain invented data only.

## Verification

The test suite covers CSR and CSC, unsorted duplicates, source-dtype order, dense and sparse nonfinite values, signed integer wrap, multiple block boundaries, empty sparse storage, buffer refusal, invalid layouts, CLI exit codes, and byte-for-byte input preservation. Randomized noncanonical matrices are checked against full-array AnnData/SciPy ingestion, independently of the block traversal.

See [RESULTS.md](RESULTS.md) and [demo-results.json](demo-results.json) for the verified synthetic walkthrough. Run the tests yourself after dependency changes. No official portal result or biological finding is claimed.

MIT licensed. Independent community contribution by StartupComplianceKit.
