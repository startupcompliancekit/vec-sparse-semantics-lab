import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import anndata as ad
import h5py
import numpy as np
from scipy import sparse

from demo import cases, duplicate, generate, oracle
from sparse_preflight import inspect, NotAssessed


class NumericIngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, x, name="input.h5ad"):
        path = self.root / name
        ad.AnnData(X=x).write_h5ad(path)
        return path

    def assert_matches_direct(self, x, budget=1):
        path = self.write(x)
        before = hashlib.sha256(path.read_bytes()).digest()
        result = inspect(path, budget)
        expected = oracle(ad.read_h5ad(path).X)
        self.assertEqual(result["verdict"], expected["verdict"])
        self.assertEqual(result["logical_nonfinite_after_float32"], expected["nonfinite"])
        self.assertEqual(result["logical_negative_after_float32"], expected["negative"])
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).digest())
        return result

    def test_ten_real_h5ad_cases_match_direct_ingestion(self):
        for name, x in cases():
            with self.subTest(name=name):
                self.assert_matches_direct(x)

    def test_finite_stored_entries_overflow_only_when_combined(self):
        for layout in ("csr", "csc"):
            r = self.assert_matches_direct(duplicate([2e38, 2e38], "float32", layout))
            self.assertEqual(r["stored_nonfinite_after_float32"], 0)
            self.assertEqual(r["logical_nonfinite_after_float32"], 1)
            self.assertEqual(r["duplicate_stored_entries"], 1)

    def test_integer_wrap_is_negative_despite_positive_storage(self):
        r = self.assert_matches_direct(duplicate([100, 100], "int8"))
        self.assertEqual(r["stored_negative"], 0)
        self.assertEqual(r["logical_negative_after_float32"], 1)

    def test_cancellation_is_a_numeric_pass_not_a_normalization_claim(self):
        r = self.assert_matches_direct(duplicate([4e38, -4e38], "float64"))
        self.assertEqual(r["stored_nonfinite_after_float32"], 2)
        self.assertEqual(r["verdict"], "NUMERIC_PASS")

    def test_multiblock_randomized_noncanonical_sparse(self):
        rng = np.random.default_rng(713)
        for layout in ("csr", "csc"):
            for dtype in ("float32", "float64", "int8", "int64"):
                with self.subTest(layout=layout, dtype=dtype):
                    major, minor = 600, 400
                    data = rng.integers(-100, 100, size=major * 12).astype(dtype)
                    ix = rng.integers(0, minor, size=data.size, dtype=np.int32)
                    # Force duplicates and retain arbitrary source order.
                    ix[1::12] = ix[0::12]
                    ptr = np.arange(0, data.size + 1, 12, dtype=np.int32)
                    cls = sparse.csr_matrix if layout == "csr" else sparse.csc_matrix
                    shape = (major, minor) if layout == "csr" else (minor, major)
                    r = self.assert_matches_direct(cls((data, ix, ptr), shape=shape))
                    self.assertGreater(r["blocks"], 1)
                    self.assertGreaterEqual(r["duplicate_stored_entries"], major)
                    self.assertLessEqual(r["max_estimated_block_bytes"], 1024 * 1024)

    def test_multiblock_dense_and_nonfinite(self):
        x = np.zeros((600, 400), dtype=np.float64)
        x[0, 0], x[300, 10], x[599, 2] = np.nan, np.inf, -np.inf
        r = self.assert_matches_direct(x)
        self.assertGreater(r["blocks"], 1)
        self.assertEqual(r["logical_nonfinite_after_float32"], 3)

    def test_zero_stored_entries_are_a_valid_numeric_matrix(self):
        for cls in (sparse.csr_matrix, sparse.csc_matrix):
            self.assertEqual(self.assert_matches_direct(cls((3, 4), dtype=np.float32))["verdict"], "NUMERIC_PASS")

    def test_stored_order_is_not_changed_before_densification(self):
        # Each order has a different source-dtype rounding outcome.
        for values in ([1e20, -1e20, -1], [1e20, -1, -1e20]):
            self.assert_matches_direct(duplicate(values, "float32"))

    def test_one_unaffordable_slice_declines_instead_of_partial_pass(self):
        path = self.write(sparse.csr_matrix((1, 200000), dtype=np.float64))
        with self.assertRaises(NotAssessed):
            inspect(path, 1)

    def test_excess_duplicate_storage_declines(self):
        path = self.write(duplicate(np.ones(50000), "float64"))
        with self.assertRaises(NotAssessed):
            inspect(path, 1)

    def test_bad_indices_are_not_assessed(self):
        path = self.write(duplicate([1, 2], "float32"))
        with h5py.File(path, "r+") as f:
            f["X/indices"][0] = 999
        with self.assertRaises(NotAssessed):
            inspect(path)

    def test_bad_pointers_are_not_assessed(self):
        path = self.write(sparse.eye(4, format="csr", dtype=np.float32))
        with h5py.File(path, "r+") as f:
            f["X/indptr"][2] = 0
        with self.assertRaises(NotAssessed):
            inspect(path)

    def test_complex_values_are_not_silently_discarded(self):
        with self.assertRaises(NotAssessed):
            inspect(self.write(np.array([[1 + 2j]])))

    def test_cli_exit_codes_and_no_path_in_json(self):
        cases_to_check = [(np.array([[1.0]]), 0, "NUMERIC_PASS"),
                          (np.array([[-1.0]]), 1, "NUMERIC_FAIL"),
                          (np.array([[1j]]), 2, "NOT_ASSESSED")]
        script = str(Path(__file__).with_name("sparse_preflight.py"))
        for x, code, verdict in cases_to_check:
            path = self.write(x)
            p = subprocess.run([sys.executable, script, str(path)], capture_output=True, text=True)
            self.assertEqual(p.returncode, code, p.stderr)
            self.assertEqual(json.loads(p.stdout)["verdict"], verdict)
            self.assertNotIn(str(path), p.stdout + p.stderr)
        p = subprocess.run([sys.executable, script, str(self.root / "missing.h5ad")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertNotIn(str(self.root), p.stdout + p.stderr)

    def test_demo_runs_and_refuses_existing_directory(self):
        destination = self.root / "demonstration"
        result = generate(destination)
        self.assertEqual(len(result), 10)
        self.assertTrue(all(r["direct_ingestion_agrees"] and r["input_bytes_unchanged"] for r in result))
        with self.assertRaises(FileExistsError):
            generate(destination)


if __name__ == "__main__":
    unittest.main(verbosity=2)
