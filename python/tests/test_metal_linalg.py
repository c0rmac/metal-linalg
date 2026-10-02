"""Tests of the Python package.

    python -m unittest discover -s python/tests -v

The C++ suites test the kernels in depth; these check that every function
reaches them from Python with the right shapes, values and errors.
"""

import math
import unittest

import mlx.core as mx

import metal_linalg as ml


def setUpModule():
    # CI builds wheels on virtual Macs, which may have no usable GPU; there
    # the package is only checked to install and import.
    if not mx.metal.is_available():
        raise unittest.SkipTest("no Metal GPU")


def max_abs(x):
    return mx.max(mx.abs(x)).item()


def mm(a, b):
    """a @ b on the CPU: some MLX releases multiply float32 on the GPU in
    reduced precision on some Macs (0.32.3 on the M5, to about 1e-2), which
    would hide the accuracy being checked."""
    with mx.stream(mx.cpu):
        return a @ b


def eye(n):
    return mx.eye(n)


class Decompositions(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)

    def test_qr(self):
        a = mx.random.normal((64, 32, 16))
        q, r = ml.qr(a)
        self.assertEqual(q.shape, (64, 32, 16))
        self.assertEqual(r.shape, (64, 16, 16))
        self.assertLess(max_abs(mm(q, r) - a), 1e-4)
        self.assertLess(max_abs(mm(q.swapaxes(-1, -2), q) - eye(16)), 1e-5)
        self.assertEqual(max_abs(mx.tril(r, -1)), 0.0)

    def test_eigh(self):
        a = mx.random.normal((32, 24, 24))
        s = a + a.swapaxes(-1, -2)
        w, v = ml.eigh(s)
        self.assertEqual(w.shape, (32, 24))
        self.assertEqual(v.shape, (32, 24, 24))
        self.assertLess(max_abs(mm(s, v) - v * w[..., None, :]) / max_abs(s), 1e-5)
        self.assertTrue(mx.all(w[..., 1:] >= w[..., :-1]).item())
        self.assertLess(max_abs(ml.eigvalsh(s) - w) / max_abs(w), 1e-5)

    def test_eigh_reads_one_triangle(self):
        a = mx.random.normal((4, 8, 8))
        s = a + a.swapaxes(-1, -2)
        junk_above = mx.tril(s) + mx.triu(mx.full((8, 8), 7.0), 1)
        junk_below = mx.triu(s) + mx.tril(mx.full((8, 8), -7.0), -1)
        self.assertLess(max_abs(ml.eigvalsh(junk_above, "L") - ml.eigvalsh(s)), 1e-4)
        self.assertLess(max_abs(ml.eigvalsh(junk_below, "U") - ml.eigvalsh(s)), 1e-4)

    def test_svd(self):
        a = mx.random.normal((16, 40, 24))
        u, s, vt = ml.svd(a)
        self.assertEqual((u.shape, s.shape, vt.shape), ((16, 40, 24), (16, 24), (16, 24, 24)))
        self.assertLess(max_abs(mm(u * s[..., None, :], vt) - a), 1e-4)
        self.assertTrue(mx.all(s[..., :-1] >= s[..., 1:]).item())
        self.assertLess(max_abs(ml.svdvals(a) - s) / max_abs(s), 1e-5)

    def test_wide_svd(self):
        a = mx.random.normal((8, 20, 50))
        u, s, vt = ml.svd(a)
        self.assertEqual((u.shape, s.shape, vt.shape), ((8, 20, 20), (8, 20), (8, 20, 50)))
        self.assertLess(max_abs(mm(u * s[..., None, :], vt) - a), 1e-4)

    def test_accepts_lists(self):
        q, r = ml.qr([[1.0, 2.0], [3.0, 4.0]])
        self.assertLess(max_abs(mm(q, r) - mx.array([[1.0, 2.0], [3.0, 4.0]])), 1e-5)

    def test_nan_stays_in_its_matrix(self):
        a = mx.random.normal((3, 6, 6))
        s = a + a.swapaxes(-1, -2)
        bad = mx.concatenate([s[:1], mx.full((1, 6, 6), math.nan), s[2:]])
        w, _ = ml.eigh(bad)
        self.assertTrue(mx.all(mx.isnan(w[1])).item())
        self.assertFalse(mx.any(mx.isnan(w[0])).item())
        self.assertFalse(mx.any(mx.isnan(w[2])).item())

    def test_errors(self):
        with self.assertRaises(ValueError):
            ml.eigh(mx.random.normal((4, 5)))      # not square
        # Complex input would otherwise lose its imaginary part silently:
        # this Hermitian matrix has eigenvalues 1 and 3, its real part 2 and 2.
        hermitian = mx.array([[2 + 0j, 1j], [-1j, 2 + 0j]])
        for fn in (ml.qr, ml.eigh, ml.eigvalsh, ml.svd, ml.svdvals):
            with self.assertRaisesRegex(ValueError, "Complex input"):
                fn(hermitian)


class Routing(unittest.TestCase):
    def test_device(self):
        self.assertIsInstance(ml.device_name(), str)
        self.assertGreaterEqual(ml.gpu_core_count(), 0)
        for source in (ml.qr_policy_source(), ml.eigh_policy_source(), ml.svd_policy_source()):
            self.assertTrue(source.split(":")[0] in ("tuned", "default", "env", "user"), source)

    def test_backend_names(self):
        self.assertIn(ml.qr_backend(64, 64, 100), {"cpu", "unblocked", "streaming_reduced"})
        self.assertIn(ml.eigh_backend(32, 4096), {"cpu", "simd", "threadgroup", "block"})
        self.assertIn(ml.svd_backend(1024, 64, 64),
                      {"cpu", "jacobi", "block_jacobi", "qr_jacobi", "qr_block_jacobi"})

    def test_policy_override(self):
        measured = ml.eigh_policy()
        try:
            ml.set_eigh_policy(gpu_min_batch=1, gpu_min_batch_times_n=0, gpu_max_n=2**32 - 1)
            self.assertEqual(ml.eigh_policy_source(), "user")
            self.assertNotEqual(ml.eigh_backend(512, 1), "cpu")
            ml.set_eigh_policy(gpu_max_n=0)
            self.assertEqual(ml.eigh_backend(8, 4096), "cpu")
            # eigenvalues alone: unset (values_gpu_min_batch = 0) follows eigh,
            # set decides on its own
            ml.set_eigh_policy(values_gpu_min_batch=0)
            self.assertEqual(ml.eigvalsh_backend(8, 4096), "cpu")
            ml.set_eigh_policy(values_gpu_max_n=64, values_gpu_min_batch_times_n=0, values_gpu_min_batch=1)
            self.assertNotEqual(ml.eigvalsh_backend(8, 4096), "cpu")
            self.assertEqual(ml.eigh_backend(8, 4096), "cpu")
        finally:
            ml.set_eigh_policy(measured)
        self.assertEqual(ml.eigh_policy(), measured)

    def test_unknown_policy_field(self):
        with self.assertRaises(KeyError):
            ml.set_svd_policy(not_a_field=1)


if __name__ == "__main__":
    unittest.main()
