"""Tests of the PyTorch package.

    python -m unittest discover -s python-torch/tests -v

The C++ suites test the kernels in depth; these check that every function
reaches them from torch with the right shapes, values, devices, dtypes,
errors and gradients, and that the custom operators compile.
"""

import math
import os
import threading
import unittest
import warnings

import torch

import metal_linalg_torch as mlt

def _mps_works():
    """MPS as reported, and doing arithmetic correctly: on some virtual Macs
    (CI's macOS 14 runners with older torch) it is reported available but
    fails to allocate or multiplies wrongly."""
    if not torch.backends.mps.is_available() or os.environ.get("METAL_LINALG_TEST_NO_MPS"):
        return False
    try:
        a = torch.randn(64, 64)
        return torch.allclose((a.to("mps") @ a.to("mps")).cpu(), a @ a, rtol=1e-3, atol=1e-3)
    except RuntimeError:
        return False


MPS = _mps_works()
DEVICES = ["cpu"] + (["mps"] if MPS else [])


def setUpModule():
    # CI builds wheels on virtual Macs, which may have no usable GPU; there
    # the package is only checked to install and import.
    if not mlt.device_name():
        raise unittest.SkipTest("no Metal GPU")


def rel(x, y):
    x, y = x.detach().cpu().double(), y.detach().cpu().double()
    return ((x - y).abs().max() / y.abs().max().clamp_min(1e-30)).item()


def orth_err(Q):
    """max |Q^T Q - I| over the batch, for Q [..., M, K]."""
    Q = Q.detach().cpu().double()
    eye = torch.eye(Q.shape[-1], dtype=torch.float64)
    return (Q.mT @ Q - eye).abs().max().item() if Q.numel() else 0.0


def spd(*lead, n, seed=0):
    """Symmetric matrices with well-separated eigenvalues."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(*lead, n, n, generator=g, dtype=torch.float64))
    w = torch.linspace(1.0, 3.0, n, dtype=torch.float64) + 0.01 * torch.rand(*lead, n, generator=g, dtype=torch.float64)
    return (q * w.unsqueeze(-2)) @ q.mT


def well_conditioned(*shape, seed=0):
    """Matrices with distinct, well-separated singular values."""
    g = torch.Generator().manual_seed(seed)
    *lead, m, n = shape
    k = min(m, n)
    u, _ = torch.linalg.qr(torch.randn(*lead, m, k, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(*lead, n, k, generator=g, dtype=torch.float64))
    s = torch.linspace(3.0, 1.0, k, dtype=torch.float64)
    return (u * s.unsqueeze(-2)) @ v.mT


class Decompositions(unittest.TestCase):
    SHAPES = [(1, 1), (5, 3), (3, 5), (16, 16), (64, 32), (32, 64), (130, 70), (2048, 64)]

    def test_qr(self):
        for dev in DEVICES:
            for shape in self.SHAPES:
                for lead in ((), (3,), (2, 2)):
                    a = torch.randn(*lead, *shape, device=dev)
                    Q, R = mlt.qr(a)
                    k = min(shape)
                    self.assertEqual(Q.shape, (*lead, shape[0], k))
                    self.assertEqual(R.shape, (*lead, k, shape[1]))
                    self.assertEqual(Q.device, a.device)
                    self.assertEqual(Q.dtype, torch.float32)
                    self.assertLess(rel(Q @ R, a), 2e-5, (dev, shape, lead))
                    self.assertLess(orth_err(Q), 2e-5, (dev, shape, lead))
                    self.assertEqual(torch.tril(R.cpu(), -1).abs().max().item() if R.numel() else 0.0, 0.0)

    def test_eigh(self):
        for dev in DEVICES:
            for n in (1, 2, 7, 32, 100, 300):
                for lead in ((), (4,)):
                    s = spd(*lead, n=n).float().to(dev)
                    L, V = mlt.eigh(s)
                    self.assertEqual(L.shape, (*lead, n))
                    self.assertEqual(V.shape, (*lead, n, n))
                    self.assertEqual(L.device, s.device)
                    self.assertTrue(bool((L[..., 1:] >= L[..., :-1]).all()))
                    self.assertLess(rel((V * L.unsqueeze(-2)) @ V.mT, s), 2e-5, (dev, n))
                    self.assertLess(orth_err(V), 2e-5, (dev, n))
                    ref = torch.linalg.eigvalsh(s.cpu().double())
                    self.assertLess(rel(L, ref), 1e-5)
                    self.assertLess(rel(mlt.eigvalsh(s), ref), 1e-5)

    def test_eigh_uplo(self):
        s = spd(n=12)
        lower = torch.tril(s) + torch.triu(torch.full_like(s, 1e3), 1)    # garbage above
        upper = torch.triu(s) + torch.tril(torch.full_like(s, 1e3), -1)   # garbage below
        ref = torch.linalg.eigvalsh(s)
        self.assertLess(rel(mlt.eigh(lower, UPLO="L").eigenvalues, ref), 1e-5)
        self.assertLess(rel(mlt.eigh(upper, UPLO="U").eigenvalues, ref), 1e-5)
        self.assertLess(rel(mlt.eigvalsh(upper, UPLO="U"), ref), 1e-5)

    def test_svd(self):
        for dev in DEVICES:
            for shape in self.SHAPES:
                for lead in ((), (3,)):
                    a = torch.randn(*lead, *shape, device=dev)
                    U, S, Vh = mlt.svd(a)
                    k = min(shape)
                    self.assertEqual(U.shape, (*lead, shape[0], k))
                    self.assertEqual(S.shape, (*lead, k))
                    self.assertEqual(Vh.shape, (*lead, k, shape[1]))
                    self.assertEqual(U.device, a.device)
                    self.assertLess(rel((U * S.unsqueeze(-2)) @ Vh, a), 2e-5, (dev, shape))
                    self.assertLess(orth_err(U), 2e-5)
                    self.assertLess(orth_err(Vh.mT), 2e-5)
                    ref = torch.linalg.svdvals(a.cpu().double())
                    self.assertLess(rel(S, ref), 2e-5)
                    self.assertLess(rel(mlt.svdvals(a), ref), 2e-5)

    def test_large_single_matrices(self):
        # The GPU bidiagonalization and tridiagonalization backends, where the
        # Mac routes them (bidiag, tridiag), else the CPU: either way correct.
        a = torch.randn(1200, 1100)
        U, S, Vh = mlt.svd(a)
        self.assertLess(rel((U * S.unsqueeze(-2)) @ Vh, a), 2e-5)
        s = a[:1100].mT @ a[:1100]
        L, V = mlt.eigh(s)
        self.assertLess(rel((V * L.unsqueeze(-2)) @ V.mT, s), 2e-5)

    def test_result_types(self):
        a = torch.randn(6, 4)
        self.assertIsInstance(mlt.qr(a), torch.return_types.linalg_qr)
        self.assertIsInstance(mlt.eigh(a.mT @ a), torch.return_types.linalg_eigh)
        self.assertIsInstance(mlt.svd(a), torch.return_types.linalg_svd)
        r = mlt.svd(a)
        self.assertIs(r.U, r[0])
        self.assertIs(mlt.eigh(a.mT @ a).eigenvalues.dtype, torch.float32)

    def test_empty(self):
        for shape in ((0, 4, 3), (2, 0, 3), (2, 3, 0), (0, 0)):
            a = torch.zeros(shape)
            Q, R = mlt.qr(a)
            *lead, m, n = shape
            k = min(m, n)
            self.assertEqual(Q.shape, (*lead, m, k))
            self.assertEqual(R.shape, (*lead, k, n))
            U, S, Vh = mlt.svd(a)
            self.assertEqual(S.shape, (*lead, k))
            self.assertEqual(mlt.svdvals(a).shape, (*lead, k))
        L, V = mlt.eigh(torch.zeros(0, 5, 5))
        self.assertEqual(L.shape, (0, 5))
        self.assertEqual(mlt.eigh(torch.zeros(3, 0, 0)).eigenvectors.shape, (3, 0, 0))

    def test_non_finite_is_per_matrix(self):
        a = torch.randn(3, 8, 8)
        a = a + a.mT
        a[1, 2, 3] = float("nan")
        a[1, 3, 2] = float("nan")
        L, V = mlt.eigh(a)
        self.assertTrue(bool(torch.isnan(L[1]).all()))
        self.assertTrue(bool(torch.isfinite(L[0]).all() and torch.isfinite(L[2]).all()))
        S = mlt.svdvals(a)
        self.assertTrue(bool(torch.isnan(S[1]).all()) and bool(torch.isfinite(S[0]).all()))


class Arguments(unittest.TestCase):
    def test_dtypes(self):
        a64 = torch.randn(5, 4, dtype=torch.float64)
        Q, R = mlt.qr(a64)
        self.assertEqual(Q.dtype, torch.float32)
        self.assertLess(rel(Q.double() @ R.double(), a64), 2e-5)
        for dt in (torch.float16, torch.bfloat16, torch.int32, torch.int64):
            S = mlt.svdvals(torch.arange(12).reshape(4, 3).to(dt))
            self.assertEqual(S.dtype, torch.float32)
            self.assertLess(rel(S, torch.linalg.svdvals(torch.arange(12.).reshape(4, 3).double())), 1e-5)
        self.assertEqual(mlt.svdvals([[3.0, 0.0], [0.0, 4.0]]).tolist(), [4.0, 3.0])

    def test_rejects(self):
        with self.assertRaises(TypeError):
            mlt.svd(torch.randn(3, 3, dtype=torch.complex64))
        with self.assertRaises(ValueError):
            mlt.qr(torch.randn(5))
        with self.assertRaises(ValueError):
            mlt.eigh(torch.randn(3, 4))
        with self.assertRaises(ValueError):
            mlt.eigvalsh(torch.randn(3, 3), UPLO="X")
        with self.assertRaises(ValueError):
            mlt.qr(torch.randn(3, 3), mode="full")
        with self.assertRaises(NotImplementedError):
            mlt.qr(torch.randn(5, 3), mode="complete")
        with self.assertRaises(NotImplementedError):
            mlt.svd(torch.randn(5, 3), full_matrices=True)

    def test_modes(self):
        a = torch.randn(3, 5)
        Q, R = mlt.qr(a, mode="r")
        self.assertEqual(Q.numel(), 0)
        self.assertEqual(R.shape, (3, 5))
        Q2, R2 = mlt.qr(a, mode="complete")   # M <= N: the same as reduced
        self.assertEqual(Q2.shape, (3, 3))
        U, S, Vh = mlt.svd(torch.randn(4, 4), full_matrices=True)   # square: the same
        self.assertEqual(U.shape, (4, 4))

    def test_non_contiguous(self):
        a = torch.randn(6, 9, 7)[:, ::2, 1:].mT    # strided, transposed
        Q, R = mlt.qr(a)
        self.assertLess(rel(Q @ R, a), 2e-5)


class Routing(unittest.TestCase):
    def test_device(self):
        self.assertTrue(mlt.device_name().startswith("Apple"))
        self.assertGreaterEqual(mlt.gpu_core_count(), 0)

    def test_backends(self):
        self.assertIn(mlt.qr_backend(64, 32, 16), {"cpu", "unblocked", "streaming_reduced"})
        eigh = {"cpu", "simd", "threadgroup", "block", "tridiag", "ql", "band"}
        self.assertIn(mlt.eigh_backend(64, 16), eigh)
        self.assertIn(mlt.eigvalsh_backend(64, 16), eigh)
        svd = {"cpu", "jacobi", "block_jacobi", "qr_jacobi", "qr_block_jacobi", "bidiag",
               "golub_kahan", "qr_golub_kahan", "band"}
        self.assertIn(mlt.svd_backend(64, 32, 16), svd)
        self.assertIn(mlt.svdvals_backend(64, 32, 16), svd)

    def test_policies(self):
        for get, set_, source, field in ((mlt.qr_policy, mlt.set_qr_policy, mlt.qr_policy_source, "gpu_min_batch"),
                                         (mlt.eigh_policy, mlt.set_eigh_policy, mlt.eigh_policy_source, "gpu_min_batch"),
                                         (mlt.svd_policy, mlt.set_svd_policy, mlt.svd_policy_source, "bidiag_min_k")):
            before = get()
            try:
                set_(**{field: before[field] + 7})
                self.assertEqual(get()[field], before[field] + 7)
                self.assertEqual(source(), "user")
                with self.assertRaises(TypeError):
                    set_(no_such_field=1)
            finally:
                set_(before)
            self.assertEqual(get(), before)

    def test_eigvalsh_band_routing(self):
        before = mlt.eigh_policy()
        try:
            mlt.set_eigh_policy(values_band_min_n=1)
            self.assertEqual(mlt.eigvalsh_backend(300, 1), "band")
            a = torch.randn(300, 300)
            s = (a + a.T) / 2
            self.assertLess(rel(mlt.eigvalsh(s), torch.linalg.eigvalsh(s)), 2e-5)
        finally:
            mlt.set_eigh_policy(before)

    def test_policy_changes_routing(self):
        before = mlt.svd_policy()
        try:
            mlt.set_svd_policy(gpu_max_k=0, bidiag_min_k=1, values_bidiag_min_k=1)
            self.assertEqual(mlt.svd_backend(300, 80), "bidiag")
            a = torch.randn(300, 80)
            U, S, Vh = mlt.svd(a)
            self.assertLess(rel((U * S.unsqueeze(-2)) @ Vh, a), 2e-5)
            mlt.set_svd_policy(values_band_min_k=1)
            self.assertEqual(mlt.svdvals_backend(300, 80), "band")
            self.assertLess(rel(mlt.svdvals(a), torch.linalg.svdvals(a)), 2e-5)
            mlt.set_svd_policy(gpu_max_k=0, bidiag_min_k=0, values_bidiag_min_k=0, values_band_min_k=0)
            self.assertEqual(mlt.svd_backend(300, 80), "cpu")
            mlt.set_svd_policy(gpu_max_k=64, gpu_min_batch_times_k=0, gpu_min_batch=1, gk_min_k=8, gk_max_k=48)
            self.assertEqual(mlt.svd_backend(40, 24, 16), "golub_kahan")
            a = torch.randn(16, 40, 24)
            U, S, Vh = mlt.svd(a)
            self.assertLess(rel((U * S.unsqueeze(-2)) @ Vh, a), 2e-5)
        finally:
            mlt.set_svd_policy(before)

    def test_calibration_status(self):
        st = mlt.calibration_status()
        self.assertEqual(set(st), {"qr", "eigh", "svd"})
        for v in st.values():
            self.assertIn(v, {"current", "stale", "incomplete", "uncalibrated"})
        self.assertTrue(issubclass(mlt.CalibrationWarning, UserWarning))


class Gradients(unittest.TestCase):
    """Against torch.linalg's own gradients, in float64 there, with losses
    that do not depend on the signs of the vectors (which the two may choose
    differently)."""

    TOL = 2e-4

    def grads(self, a, loss_mlt, loss_ref):
        x = a.clone().float().requires_grad_(True)
        loss_mlt(x).backward()
        y = a.clone().double().requires_grad_(True)
        loss_ref(y).backward()
        return x.grad, y.grad

    def test_qr(self):
        for shape in ((7, 4), (4, 4), (4, 7), (3, 6, 3)):
            a = well_conditioned(*shape, seed=1)
            g = torch.Generator().manual_seed(2)
            *lead, m, n = shape
            k = min(m, n)
            W1, W2 = torch.randn(*lead, m, k, generator=g), torch.randn(*lead, k, n, generator=g)

            def loss(qr):
                def f(x):
                    Q, R = qr(x)
                    d = torch.sign(torch.diagonal(R[..., :k], dim1=-2, dim2=-1))   # unique: R's diagonal > 0
                    return ((Q * d.unsqueeze(-2)) * W1.to(Q)).sum() + ((d.unsqueeze(-1) * R) * W2.to(R)).sum()
                return f
            gm, gr = self.grads(a, loss(mlt.qr), loss(torch.linalg.qr))
            self.assertLess(rel(gm, gr), self.TOL, shape)
            gm, gr = self.grads(a, lambda x: (mlt.qr(x, mode="r").R.abs() * W2).sum(),
                                lambda x: (torch.linalg.qr(x).R.abs() * W2.double()).sum())
            self.assertLess(rel(gm, gr), self.TOL, shape)

    def test_eigh(self):
        for lead, n in (((), 6), ((3,), 5)):
            a = spd(*lead, n=n, seed=3)
            g = torch.Generator().manual_seed(4)
            c, W = torch.randn(n, generator=g), torch.randn(*lead, n, n, generator=g)

            def loss(eigh):
                def f(x):
                    L, V = eigh(x)
                    proj = (V * c.to(V)) @ V.mT          # sign-invariant
                    return (L * c.to(L)).sum() + (proj * W.to(V)).sum()
                return f
            gm, gr = self.grads(a, loss(mlt.eigh), loss(torch.linalg.eigh))
            self.assertLess(rel(gm, gr), self.TOL)
            gm, gr = self.grads(a, lambda x: (mlt.eigvalsh(x) * c).sum(),
                                lambda x: (torch.linalg.eigvalsh(x) * c.double()).sum())
            self.assertLess(rel(gm, gr), self.TOL)
            # The operator called directly, without the eigh shortcut: its own backward.
            gm, gr = self.grads(a, lambda x: (torch.ops.metal_linalg.eigvalsh(x, True) * c).sum(),
                                lambda x: (torch.linalg.eigvalsh(x) * c.double()).sum())
            self.assertLess(rel(gm, gr), self.TOL)

    def test_svd(self):
        for shape in ((7, 4), (4, 4), (4, 7), (2, 6, 3)):
            a = well_conditioned(*shape, seed=5)
            g = torch.Generator().manual_seed(6)
            *lead, m, n = shape
            k = min(m, n)
            c, W, Wu = torch.randn(k, generator=g), torch.randn(*lead, m, n, generator=g), torch.randn(*lead, m, m, generator=g)

            def loss(svd):
                def f(x):
                    U, S, Vh = svd(x, full_matrices=False)
                    paired = (U * c.to(U)) @ Vh            # sign-invariant pairs u_i v_i^T
                    return (S * c.to(S)).sum() + (paired * W.to(U)).sum() + ((U @ U.mT) * Wu.to(U)).sum()
                return f
            gm, gr = self.grads(a, loss(mlt.svd), loss(torch.linalg.svd))
            self.assertLess(rel(gm, gr), self.TOL, shape)
            gm, gr = self.grads(a, lambda x: (mlt.svdvals(x) * c).sum(),
                                lambda x: (torch.linalg.svdvals(x) * c.double()).sum())
            self.assertLess(rel(gm, gr), self.TOL, shape)
            gm, gr = self.grads(a, lambda x: (torch.ops.metal_linalg.svdvals(x) * c).sum(),
                                lambda x: (torch.linalg.svdvals(x) * c.double()).sum())
            self.assertLess(rel(gm, gr), self.TOL, shape)

    def test_float64_input_gets_float64_grad(self):
        a = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        mlt.svdvals(a).sum().backward()
        self.assertEqual(a.grad.dtype, torch.float64)

    @unittest.skipUnless(MPS, "no MPS")
    def test_on_mps(self):
        a = well_conditioned(3, 6, 4, seed=7)
        x = a.float().to("mps").requires_grad_(True)
        U, S, Vh = mlt.svd(x)
        (S.sum() + ((U @ Vh) ** 2).sum()).backward()
        self.assertEqual(x.grad.device.type, "mps")
        y = a.clone().requires_grad_(True)
        U, S, Vh = torch.linalg.svd(y, full_matrices=False)
        (S.sum() + ((U @ Vh) ** 2).sum()).backward()
        self.assertLess(rel(x.grad, y.grad), self.TOL)


class Operators(unittest.TestCase):
    def test_opcheck(self):
        a = torch.randn(3, 6, 4)
        s = spd(2, n=5).float()
        cases = [(torch.ops.metal_linalg.qr.default, (a,)),
                 (torch.ops.metal_linalg.qr.default, (a.mT.contiguous(),)),
                 (torch.ops.metal_linalg.eigh.default, (s, True)),
                 (torch.ops.metal_linalg.eigvalsh.default, (s, False)),
                 (torch.ops.metal_linalg.svd.default, (a,)),
                 (torch.ops.metal_linalg.svdvals.default, (a,))]
        for op, args in cases:
            torch.library.opcheck(op, args)
            grad_args = tuple(x.clone().requires_grad_(True) if isinstance(x, torch.Tensor) else x for x in args)
            torch.library.opcheck(op, grad_args)

    def test_compile(self):
        def f(x):
            L, V = mlt.eigh(x @ x.mT + 0.1 * torch.eye(x.shape[-1]))
            U, S, Vh = mlt.svd(x)
            R = mlt.qr(x).R
            return (L.sum() + S.sum() + R.diagonal(dim1=-2, dim2=-1).abs().sum() + mlt.svdvals(x).sum()
                    + mlt.eigh(x @ x.mT).eigenvalues.sum() + mlt.svd(x).S.sum())
        cf = torch.compile(f, backend="aot_eager", fullgraph=True)
        x = torch.randn(4, 6, 6)
        self.assertLess(abs(cf(x).item() - f(x).item()), 1e-3)
        xg = x.clone().requires_grad_(True)
        cf(xg).backward()
        xe = x.clone().requires_grad_(True)
        f(xe).backward()
        self.assertLess(rel(xg.grad, xe.grad), 1e-4)

    def test_threads(self):
        a = torch.randn(8, 40, 30)
        errors = []

        def work():
            try:
                for _ in range(5):
                    U, S, Vh = mlt.svd(a)
                    if rel((U * S.unsqueeze(-2)) @ Vh, a) > 2e-5:
                        errors.append("svd")
                    Q, R = mlt.qr(a)
                    if rel(Q @ R, a) > 2e-5:
                        errors.append("qr")
            except Exception as e:   # pragma: no cover
                errors.append(repr(e))
        ts = [threading.Thread(target=work) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
