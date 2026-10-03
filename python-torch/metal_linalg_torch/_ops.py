"""The decompositions as torch custom operators: torch.ops.metal_linalg.*.

Each operator takes a float32 tensor [..., M, N] on the CPU or on MPS and
returns float32 tensors on the same device. Its fake (meta) implementation
gives torch.compile the output shapes without running anything, and its
autograd formula is the one torch.linalg uses for the same decomposition, so
the operators train, compile and export like built-in ones.

The library works on its own Metal command queue, on host memory: an MPS
tensor is copied to the CPU (which waits for MPS to finish the work queued
on it) and the results back, two copies at memory bandwidth around a
decomposition. A CPU tensor is used in place.
"""

import torch
from torch import Tensor

from . import _lib

_U32_MAX = 0xFFFFFFFF


def _dims(a, name):
    *lead, m, n = a.shape
    batch = 1
    for d in lead:
        batch *= d
    if max(batch, m, n) > _U32_MAX:
        raise ValueError(f"metal_linalg_torch.{name}: shape {tuple(a.shape)} is too large "
                         f"(each of batch, rows and columns must be below 2**32)")
    return lead, batch, m, n


def _host(a):
    """`a` as a contiguous float32 CPU tensor; no copy if it already is one."""
    return a.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _empty(shape):
    return torch.empty(shape, dtype=torch.float32)


def _back(device, *ts):
    return tuple(t if device.type == "cpu" else t.to(device) for t in ts)


# ---------------------------------------------------------------------------
# QR
# ---------------------------------------------------------------------------

@torch.library.custom_op("metal_linalg::qr", mutates_args=())
def qr(a: Tensor) -> tuple[Tensor, Tensor]:
    lead, batch, m, n = _dims(a, "qr")
    k = min(m, n)
    q, r = _empty((*lead, m, k)), _empty((*lead, k, n))
    if batch and k:
        h = _host(a)
        _lib.qr(h.data_ptr(), batch, m, n, q.data_ptr(), r.data_ptr())
    return _back(a.device, q, r)


@qr.register_fake
def _(a):
    *lead, m, n = a.shape
    k = min(m, n)
    return a.new_empty((*lead, m, k)), a.new_empty((*lead, k, n))


def _qr_square_grad(Q, R, gQ, gR):
    """A = Q R with R square (M >= N): gA = (gQ + Q copyltu(M)) R^-T,
    M = R gR^T - gQ^T Q, copyltu copying the lower triangle to the upper."""
    M = None
    if gR is not None:
        M = R @ gR.mT
    if gQ is not None:
        M = -(gQ.mT @ Q) if M is None else M - gQ.mT @ Q
    b = Q @ (M.tril() + M.tril(-1).mT)
    if gQ is not None:
        b = b + gQ
    # b R^-T: solve X R^T = b.
    return torch.linalg.solve_triangular(R.mT, b, upper=False, left=False)


def qr_grad(Q, R, gQ, gR):
    m, n = Q.shape[-2], R.shape[-1]
    if gQ is None and gR is None:
        return None
    if m >= n:
        return _qr_square_grad(Q, R, gQ, gR)
    # Wide: A = [A1 | A2] = Q [X | Y] with X square. Q and X depend on A1
    # alone and Y = Q^T A2, so A1's part is the square formula with
    # gQ + A2 gY^T, and A2's is Q gY.
    X, Y = R[..., :m], R[..., m:]
    gX = gY = None
    if gR is not None:
        gX, gY = gR[..., :m], gR[..., m:]
        A2gY = (Q @ Y) @ gY.mT
        gQ = A2gY if gQ is None else gQ + A2gY
    gAX = _qr_square_grad(Q, X, gQ, gX)
    gAY = Q @ gY if gY is not None else torch.zeros_like(Y)
    return torch.cat([gAX, gAY], dim=-1)


def _qr_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*output)


def _qr_backward(ctx, gQ, gR):
    Q, R = ctx.saved_tensors
    return qr_grad(Q, R, gQ, gR)


qr.register_autograd(_qr_backward, setup_context=_qr_setup)


# ---------------------------------------------------------------------------
# Symmetric eigendecomposition
# ---------------------------------------------------------------------------

def _square(a, name):
    if a.shape[-1] != a.shape[-2]:
        raise ValueError(f"metal_linalg_torch.{name}: expected square matrices, got shape {tuple(a.shape)}")


@torch.library.custom_op("metal_linalg::eigh", mutates_args=())
def eigh(a: Tensor, lower: bool) -> tuple[Tensor, Tensor]:
    _square(a, "eigh")
    lead, batch, n, _ = _dims(a, "eigh")
    w, v = _empty((*lead, n)), _empty((*lead, n, n))
    if batch and n:
        h = _host(a)
        _lib.eigh(h.data_ptr(), batch, n, lower, w.data_ptr(), v.data_ptr())
    return _back(a.device, w, v)


@eigh.register_fake
def _(a, lower):
    *lead, n, _ = a.shape
    return a.new_empty((*lead, n)), a.new_empty((*lead, n, n))


@torch.library.custom_op("metal_linalg::eigvalsh", mutates_args=())
def eigvalsh(a: Tensor, lower: bool) -> Tensor:
    _square(a, "eigvalsh")
    lead, batch, n, _ = _dims(a, "eigvalsh")
    w = _empty((*lead, n))
    if batch and n:
        h = _host(a)
        _lib.eigh(h.data_ptr(), batch, n, lower, w.data_ptr(), None)
    return _back(a.device, w)[0]


@eigvalsh.register_fake
def _(a, lower):
    *lead, n, _ = a.shape
    return a.new_empty((*lead, n))


def eigh_grad(w, V, gw, gV):
    """torch.linalg.eigh's: gA = V (diag(gw) + skew(V^T gV) / E) V^T,
    E_ij = w_j - w_i, skew(X) = (X - X^T) / 2."""
    if gw is None and gV is None:
        return None
    if gV is not None:
        X = V.mT @ gV
        X = 0.5 * (X - X.mT)
        E = w.unsqueeze(-2) - w.unsqueeze(-1)
        E.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        inner = X / E
        if gw is not None:
            inner = inner + torch.diag_embed(gw)
    else:
        inner = torch.diag_embed(gw)
    return V @ inner @ V.mT


def _eigh_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*output)


def _eigh_backward(ctx, gw, gV):
    w, V = ctx.saved_tensors
    return eigh_grad(w, V, gw, gV), None


eigh.register_autograd(_eigh_backward, setup_context=_eigh_setup)


def _eigvalsh_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.lower = inputs[1]
    ctx.save_for_backward(inputs[0])


def _eigvalsh_backward(ctx, gw):
    # The eigenvalues alone carry no eigenvectors, which the gradient needs:
    # computed here. metal_linalg_torch.eigvalsh avoids this second solve by
    # calling eigh when its input requires grad.
    (a,) = ctx.saved_tensors
    w, V = torch.ops.metal_linalg.eigh(a, ctx.lower)
    return eigh_grad(w, V, gw, None), None


eigvalsh.register_autograd(_eigvalsh_backward, setup_context=_eigvalsh_setup)


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

@torch.library.custom_op("metal_linalg::svd", mutates_args=())
def svd(a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    lead, batch, m, n = _dims(a, "svd")
    k = min(m, n)
    u, s, vt = _empty((*lead, m, k)), _empty((*lead, k)), _empty((*lead, k, n))
    if batch and k:
        h = _host(a)
        _lib.svd(h.data_ptr(), batch, m, n, u.data_ptr(), s.data_ptr(), vt.data_ptr())
    return _back(a.device, u, s, vt)


@svd.register_fake
def _(a):
    *lead, m, n = a.shape
    k = min(m, n)
    return a.new_empty((*lead, m, k)), a.new_empty((*lead, k)), a.new_empty((*lead, k, n))


@torch.library.custom_op("metal_linalg::svdvals", mutates_args=())
def svdvals(a: Tensor) -> Tensor:
    lead, batch, m, n = _dims(a, "svdvals")
    k = min(m, n)
    s = _empty((*lead, k))
    if batch and k:
        h = _host(a)
        _lib.svd(h.data_ptr(), batch, m, n, None, s.data_ptr(), None)
    return _back(a.device, s)[0]


@svdvals.register_fake
def _(a):
    *lead, m, n = a.shape
    return a.new_empty((*lead, min(m, n)))


def svd_grad(U, S, Vh, gU, gS, gVh):
    """torch.linalg.svd's (full_matrices=False), for real input:
    gA = U [(skew(U^T gU) S + S skew(V^T gV)) / E + diag(gS)] V^T
         + (I - U U^T) gU S^-1 V^T + U S^-1 gV^T (I - V V^T),
    E_ij = S_j^2 - S_i^2, skew(X) = X - X^T."""
    if gU is None and gS is None and gVh is None:
        return None
    m, n = U.shape[-2], Vh.shape[-1]
    k = S.shape[-1]
    if gU is None and gVh is None:
        return (U * gS.unsqueeze(-2)) @ Vh
    S2 = S * S
    E = S2.unsqueeze(-2) - S2.unsqueeze(-1)
    E.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    inner = None
    if gU is not None:
        UhgU = U.mT @ gU
        inner = (UhgU - UhgU.mT) * S.unsqueeze(-2)
    if gVh is not None:
        VhgV = Vh @ gVh.mT
        t = S.unsqueeze(-1) * (VhgV - VhgV.mT)
        inner = t if inner is None else inner + t
    inner = inner / E
    if gS is not None:
        inner = inner + torch.diag_embed(gS)
    if m > k and gU is not None:
        gA = U @ inner
        gUSinv = gU / S.unsqueeze(-2)
        gA = gA + gUSinv - U @ (U.mT @ gUSinv)
        return gA @ Vh
    if n > k and gVh is not None:
        gA = inner @ Vh
        SinvgVh = gVh / S.unsqueeze(-1)
        gA = gA + SinvgVh - (SinvgVh @ Vh.mT) @ Vh
        return U @ gA
    return U @ (inner @ Vh) if m >= n else (U @ inner) @ Vh


def _svd_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*output)


def _svd_backward(ctx, gU, gS, gVh):
    U, S, Vh = ctx.saved_tensors
    return svd_grad(U, S, Vh, gU, gS, gVh)


svd.register_autograd(_svd_backward, setup_context=_svd_setup)


def _svdvals_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(inputs[0])


def _svdvals_backward(ctx, gS):
    # As for eigvalsh: the singular vectors, computed here.
    (a,) = ctx.saved_tensors
    U, S, Vh = torch.ops.metal_linalg.svd(a)
    return svd_grad(U, S, Vh, None, gS, None)


svdvals.register_autograd(_svdvals_backward, setup_context=_svdvals_setup)
