"""The decompositions as torch custom operators: torch.ops.metal_linalg.*.

Each operator takes a float32 tensor [..., M, N] on the CPU or on MPS and
returns float32 tensors on the same device. Its fake (meta) implementation
gives torch.compile the output shapes without running anything, and its
autograd formula is the one torch.linalg uses for the same decomposition, so
the operators train, compile and export like built-in ones.

The library works on its own Metal command queue, on memory the CPU can
address. A CPU tensor is used in place. So is an MPS tensor: on Apple
Silicon its memory is a Metal buffer in shared storage, which the library's
kernels and its CPU path read and write directly, and its results go into
MPS tensors allocated for them. The call first waits for the work MPS has
queued (torch.mps.synchronize()), since that work may still be writing the
input. Where MPS memory is not in shared storage, or with
METAL_LINALG_TORCH_MPS_COPY=1, an MPS tensor is copied to the CPU and the
results back instead.
"""

import ctypes
import os

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


def _empty(shape, device="cpu"):
    return torch.empty(shape, dtype=torch.float32, device=device)


def _alloc(spec, device="cpu"):
    """An output: a shape (float32), or (shape, dtype)."""
    if len(spec) == 2 and isinstance(spec[1], torch.dtype):
        return torch.empty(spec[0], dtype=spec[1], device=device)
    return _empty(spec, device)


def _back(device, *ts):
    return tuple(t if device.type == "cpu" else t.to(device) for t in ts)


def _mps_address(t):
    """The CPU address of contiguous MPS tensor t's memory, or None if the
    library cannot use it in place. An MPS storage's data pointer is its
    id<MTLBuffer>."""
    return _lib.buffer_contents(t.untyped_storage().data_ptr(),
                                t.storage_offset() * t.element_size(), t.numel() * t.element_size())


def _probe_mps():
    """Whether MPS tensors are in memory the library can read and write in
    place: a CPU read of a tensor MPS wrote, and an MPS read of a CPU write."""
    try:
        if not torch.backends.mps.is_available():
            return False
        t = torch.arange(1, 65, dtype=torch.float32, device="mps") * 2
        torch.mps.synchronize()
        address = _mps_address(t)
        if not address:
            return False
        seen = (ctypes.c_float * 64).from_address(address)
        if list(seen) != [2.0 * i for i in range(1, 65)]:
            return False
        seen[0] = -1.0
        return t[0].item() == -1.0
    except Exception:   # pragma: no cover - a torch whose MPS storage is not a Metal buffer
        return False


_mps_in_place = None


def mps_in_place():
    """Whether MPS tensors are read and written in place (else copied)."""
    global _mps_in_place
    if _mps_in_place is None:
        _mps_in_place = (os.environ.get("METAL_LINALG_TORCH_MPS_COPY", "0") in ("", "0")
                         and _probe_mps())
    return _mps_in_place


def _run(a, shapes, call):
    """Runs the library on `a` into new outputs of `shapes` (float32, or
    (shape, dtype)) on a's device: call(input_address, *output_addresses) on
    memory the library reads and writes in place."""
    device = a.device
    if device.type == "mps" and mps_in_place():
        x = a.detach().to(dtype=torch.float32).contiguous()
        outs = tuple(_alloc(s, device) for s in shapes)
        # The work MPS has queued may still be writing x (or reading the
        # memory the outputs were given); the library uses its own queue.
        torch.mps.synchronize()
        addresses = [_mps_address(t) for t in (x, *outs)]
        if all(addresses):
            # The tensors' own Metal buffers, made known to the library for
            # the call, which its GPU backends then use rather than wrapping
            # the memory in new buffers (whose pages the GPU maps on first
            # use: about 1 ms for 64 MB)
            known = [(address, t.untyped_storage().data_ptr())
                     for address, t in zip(addresses, (x, *outs)) if t.storage_offset() == 0]
            known = [k for k in known if _lib.know_buffer(*k)]
            try:
                call(*addresses)
            finally:
                for k in known:
                    _lib.forget_buffer(*k)
            return outs
    x = _host(a)
    outs = tuple(_alloc(s) for s in shapes)
    call(x.data_ptr(), *(t.data_ptr() for t in outs))
    return _back(device, *outs)


# ---------------------------------------------------------------------------
# QR
# ---------------------------------------------------------------------------

def _qr_shapes(shape, mode):
    """Q's and R's shapes for `mode`, as torch.linalg.qr's: Q empty for "r",
    square for "complete" (R then [..., M, N])."""
    *lead, m, n = shape
    k = min(m, n)
    if mode == "reduced":
        return (*lead, m, k), (*lead, k, n)
    if mode == "r":
        return (0,), (*lead, k, n)
    if mode == "complete":
        return (*lead, m, m), (*lead, m, n)
    raise ValueError(f"metal_linalg::qr: mode must be 'reduced', 'r' or 'complete', got {mode!r}")


# The schema written out: torch 2.4's inference takes no str default.
@torch.library.custom_op("metal_linalg::qr", mutates_args=(),
                         schema='(Tensor a, str mode="reduced") -> (Tensor, Tensor)')
def qr(a: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    lead, batch, m, n = _dims(a, "qr")
    q_shape, r_shape = _qr_shapes(a.shape, mode)
    if not (batch and min(m, n)):
        Q, R = _empty(q_shape, a.device), _empty(r_shape, a.device)
        if mode == "complete" and batch and m:   # nothing to factor: Q = I
            Q.copy_(torch.eye(m, dtype=torch.float32, device=a.device).expand(q_shape))
        return Q, R
    if mode == "r":
        (R,) = _run(a, (r_shape,), lambda x, r: _lib.qr(x, batch, m, n, None, r, mode))
        return _empty(q_shape, a.device), R
    return _run(a, (q_shape, r_shape), lambda x, q, r: _lib.qr(x, batch, m, n, q, r, mode))


@qr.register_fake
def _(a, mode="reduced"):
    q_shape, r_shape = _qr_shapes(a.shape, mode)
    return a.new_empty(q_shape), a.new_empty(r_shape)


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
    ctx.mode = inputs[1] if len(inputs) > 1 else "reduced"
    ctx.save_for_backward(*output)


def _qr_backward(ctx, gQ, gR):
    Q, R = ctx.saved_tensors
    # As torch.linalg.qr's backward. (metal_linalg_torch.qr computes Q for
    # mode="r" when its input requires grad, so R alone is differentiable there.)
    if ctx.mode == "r" and gR is not None:
        raise RuntimeError("metal_linalg::qr: the derivative of QR depends on Q, which is not computed "
                           "when mode='r'; use mode='reduced' to differentiate")
    if ctx.mode == "complete" and Q.shape[-2] > R.shape[-1] and (gQ is not None or gR is not None):
        raise RuntimeError("metal_linalg::qr: the QR decomposition is not differentiable when "
                           "mode='complete' and M > N")
    return qr_grad(Q, R, gQ, gR), None


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
    shapes = ((*lead, n), (*lead, n, n))
    if not (batch and n):
        return tuple(_empty(s, a.device) for s in shapes)
    return _run(a, shapes, lambda x, w, v: _lib.eigh(x, batch, n, lower, w, v))


@eigh.register_fake
def _(a, lower):
    *lead, n, _ = a.shape
    return a.new_empty((*lead, n)), a.new_empty((*lead, n, n))


@torch.library.custom_op("metal_linalg::eigvalsh", mutates_args=())
def eigvalsh(a: Tensor, lower: bool) -> Tensor:
    _square(a, "eigvalsh")
    lead, batch, n, _ = _dims(a, "eigvalsh")
    if not (batch and n):
        return _empty((*lead, n), a.device)
    return _run(a, ((*lead, n),), lambda x, w: _lib.eigh(x, batch, n, lower, w, None))[0]


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
    shapes = ((*lead, m, k), (*lead, k), (*lead, k, n))
    if not (batch and k):
        return tuple(_empty(s, a.device) for s in shapes)
    return _run(a, shapes, lambda x, u, s, vt: _lib.svd(x, batch, m, n, u, s, vt))


@svd.register_fake
def _(a):
    *lead, m, n = a.shape
    k = min(m, n)
    return a.new_empty((*lead, m, k)), a.new_empty((*lead, k)), a.new_empty((*lead, k, n))


@torch.library.custom_op("metal_linalg::svdvals", mutates_args=())
def svdvals(a: Tensor) -> Tensor:
    lead, batch, m, n = _dims(a, "svdvals")
    k = min(m, n)
    if not (batch and k):
        return _empty((*lead, k), a.device)
    return _run(a, ((*lead, k),), lambda x, s: _lib.svd(x, batch, m, n, None, s, None))[0]


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


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------

# info is int32, as torch.linalg.cholesky_ex's: the library writes uint32,
# whose values are below 2**31.
@torch.library.custom_op("metal_linalg::cholesky", mutates_args=(),
                         schema="(Tensor a, bool upper=False) -> (Tensor, Tensor)")
def cholesky(a: Tensor, upper: bool = False) -> tuple[Tensor, Tensor]:
    _square(a, "cholesky")
    lead, batch, n, _ = _dims(a, "cholesky")
    if not (batch and n):
        return _empty((*lead, n, n), a.device), torch.zeros(lead, dtype=torch.int32, device=a.device)
    return _run(a, ((*lead, n, n), (tuple(lead), torch.int32)),
                lambda x, l, i: _lib.cholesky(x, batch, n, upper, l, i))


@cholesky.register_fake
def _(a, upper=False):
    *lead, n, _ = a.shape
    return a.new_empty((*lead, n, n)), a.new_empty(lead, dtype=torch.int32)


def cholesky_grad(L, gL, upper):
    """torch.linalg.cholesky's (Murray 2016, arXiv 1602.07527), for real
    input: with L lower and Phi(X) = tril(X), its diagonal halved,
    gA = L^-T sym(Phi(L^T gL)) L^-1, sym(X) = (X + X^T) / 2."""
    if gL is None:
        return None
    if upper:
        L, gL = L.mT, gL.mT
    gA = (L.mT @ gL).tril()
    gA = 0.5 * (gA + gA.tril(-1).mT)
    gA = torch.linalg.solve_triangular(L.mT, gA, upper=True, left=True)
    return torch.linalg.solve_triangular(L, gA, upper=False, left=False)


def _cholesky_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.upper = inputs[1] if len(inputs) > 1 else False
    ctx.save_for_backward(output[0])


def _cholesky_backward(ctx, gL, ginfo):
    (L,) = ctx.saved_tensors
    return cholesky_grad(L, gL, ctx.upper), None


cholesky.register_autograd(_cholesky_backward, setup_context=_cholesky_setup)
