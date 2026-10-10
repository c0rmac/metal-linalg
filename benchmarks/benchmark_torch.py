"""The PyTorch comparison in README.md and python-torch/README.md:
metal-linalg-torch against torch.linalg on the CPU and on MPS, best of five.

    python benchmarks/benchmark_torch.py            # the README table
    python benchmarks/benchmark_torch.py --new      # 2.18.0's functions: Cholesky, LU, solve, inv, triangular
    python benchmarks/benchmark_torch.py --mps-ab   # MPS tensors in place against copied

Run it on an idle Mac. Some torch releases have no MPS kernels for eigh,
eigvalsh or svdvals; PYTORCH_ENABLE_MPS_FALLBACK=1 (set here unless already
set) runs those on the CPU, which is what a torch.linalg user on MPS gets.
metal-linalg-torch is given the same MPS tensors as torch's MPS path.

--mps-ab times this package alone on MPS tensors, used in place and copied
to the CPU and back (as before 2.14), alternating the two on each repeat.
"""

import argparse
import os
import sys
import time
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

import metal_linalg_torch as mlt  # noqa: E402

warnings.simplefilter("ignore")   # torch's notices that it fell back to the CPU


def symmetric(*shape):
    a = torch.randn(*shape)
    return 0.5 * (a + a.mT)


# name, input, torch.linalg call, metal-linalg-torch call
TABLE = [
    ("QR, 1024 × 128×128", lambda: torch.randn(1024, 128, 128), torch.linalg.qr, mlt.qr),
    ("SVD, 256 × 128×64", lambda: torch.randn(256, 128, 64),
     lambda a: torch.linalg.svd(a, full_matrices=False), mlt.svd),
    ("SVD, 4096 × 32×32", lambda: torch.randn(4096, 32, 32),
     lambda a: torch.linalg.svd(a, full_matrices=False), mlt.svd),
    ("eigh, 4096 × 16×16", lambda: symmetric(4096, 16, 16), torch.linalg.eigh, mlt.eigh),
    ("eigh, one 2048×2048", lambda: symmetric(2048, 2048), torch.linalg.eigh, mlt.eigh),
    ("SVD, one 4096×4096", lambda: torch.randn(4096, 4096),
     lambda a: torch.linalg.svd(a, full_matrices=False), mlt.svd),
    ("eigvalsh, one 4096×4096", lambda: symmetric(4096, 4096), torch.linalg.eigvalsh, mlt.eigvalsh),
    ("svdvals, one 4096×4096", lambda: torch.randn(4096, 4096), torch.linalg.svdvals, mlt.svdvals),
]

def spd(*shape):
    a = torch.randn(*shape) / shape[-1] ** 0.5
    return a @ a.mT + torch.eye(shape[-1])


def general(*shape):
    return torch.randn(*shape) / shape[-1] ** 0.5 + 2 * torch.eye(shape[-1])


def with_rhs(make, k):
    """A tuple (A, B), B with k right-hand sides."""
    def f():
        a = make()
        return a, torch.randn(*a.shape[:-1], k)
    return f


# 2.18.0's functions: name, input (a tensor, or a tuple of them), torch.linalg
# call, metal-linalg-torch call
NEW = [
    ("cholesky, one 4096×4096", lambda: spd(4096, 4096), torch.linalg.cholesky, mlt.cholesky),
    ("cholesky, 4 × 2048×2048", lambda: spd(4, 2048, 2048), torch.linalg.cholesky, mlt.cholesky),
    ("cholesky, 4096 × 32×32", lambda: spd(4096, 32, 32), torch.linalg.cholesky, mlt.cholesky),
    ("lu_factor, one 4096×4096", lambda: general(4096, 4096), torch.linalg.lu_factor, mlt.lu_factor),
    ("lu_factor, 4 × 2048×2048", lambda: general(4, 2048, 2048), torch.linalg.lu_factor, mlt.lu_factor),
    ("solve, one 4096×4096, 1 rhs", with_rhs(lambda: general(4096, 4096), 1),
     lambda t: torch.linalg.solve(*t), lambda t: mlt.solve(*t)),
    ("solve, one 2048×2048, 512 rhs", with_rhs(lambda: general(2048, 2048), 512),
     lambda t: torch.linalg.solve(*t), lambda t: mlt.solve(*t)),
    ("inv, one 4096×4096", lambda: general(4096, 4096), torch.linalg.inv, mlt.inv),
    ("inv, 1024 × 64×64", lambda: general(1024, 64, 64), torch.linalg.inv, mlt.inv),
    ("solve_triangular, 4096×4096, 4096 rhs", with_rhs(lambda: torch.tril(general(4096, 4096)), 4096),
     lambda t: torch.linalg.solve_triangular(*t, upper=False), lambda t: mlt.solve_triangular(*t, upper=False)),
]

# name, input, call: small batches (where the copies weighed most), large
# batches, mid-size matrices on the CPU path, and large single matrices.
AB = [
    ("eigh, 16 × 16×16", lambda: symmetric(16, 16, 16), mlt.eigh),
    ("QR, 16 × 48×16", lambda: torch.randn(16, 48, 16), mlt.qr),
    ("SVD, 1024 × 32×32", lambda: torch.randn(1024, 32, 32), mlt.svd),
    ("eigh, 16384 × 32×32", lambda: symmetric(16384, 32, 32), mlt.eigh),
    ("QR, 16384 × 64×32", lambda: torch.randn(16384, 64, 32), mlt.qr),
    ("svdvals, 4096 × 64×32", lambda: torch.randn(4096, 64, 32), mlt.svdvals),
    ("SVD, 256 × 128×64", lambda: torch.randn(256, 128, 64), mlt.svd),
    ("eigh, 64 × 256×256", lambda: symmetric(64, 256, 256), mlt.eigh),
    ("QR, 1024 × 128×128", lambda: torch.randn(1024, 128, 128), mlt.qr),
    ("eigh, one 1024×1024", lambda: symmetric(1024, 1024), mlt.eigh),
    ("eigh, one 2048×2048", lambda: symmetric(2048, 2048), mlt.eigh),
    ("SVD, one 2048×2048", lambda: torch.randn(2048, 2048), mlt.svd),
    ("QR, one 2048×2048", lambda: torch.randn(2048, 2048), mlt.qr),
    ("eigvalsh, one 4096×4096", lambda: symmetric(4096, 4096), mlt.eigvalsh),
    ("svdvals, one 4096×4096", lambda: torch.randn(4096, 4096), mlt.svdvals),
]


def once(f, x):
    """Milliseconds for f(x), with the MPS work queued before and by it done."""
    mps = (x[0] if isinstance(x, tuple) else x).device.type == "mps"
    if mps:
        torch.mps.synchronize()
    t = time.perf_counter()
    f(x)
    if mps:
        torch.mps.synchronize()
    return (time.perf_counter() - t) * 1e3


def best(f, x, repeats):
    once(f, x)   # warm-up: shader compilation, workspaces, the first-call probe
    return min(once(f, x) for _ in range(repeats))


def fmt(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 10:
        return f"{ms:.0f} ms"
    if ms >= 1:
        return f"{ms:.1f} ms"
    return f"{ms:.2f} ms"


def header():
    print(f"{mlt.device_name()} ({mlt.gpu_core_count()} GPU cores), torch {torch.__version__}, "
          f"metal-linalg-torch {mlt.__version__}, MPS tensors in place: {mlt.mps_in_place()}\n")


def table(repeats, rows=TABLE):
    header()
    print("| | torch, CPU | torch, MPS | metal-linalg-torch |")
    print("|---|---|---|---|")
    for name, make, ref, ours in rows:
        torch.manual_seed(0)
        a = make()
        m = tuple(t.to("mps") for t in a) if isinstance(a, tuple) else a.to("mps")
        cells = [best(ref, a, repeats), best(ref, m, repeats), best(ours, m, repeats)]
        print(f"| {name} | " + " | ".join(fmt(c) for c in cells) + " |", flush=True)


def mps_ab(repeats):
    header()
    if not mlt.mps_in_place():
        sys.exit("MPS tensors are not used in place on this Mac and torch; nothing to compare.")
    ops = sys.modules["metal_linalg_torch._ops"]

    def mode(in_place):
        ops._mps_in_place = in_place   # what METAL_LINALG_TORCH_MPS_COPY=1 sets, per call here

    print("| | copied | in place | copied / in place |")
    print("|---|---|---|---|")
    for name, make, f in AB:
        torch.manual_seed(0)
        m = make().to("mps")
        times = {True: [], False: []}
        for in_place in (True, False):   # warm-up of both paths
            mode(in_place)
            once(f, m)
        for _ in range(repeats):
            for in_place in (False, True):
                mode(in_place)
                times[in_place].append(once(f, m))
        mode(True)
        copied, in_place = min(times[False]), min(times[True])
        print(f"| {name} | {fmt(copied)} | {fmt(in_place)} | {copied / in_place:.2f}x |", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mps-ab", action="store_true", help="MPS tensors in place against copied")
    p.add_argument("--new", action="store_true", help="2.18.0's functions: Cholesky, LU, solve, inv, triangular")
    p.add_argument("--repeats", type=int, default=5, help="timed calls per cell (best kept)")
    args = p.parse_args()
    if not torch.backends.mps.is_available():
        sys.exit("needs MPS")
    if args.new:
        table(args.repeats, NEW)
    else:
        (mps_ab if args.mps_ab else table)(args.repeats)


if __name__ == "__main__":
    main()
