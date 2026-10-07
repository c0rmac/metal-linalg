#pragma once
// Divide and conquer for the symmetric tridiagonal eigenproblem and the
// bidiagonal SVD on the CPU's cores (divide_conquer.cpp): LAPACK's sstedc
// and sbdsdc, with their subproblems, merges and the merges' inner loops
// spread over threads.

#include <cstddef>
#include <cstdint>

namespace metal_linalg::detail {

// Where the GPU is idle during a solve, its largest merges' matrix products
// can run there (an MPS product took the top merge's from 23 ms to a few at
// 4096): `gemm` takes C = A B (+ C), column-major, with every operand inside
// memory added before, or returns false to leave it to the CPU. The solve
// adds and removes its page-aligned temporaries itself; the caller adds the
// output (Z, U, VT).
class GpuGemm {
public:
    virtual ~GpuGemm() = default;
    virtual void add(const float* base, size_t floats) = 0;
    virtual void remove(const float* base) = 0;
    virtual bool gemm(long m, long n, long k, const float* A, long lda, const float* B, long ldb, float* C, long ldc,
                      bool accumulate) = 0;
};

// T = Z diag(d) Z^T for the symmetric tridiagonal with diagonal d (n) and
// off-diagonal e (n - 1), as LAPACK's sstedc('I'), on up to `threads`
// threads: d gets the eigenvalues, ascending, e is destroyed, Z (n x n,
// column-major, ld ldz) the eigenvectors. Returns LAPACK's info, 0 on
// success.
long tridiagonal_eigensystem(uint32_t n, float* d, float* e, float* z, size_t ldz, unsigned threads,
                             GpuGemm* gpu = nullptr);

// B = U diag(d) VT for the upper bidiagonal with diagonal d (n) and
// superdiagonal e (n - 1), as LAPACK's sbdsdc('U', 'I'), on up to `threads`
// threads: d gets the singular values, descending, e is destroyed, U and VT
// (n x n, column-major, ld ldu and ldvt) the singular vectors. Returns
// LAPACK's info, 0 on success.
long bidiagonal_svd(uint32_t n, float* d, float* e, float* u, size_t ldu, float* vt, size_t ldvt, unsigned threads,
                    GpuGemm* gpu = nullptr);

} // namespace metal_linalg::detail
