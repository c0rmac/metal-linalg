/* Tests of the C API (c_api.h), compiled as C: small decompositions with known
 * answers, the errors, and the routing and policy calls. The kernels
 * themselves are tested by test_core and the MLX suites. */
#include <metal_linalg/c_api.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        ++g_checks;                                         \
        if (!(cond)) {                                      \
            ++g_failures;                                   \
            printf("  FAIL  ");                             \
            printf(__VA_ARGS__);                            \
            printf("\n");                                   \
        }                                                   \
    } while (0)

static int near(float x, float y) { return fabsf(x - y) <= 1e-5f * (1.0f + fabsf(y)); }

int main(void) {
    printf("\nC API tests, %s (%u GPU cores)\n", metal_linalg_device_name(), metal_linalg_gpu_core_count());

    /* QR of two 3 x 2 matrices: Q R reproduces A, and R is upper triangular. */
    {
        const float a[12] = {1, 2, 3, 4, 5, 6,   2, 0, 0, 3, 1, 1};
        float q[12], r[8];
        CHECK(metal_linalg_qr(a, 2, 3, 2, q, r) == METAL_LINALG_OK, "qr: %s", metal_linalg_last_error());
        for (int b = 0; b < 2; ++b) {
            CHECK(r[b * 4 + 2] == 0.0f, "qr: R[%d] not upper triangular", b);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 2; ++j) {
                    float s = 0.0f;
                    for (int t = 0; t < 2; ++t) s += q[b * 6 + i * 2 + t] * r[b * 4 + t * 2 + j];
                    CHECK(near(s, a[b * 6 + i * 2 + j]), "qr: (QR)[%d][%d][%d] = %g, want %g", b, i, j, s, a[b * 6 + i * 2 + j]);
                }
        }
    }

    /* The modes: R alone is the reduced R, without q; the complete Q is
     * square and orthogonal, its first K columns the reduced Q's, and R has
     * zero rows below K. */
    {
        const float a[12] = {1, 2, 3, 4, 5, 6,   2, 0, 0, 3, 1, 1};
        float q[12], r[8], r_alone[8], qc[18], rc[12];
        CHECK(metal_linalg_qr(a, 2, 3, 2, q, r) == METAL_LINALG_OK, "qr: %s", metal_linalg_last_error());
        CHECK(metal_linalg_qr_with_mode(a, 2, 3, 2, METAL_LINALG_QR_R, NULL, r_alone) == METAL_LINALG_OK, "qr R alone: %s",
              metal_linalg_last_error());
        for (int i = 0; i < 8; ++i) CHECK(near(r_alone[i], r[i]), "qr R alone: r[%d] = %g, want %g", i, r_alone[i], r[i]);
        CHECK(metal_linalg_qr_with_mode(a, 2, 3, 2, METAL_LINALG_QR_COMPLETE, qc, rc) == METAL_LINALG_OK, "qr complete: %s",
              metal_linalg_last_error());
        for (int b = 0; b < 2; ++b) {
            CHECK(rc[b * 6 + 4] == 0.0f && rc[b * 6 + 5] == 0.0f, "qr complete: R[%d] row 2 not zero", b);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    float s = 0.0f, x = 0.0f;
                    for (int t = 0; t < 3; ++t) s += qc[b * 9 + t * 3 + i] * qc[b * 9 + t * 3 + j];
                    CHECK(fabsf(s - (i == j)) < 1e-5f, "qr complete: (Q^T Q)[%d][%d][%d] = %g", b, i, j, s);
                    if (j < 2) {
                        for (int t = 0; t < 3; ++t) x += qc[b * 9 + i * 3 + t] * rc[b * 6 + t * 2 + j];
                        CHECK(near(x, a[b * 6 + i * 2 + j]), "qr complete: (QR)[%d][%d][%d] = %g", b, i, j, x);
                        CHECK(near(qc[b * 9 + i * 3 + j], q[b * 6 + i * 2 + j]), "qr complete: Q[%d][%d][%d] = %g, want %g",
                              b, i, j, qc[b * 9 + i * 3 + j], q[b * 6 + i * 2 + j]);
                    }
                }
        }
        CHECK(metal_linalg_qr_with_mode(a, 2, 3, 2, METAL_LINALG_QR_COMPLETE, NULL, rc) == METAL_LINALG_INVALID_ARGUMENT,
              "qr complete without q not rejected");
        CHECK(metal_linalg_qr_with_mode(a, 2, 3, 2, (metal_linalg_qr_mode)7, q, r) == METAL_LINALG_INVALID_ARGUMENT,
              "qr mode 7 not rejected");
        /* Nothing to factor (no columns): the complete Q is the identity. */
        CHECK(metal_linalg_qr_with_mode(a, 2, 3, 0, METAL_LINALG_QR_COMPLETE, qc, NULL) == METAL_LINALG_OK,
              "qr complete of 3 x 0: %s", metal_linalg_last_error());
        for (int i = 0; i < 18; ++i)
            CHECK(qc[i] == (float)(i % 9 % 4 == 0), "qr complete of 3 x 0: Q[%d] = %g", i, qc[i]);
    }

    /* eigh of [[2, 1], [1, 2]]: eigenvalues 1 and 3, eigenvectors (1, -1) and (1, 1) / sqrt 2. */
    {
        const float a[4] = {2, 1, 1, 2};
        float w[2], v[4];
        uint32_t info = 0;
        CHECK(metal_linalg_eigh(a, 1, 2, 1, w, v, &info) == METAL_LINALG_OK, "eigh: %s", metal_linalg_last_error());
        CHECK(near(w[0], 1.0f) && near(w[1], 3.0f), "eigh: w = %g, %g, want 1, 3", w[0], w[1]);
        CHECK(near(fabsf(v[0]), 0.70710678f) && near(v[0], -v[2]), "eigh: first eigenvector (%g, %g)", v[0], v[2]);
        CHECK(METAL_LINALG_INFO_CONVERGED(info) && !METAL_LINALG_INFO_NONFINITE(info), "eigh: info %#x", info);

        float w_only[2];
        CHECK(metal_linalg_eigh(a, 1, 2, 0, w_only, NULL, NULL) == METAL_LINALG_OK, "eigvalsh: %s", metal_linalg_last_error());
        CHECK(near(w_only[0], 1.0f) && near(w_only[1], 3.0f), "eigvalsh: %g, %g", w_only[0], w_only[1]);
    }

    /* SVD of a 2 x 3 matrix with orthogonal rows of norms 3 and 2. */
    {
        const float a[6] = {3, 0, 0,   0, 0, 2};
        float u[4], s[2], vt[6];
        uint32_t info = 0;
        CHECK(metal_linalg_svd(a, 1, 2, 3, u, s, vt, &info) == METAL_LINALG_OK, "svd: %s", metal_linalg_last_error());
        CHECK(near(s[0], 3.0f) && near(s[1], 2.0f), "svd: s = %g, %g, want 3, 2", s[0], s[1]);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) {
                float x = 0.0f;
                for (int t = 0; t < 2; ++t) x += u[i * 2 + t] * s[t] * vt[t * 3 + j];
                CHECK(near(x, a[i * 3 + j]), "svd: (U S Vt)[%d][%d] = %g", i, j, x);
            }
        CHECK(METAL_LINALG_INFO_CONVERGED(info), "svd: info %#x", info);
    }

    /* Cholesky of [[4, 2], [2, 3]]: L = [[2, 0], [1, sqrt 2]]; the second
     * matrix, [[1, 2], [2, 1]], is not positive definite at its second pivot. */
    {
        const float a[8] = {4, 2, 2, 3,   1, 2, 2, 1};
        float l[8], u[8];
        uint32_t info[2] = {7, 7};
        CHECK(metal_linalg_cholesky(a, 2, 2, 0, l, info) == METAL_LINALG_OK, "cholesky: %s", metal_linalg_last_error());
        CHECK(near(l[0], 2.0f) && l[1] == 0.0f && near(l[2], 1.0f) && near(l[3], 1.41421356f),
              "cholesky: L = %g %g %g %g", l[0], l[1], l[2], l[3]);
        CHECK(info[0] == 0 && info[1] == 2, "cholesky: info %u %u, want 0 2", info[0], info[1]);
        CHECK(isnan(l[4]) && isnan(l[7]), "cholesky: the failed matrix not NaN");
        CHECK(metal_linalg_cholesky(a, 1, 2, 1, u, NULL) == METAL_LINALG_OK, "cholesky upper: %s",
              metal_linalg_last_error());
        CHECK(near(u[0], 2.0f) && near(u[1], 1.0f) && u[2] == 0.0f && near(u[3], 1.41421356f),
              "cholesky upper: U = %g %g %g %g", u[0], u[1], u[2], u[3]);
        CHECK(metal_linalg_cholesky(NULL, 1, 2, 0, l, NULL) == METAL_LINALG_INVALID_ARGUMENT, "cholesky(NULL) not rejected");
    }

    /* LU of [[1, 2], [3, 4]]: rows swapped (pivot 1), L = [[1, 0], [1/3, 1]],
     * U = [[3, 4], [0, 2/3]]; solve and inverse; a singular second matrix. */
    {
        const float a[8] = {1, 2, 3, 4,   1, 2, 2, 4};
        float lu[8], x[4], inv[8];
        uint32_t piv[4], info[2] = {7, 7};
        CHECK(metal_linalg_lu_factor(a, 2, 2, lu, piv, info) == METAL_LINALG_OK, "lu_factor: %s", metal_linalg_last_error());
        CHECK(piv[0] == 1 && piv[1] == 1, "lu_factor: pivots %u %u, want 1 1", piv[0], piv[1]);
        CHECK(near(lu[0], 3.0f) && near(lu[1], 4.0f) && near(lu[2], 1.0f / 3.0f) && near(lu[3], 2.0f / 3.0f),
              "lu_factor: LU = %g %g %g %g", lu[0], lu[1], lu[2], lu[3]);
        CHECK(info[0] == 0 && info[1] == 2, "lu_factor: info %u %u, want 0 2", info[0], info[1]);
        const float b[2] = {5, 6};   /* x = (-4, 4.5) */
        CHECK(metal_linalg_solve(a, 1, 2, b, 1, x, NULL) == METAL_LINALG_OK, "solve: %s", metal_linalg_last_error());
        CHECK(near(x[0], -4.0f) && near(x[1], 4.5f), "solve: x = %g %g", x[0], x[1]);
        CHECK(metal_linalg_inv(a, 2, 2, inv, info) == METAL_LINALG_OK, "inv: %s", metal_linalg_last_error());
        CHECK(near(inv[0], -2.0f) && near(inv[1], 1.0f) && near(inv[2], 1.5f) && near(inv[3], -0.5f),
              "inv: %g %g %g %g", inv[0], inv[1], inv[2], inv[3]);
        CHECK(isnan(inv[4]) && info[1] == 2, "inv: the singular matrix not NaN (info %u)", info[1]);
        CHECK(metal_linalg_solve(a, 1, 2, NULL, 1, x, NULL) == METAL_LINALG_INVALID_ARGUMENT, "solve(b NULL) not rejected");
        /* triangular: [[2, 99], [1, 4]] lower (99 never read), x = (1, 2) for b = (2, 9) */
        const float tl[4] = {2, 99, 1, 4}, tb[2] = {2, 9};
        CHECK(metal_linalg_solve_triangular(tl, 1, 2, tb, 1, 0, 0, x) == METAL_LINALG_OK, "solve_triangular: %s",
              metal_linalg_last_error());
        CHECK(near(x[0], 1.0f) && near(x[1], 2.0f), "solve_triangular: x = %g %g", x[0], x[1]);
        CHECK(strlen(metal_linalg_trsm_backend(64, 64, 1)) > 0, "trsm backend name");
        CHECK(strlen(metal_linalg_trsm_policy_source()) > 0, "trsm policy source");
        CHECK(strlen(metal_linalg_lu_backend(64, 4)) > 0, "lu backend name");
        const metal_linalg_lu_policy lm = metal_linalg_lu_policy_get();
        metal_linalg_lu_policy lp = lm;
        lp.gpu_min_n = 100;
        lp.gpu_max_batch = 0;
        metal_linalg_lu_policy_set(&lp);
        CHECK(strcmp(metal_linalg_lu_backend(100, 9), "blocked") == 0, "lu gpu_min_n = 100 routes to %s",
              metal_linalg_lu_backend(100, 9));
        CHECK(strcmp(metal_linalg_lu_policy_source(), "user") == 0, "lu source after set");
        metal_linalg_lu_policy_set(&lm);
    }

    /* A NaN gives NaN for its matrix and is reported in info. */
    {
        const float a[8] = {2, 1, 1, 2,   NAN, 0, 0, 1};
        float w[4];
        uint32_t info[2];
        CHECK(metal_linalg_eigh(a, 2, 2, 1, w, NULL, info) == METAL_LINALG_OK, "eigh NaN: %s", metal_linalg_last_error());
        CHECK(!isnan(w[0]) && !isnan(w[1]) && isnan(w[2]) && isnan(w[3]), "eigh NaN: w = %g %g %g %g", w[0], w[1], w[2], w[3]);
        CHECK(METAL_LINALG_INFO_NONFINITE(info[1]) && !METAL_LINALG_INFO_NONFINITE(info[0]), "eigh NaN: info %#x %#x", info[0], info[1]);
    }

    /* Errors: a status and a message, and the next call still works. */
    {
        float q[4], r[4], s[2], u[4];
        CHECK(metal_linalg_qr(NULL, 1, 2, 2, q, r) == METAL_LINALG_INVALID_ARGUMENT, "qr(NULL) not rejected");
        CHECK(strlen(metal_linalg_last_error()) > 0, "no error message");
        const float a[4] = {1, 0, 0, 1};
        CHECK(metal_linalg_svd(a, 1, 2, 2, u, s, NULL, NULL) == METAL_LINALG_INVALID_ARGUMENT, "svd(u, no vt) not rejected");
        CHECK(metal_linalg_qr(a, 1, 2, 2, q, r) == METAL_LINALG_OK, "qr after an error: %s", metal_linalg_last_error());
        CHECK(metal_linalg_qr(NULL, 0, 2, 2, NULL, NULL) == METAL_LINALG_OK, "an empty batch should need no buffers");
    }

    /* The calibration notice as a string: "" when current, else the message
     * the library would print; never NULL, and NULL input gives "". */
    {
        (void)metal_linalg_svd_policy_source();   /* resolves the policy */
        const char* msg = metal_linalg_calibration_message("SVD");
        const char* src = metal_linalg_svd_policy_source();
        CHECK(msg != NULL, "calibration_message returned NULL");
        CHECK((strncmp(src, "tuned:", 6) == 0) == (msg[0] == 0) || strncmp(src, "env:", 4) == 0 ||
              strcmp(src, "user") == 0, "calibration message '%s' for policy source '%s'", msg, src);
        CHECK(metal_linalg_calibration_message(NULL)[0] == 0, "calibration_message(NULL) not empty");
    }

    /* Routing and policies. */
    {
        CHECK(strlen(metal_linalg_qr_backend(64, 64, 10)) > 0, "qr backend name");
        CHECK(strlen(metal_linalg_svd_backend(64, 64, 10)) > 0, "svd backend name");
        CHECK(strlen(metal_linalg_eigh_policy_source()) > 0, "eigh policy source");

        const metal_linalg_eigh_policy measured = metal_linalg_eigh_policy_get();
        metal_linalg_eigh_policy p = measured;
        p.gpu_max_n = 0;   /* never the GPU */
        metal_linalg_eigh_policy_set(&p);
        CHECK(strcmp(metal_linalg_eigh_backend(8, 4096), "cpu") == 0, "gpu_max_n = 0 still routes to %s",
              metal_linalg_eigh_backend(8, 4096));
        CHECK(strcmp(metal_linalg_eigh_policy_source(), "user") == 0, "source after set: %s",
              metal_linalg_eigh_policy_source());
        p.gpu_max_n = METAL_LINALG_NO_LIMIT;
        p.gpu_min_batch_times_n = 0;
        p.gpu_min_batch = 1;
        metal_linalg_eigh_policy_set(&p);
        CHECK(strcmp(metal_linalg_eigh_backend(8, 4096), "cpu") != 0, "unlimited GPU still routes to cpu");
        /* The band backend for eigenvalues alone, from its threshold. */
        p.values_band_min_n = 2048;
        metal_linalg_eigh_policy_set(&p);
        CHECK(metal_linalg_eigh_policy_get().values_band_min_n == 2048, "values_band_min_n not set");
        CHECK(strcmp(metal_linalg_eigvalsh_backend(4096, 1), "band") == 0, "eigvalsh 4096 routes to %s",
              metal_linalg_eigvalsh_backend(4096, 1));
        p.values_band_min_n = 0;
        p.values_band_width = 32;
        metal_linalg_eigh_policy_set(&p);
        CHECK(metal_linalg_eigh_policy_get().values_band_width == 32, "eigh values_band_width not set");
        p.values_band_width = 0;
        metal_linalg_eigh_policy_set(&p);
        /* With eigenvectors, the band backend from band_min_n (within tridiag_max_batch), and
           tridiag_batch inside its window. */
        {
            metal_linalg_eigh_policy q = measured;
            q.gpu_max_n = 0;
            q.tridiag_min_n = 1024;
            q.tridiag_max_batch = 4;
            q.band_min_n = 3072;
            q.tridiag_batch_min_n = 96;
            q.tridiag_batch_max_n = 512;
            q.tridiag_batch_min_batch = 64;
            metal_linalg_eigh_policy_set(&q);
            CHECK(metal_linalg_eigh_policy_get().band_min_n == 3072, "band_min_n not set");
            CHECK(strcmp(metal_linalg_eigh_backend(4096, 1), "band") == 0, "band_min_n = 3072: N=4096 routes to %s",
                  metal_linalg_eigh_backend(4096, 1));
            CHECK(strcmp(metal_linalg_eigh_backend(2048, 1), "tridiag") == 0, "below band_min_n: N=2048 routes to %s",
                  metal_linalg_eigh_backend(2048, 1));
            CHECK(strcmp(metal_linalg_eigh_backend(256, 64), "tridiag_batch") == 0,
                  "tridiag_batch window: 64 x 256 routes to %s", metal_linalg_eigh_backend(256, 64));
            CHECK(strcmp(metal_linalg_eigvalsh_backend(256, 64), "tridiag_batch") != 0,
                  "the values window unset still routes eigvalsh to tridiag_batch");
            metal_linalg_eigh_policy_set(&p);
        }
        /* Eigenvalues alone: values_gpu_min_batch = 0 follows eigh; set, it decides apart. */
        p.values_gpu_min_batch = 0;
        metal_linalg_eigh_policy_set(&p);
        CHECK(strcmp(metal_linalg_eigvalsh_backend(8, 4096), "cpu") != 0, "unset values boundary did not follow eigh");
        p.values_gpu_max_n = 0;
        p.values_gpu_min_batch_times_n = 0;
        p.values_gpu_min_batch = 1;
        metal_linalg_eigh_policy_set(&p);
        CHECK(strcmp(metal_linalg_eigvalsh_backend(8, 4096), "cpu") == 0, "values_gpu_max_n = 0 still routes to %s",
              metal_linalg_eigvalsh_backend(8, 4096));
        CHECK(strcmp(metal_linalg_eigh_backend(8, 4096), "cpu") != 0, "the values boundary moved eigh");
        /* The tridiag backend instead of the CPU, from its threshold. */
        p = measured;
        p.gpu_max_n = 0;
        p.tridiag_min_n = 256;
        p.band_min_n = 0;   /* (a device's band threshold may be below 512) */
        metal_linalg_eigh_policy_set(&p);
        CHECK(strcmp(metal_linalg_eigh_backend(512, 1), "tridiag") == 0, "tridiag_min_n = 256: N=512 routes to %s",
              metal_linalg_eigh_backend(512, 1));
        CHECK(strcmp(metal_linalg_eigh_backend(128, 1), "cpu") == 0, "tridiag below its threshold");
        metal_linalg_eigh_policy_set(&measured);
        /* The bidiag SVD backend instead of the CPU, and svdvals' own threshold. */
        const metal_linalg_svd_policy svd_measured = metal_linalg_svd_policy_get();
        metal_linalg_svd_policy sp = svd_measured;
        sp.gpu_max_k = 0;
        sp.bidiag_min_k = 256;
        sp.band_min_k = 0;   /* (as the eigensolver's band threshold above) */
        sp.values_bidiag_min_k = 0;
        metal_linalg_svd_policy_set(&sp);
        CHECK(strcmp(metal_linalg_svd_backend(512, 512, 1), "bidiag") == 0, "bidiag_min_k = 256: 512x512 routes to %s",
              metal_linalg_svd_backend(512, 512, 1));
        CHECK(strcmp(metal_linalg_svdvals_backend(512, 512, 1), "cpu") == 0, "values_bidiag_min_k = 0 still bidiag");
        /* The golub_kahan window on the GPU: directly where the matrix fits, after a QR where it does not. */
        sp = svd_measured;
        sp.gpu_max_k = 64;
        sp.gpu_min_batch_times_k = 0;
        sp.gpu_min_batch = 1;
        sp.gpu_max_l = 0xFFFFFFFFu;
        sp.gk_min_k = 8;
        sp.gk_max_k = 48;
        metal_linalg_svd_policy_set(&sp);
        CHECK(metal_linalg_svd_policy_get().gk_max_k == 48, "gk_max_k not set");
        sp.gpu_big_batch_max_k = 80;
        sp.gpu_big_batch_min = 1024;
        metal_linalg_svd_policy_set(&sp);
        CHECK(metal_linalg_svd_policy_get().gpu_big_batch_max_k == 80 &&
              metal_linalg_svd_policy_get().gpu_big_batch_min == 1024, "SVD large-batch clause not set");
        sp.gpu_big_batch_min = 0;
        sp.values_band_min_k = 2048;
        metal_linalg_svd_policy_set(&sp);
        CHECK(metal_linalg_svd_policy_get().values_band_min_k == 2048, "values_band_min_k not set");
        sp.values_band_min_k = 0;
        sp.values_band_width = 8;
        metal_linalg_svd_policy_set(&sp);
        CHECK(metal_linalg_svd_policy_get().values_band_width == 8, "svd values_band_width not set");
        sp.values_band_width = 0;
        sp.band_min_k = 1536;
        sp.bidiag_min_k = 1024;
        metal_linalg_svd_policy_set(&sp);
        CHECK(metal_linalg_svd_policy_get().band_min_k == 1536, "band_min_k not set");
        CHECK(strcmp(metal_linalg_svd_backend(2048, 2048, 1), "band") == 0, "band_min_k: 2048x2048 routes to %s",
              metal_linalg_svd_backend(2048, 2048, 1));
        CHECK(strcmp(metal_linalg_svd_backend(1200, 1200, 1), "bidiag") == 0, "below band_min_k: 1200x1200 routes to %s",
              metal_linalg_svd_backend(1200, 1200, 1));
        sp.band_min_k = 0;
        metal_linalg_svd_policy_set(&sp);
        /* bidiag_batch inside its window, k in [64, 256] and l up to 512, from batch 32. */
        {
            metal_linalg_svd_policy q = sp;
            q.gpu_max_k = 0;
            q.gpu_big_batch_max_k = 0;
            q.bidiag_batch_min_k = 64;
            q.bidiag_batch_max_k = 256;
            q.bidiag_batch_min_batch = 32;
            q.bidiag_batch_max_l = 512;
            metal_linalg_svd_policy_set(&q);
            CHECK(metal_linalg_svd_policy_get().bidiag_batch_max_l == 512, "bidiag_batch_max_l not set");
            CHECK(strcmp(metal_linalg_svd_backend(256, 128, 32), "bidiag_batch") == 0,
                  "bidiag_batch window: 32 x 256x128 routes to %s", metal_linalg_svd_backend(256, 128, 32));
            CHECK(strcmp(metal_linalg_svd_backend(600, 128, 32), "bidiag_batch") != 0,
                  "above bidiag_batch_max_l still routes 600x128 to bidiag_batch");
            CHECK(strcmp(metal_linalg_svdvals_backend(256, 128, 32), "bidiag_batch") != 0,
                  "the values window unset still routes svdvals to bidiag_batch");
            metal_linalg_svd_policy_set(&sp);
        }
        CHECK(strcmp(metal_linalg_svd_backend(40, 24, 16), "golub_kahan") == 0, "gk window: 40x24 routes to %s",
              metal_linalg_svd_backend(40, 24, 16));
        CHECK(strcmp(metal_linalg_svd_backend(4000, 16, 16), "qr_golub_kahan") == 0, "gk window: 4000x16 routes to %s",
              metal_linalg_svd_backend(4000, 16, 16));
        CHECK(strcmp(metal_linalg_svd_backend(49, 49, 16), "jacobi") == 0, "outside the gk window: 49x49 routes to %s",
              metal_linalg_svd_backend(49, 49, 16));
        {
            float a[2 * 6 * 4], u[2 * 6 * 4], s[2 * 4], vt[2 * 4 * 4];
            uint32_t info[2];
            for (int i = 0; i < 2 * 6 * 4; ++i) a[i] = (float)((i * 7) % 11) - 5.0f;
            sp.gk_min_k = 1;
            metal_linalg_svd_policy_set(&sp);
            CHECK(strcmp(metal_linalg_svd_backend(6, 4, 2), "golub_kahan") == 0, "gk window: 6x4 routes to %s",
                  metal_linalg_svd_backend(6, 4, 2));
            CHECK(metal_linalg_svd(a, 2, 6, 4, u, s, vt, info) == METAL_LINALG_OK, "svd (golub_kahan): %s",
                  metal_linalg_last_error());
            float err = 0.0f;
            for (int b = 0; b < 2; ++b)
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 4; ++j) {
                        float r = 0.0f;
                        for (int t = 0; t < 4; ++t) r += u[b * 24 + i * 4 + t] * s[b * 4 + t] * vt[b * 16 + t * 4 + j];
                        err = fmaxf(err, fabsf(r - a[b * 24 + i * 4 + j]));
                    }
            CHECK(err < 1e-4f, "svd (golub_kahan) reconstruction error %g", err);
        }
        /* Cholesky: never the GPU, then the kernels by size. */
        {
            const metal_linalg_cholesky_policy cm = metal_linalg_cholesky_policy_get();
            metal_linalg_cholesky_policy cp = cm;
            cp.gpu_max_n = 0;
            cp.gpu_large_min_n = 0;
            metal_linalg_cholesky_policy_set(&cp);
            CHECK(strcmp(metal_linalg_cholesky_backend(16, 4096), "cpu") == 0, "cholesky gpu_max_n = 0 routes to %s",
                  metal_linalg_cholesky_backend(16, 4096));
            CHECK(strcmp(metal_linalg_cholesky_policy_source(), "user") == 0, "cholesky source after set: %s",
                  metal_linalg_cholesky_policy_source());
            cp.gpu_max_n = METAL_LINALG_NO_LIMIT;
            cp.gpu_min_batch_times_n = 0;
            cp.gpu_min_batch = 1;
            cp.gpu_min_n = 0;
            cp.simd_max_n = 32;
            cp.blocked_min_n = 1024;
            cp.blocked_max_batch = 2;
            metal_linalg_cholesky_policy_set(&cp);
            CHECK(metal_linalg_cholesky_policy_get().blocked_max_batch == 2, "cholesky blocked_max_batch not set");
            CHECK(strcmp(metal_linalg_cholesky_backend(24, 64), "simd") == 0, "cholesky 64 x 24 routes to %s",
                  metal_linalg_cholesky_backend(24, 64));
            CHECK(strcmp(metal_linalg_cholesky_backend(200, 64), "threadgroup") == 0, "cholesky 64 x 200 routes to %s",
                  metal_linalg_cholesky_backend(200, 64));
            CHECK(strcmp(metal_linalg_cholesky_backend(2048, 1), "blocked") == 0, "cholesky 1 x 2048 routes to %s",
                  metal_linalg_cholesky_backend(2048, 1));
            metal_linalg_cholesky_policy_set(&cm);
        }
        metal_linalg_svd_policy_set(&svd_measured);
        metal_linalg_eigh_policy_set(&measured);
        CHECK(metal_linalg_eigh_policy_get().gpu_max_n == measured.gpu_max_n, "policy not restored");
    }

    printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
