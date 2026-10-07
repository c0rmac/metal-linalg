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
        metal_linalg_svd_policy_set(&sp);
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
        metal_linalg_svd_policy_set(&svd_measured);
        metal_linalg_eigh_policy_set(&measured);
        CHECK(metal_linalg_eigh_policy_get().gpu_max_n == measured.gpu_max_n, "policy not restored");
    }

    printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
