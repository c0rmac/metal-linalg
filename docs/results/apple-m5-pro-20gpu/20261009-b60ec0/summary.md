# Apple M5 Pro, 20 GPU cores — submission 20261009-b60ec0

MacBook Pro (16-inch, M5 Pro) (Mac17,8), 18 CPU cores, 48 GB, macOS 27.0.1, MLX 0.32.1, metal-linalg b17351d. Measured 2026-10-09.

| decomposition | trustworthy | row for `kTuned[]` | minutes | report |
|---|---|---|---|---|
| qr | yes | `{"Apple M5 Pro", 20, 80, 768, 8,   448, 1448, 1, 0,   512, 0,   0},` | 4.7 | [qr/report.md](qr/report.md) |
| eigh | yes | `{"Apple M5 Pro", 20,   0, 256, 64, 64,   16, 8192, 1,   48, 8192, 1,   1024, 1024, 128, 2,   2, 64,   256,   64, 256,   1536, 16,   512,   96, 1024, 128,   96, 256, 256,   48},` | 12.0 | [eigh/report.md](eigh/report.md) |
| svd | yes | `{"Apple M5 Pro", 20,   256, 16,   256, 64, 64,   8, 4096, 1, 2048,   80, 8192, 1, 2048,   1024, 1024, 16, 2,   4, 80,   256,   80, 256,   768, 16,   512,   96, 1024, 64, kSvdNoLimit,   96, 1024, 64, kSvdNoLimit,   32},` | 21.6 | [svd/report.md](svd/report.md) |
