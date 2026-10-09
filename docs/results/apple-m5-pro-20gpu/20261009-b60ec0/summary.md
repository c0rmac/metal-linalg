# Apple M5 Pro, 20 GPU cores — submission 20261009-b60ec0

MacBook Pro (16-inch, M5 Pro) (Mac17,8), 18 CPU cores, 48 GB, macOS 27.0.1, MLX 0.32.1, metal-linalg b17351d. Measured 2026-10-09.

| decomposition | trustworthy | row for `kTuned[]` | minutes | report |
|---|---|---|---|---|
| qr | yes | `{"Apple M5 Pro", 20, 80, 768, 8,   448, 1448, 1, 0,   512, 0,   0},` | 4.7 | [qr/report.md](qr/report.md) |
| eigh | yes | `{"Apple M5 Pro", 20,   0, 256, 64, 64,   16, 8192, 1,   32, 8192, 1,   1024, 1024, 128, 2,   2, 64,   0,   48, 256,   1536, 16,   512,   96, 1024, 128,   64, 256, 512},` | 12.0 | [eigh/report.md](eigh/report.md) |
| svd | yes | `{"Apple M5 Pro", 20,   256, 16,   256, 64, 64,   8, 2048, 1, 1024,   32, 4096, 1, 1024,   1024, 1024, 16, 2,   4, 56,   0,   48, 256,   768, 16,   512,   64, 1024, 64, kSvdNoLimit,   64, 1024, 64, kSvdNoLimit},` | 21.6 | [svd/report.md](svd/report.md) |
