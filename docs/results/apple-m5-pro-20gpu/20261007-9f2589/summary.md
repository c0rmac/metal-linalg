# Apple M5 Pro, 20 GPU cores — submission 20261007-9f2589

MacBook Pro (16-inch, M5 Pro) (Mac17,8), 18 CPU cores, 48 GB, macOS 27.0.1, MLX 0.32.1, metal-linalg 7d0a647. Measured 2026-10-07.

| decomposition | trustworthy | row for `kTuned[]` | minutes | report |
|---|---|---|---|---|
| qr | yes | `{"Apple M5 Pro", 20, 80, 768, 8,   448, 1448, 1, 0,   512, 0,   0},` | 4.9 | [qr/report.md](qr/report.md) |
| eigh | yes | `{"Apple M5 Pro", 20,   0, 96, 0, 0,   16, 8192, 1,   48, 16384, 1,   1024, 1024, 4, 2,   2, 64,   4096,   48, 256,   2048, 16},` | 31.9 | [eigh/report.md](eigh/report.md) |
| svd | yes | `{"Apple M5 Pro", 20,   256, 16,   192, 64, 64,   8, 4096, 1, 2048,   80, 16384, 1, 2048,   1024, 1024, 4, 2,   8, 80,   256,   80, 256,   768, 16,   1024},` | 55.7 | [svd/report.md](svd/report.md) |
