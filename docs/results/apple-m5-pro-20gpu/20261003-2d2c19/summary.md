# Apple M5 Pro, 20 GPU cores — submission 20261003-2d2c19

MacBook Pro (16-inch, M5 Pro) (Mac17,8), 18 CPU cores, 48 GB, macOS 27.0.1, MLX 0.32.1, metal-linalg 7b5b547+changes. Measured 2026-10-03.

| decomposition | trustworthy | row for `kTuned[]` | minutes | report |
|---|---|---|---|---|
| qr | yes | `{"Apple M5 Pro", 20, 512, 512, 16,   8, 10240, 1, 0,   1024, 4},` | 4.0 | [qr/report.md](qr/report.md) |
| eigh | yes | `{"Apple M5 Pro", 20,   0, 96, 0, 0,   48, 8192, 1,   0, 0, 1,   1536, 3072, 2, 1,   12, 64},` | 29.7 | [eigh/report.md](eigh/report.md) |
| svd | yes | `{"Apple M5 Pro", 20,   512, 32,   192, 64, 64,   8, 4096, 1,   1024, 2048, 1, 1},` | 45.6 | [svd/report.md](svd/report.md) |
