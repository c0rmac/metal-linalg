# Apple M5 Pro, 20 GPU cores — submission 20261003-c0878c

MacBook Pro (16-inch, M5 Pro) (Mac17,8), 18 CPU cores, 48 GB, macOS 27.0.1, MLX 0.32.1, metal-linalg eaa5554+changes. Measured 2026-10-03.

| decomposition | trustworthy | row for `kTuned[]` | minutes | report |
|---|---|---|---|---|
| qr | yes | `{"Apple M5 Pro", 20, 512, 512, 16,   192, 40960, 1, 128,   1024, 4},` | 4.2 | [qr/report.md](qr/report.md) |
| eigh | yes | `{"Apple M5 Pro", 20,   0, 96, 0, 0,   48, 8192, 1,   48, 16384, 1,   1536, 3072, 2, 1,   12, 64,   1024,   64, 1024},` | 28.3 | [eigh/report.md](eigh/report.md) |
| svd | yes | `{"Apple M5 Pro", 20,   512, 32,   192, 64, 64,   56, 16384, 1, 256,   56, 16384, 1, 56,   1024, 2048, 1, 1,   8, 80,   1024,   80, 1024},` | 56.2 | [svd/report.md](svd/report.md) |
