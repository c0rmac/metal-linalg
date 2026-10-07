# Apple M5 Pro, 20 GPU cores — submission 20261007-246324

MacBook Pro (16-inch, M5 Pro) (Mac17,8), 18 CPU cores, 48 GB, macOS 27.0.1, MLX 0.32.1, metal-linalg aaa53c1. Measured 2026-10-07.

| decomposition | trustworthy | row for `kTuned[]` | minutes | report |
|---|---|---|---|---|
| qr | yes | `{"Apple M5 Pro", 20, 384, 384, 16,   128, 20480, 1, 16,   768, 4,   64},` | 4.3 | [qr/report.md](qr/report.md) |
| eigh | yes | `{"Apple M5 Pro", 20,   0, 96, 0, 0,   32, 16384, 1,   48, 16384, 1,   1024, 1024, 4, 2,   2, 64,   2048,   64, 2048,   2048, 16},` | 30.9 | [eigh/report.md](eigh/report.md) |
| svd | yes | `{"Apple M5 Pro", 20,   512, 32,   192, 64, 64,   56, 16384, 1, 256,   56, 16384, 1, 256,   1024, 1024, 4, 2,   8, 80,   1024,   80, 1024,   768, 16,   1024},` | 53.3 | [svd/report.md](svd/report.md) |
