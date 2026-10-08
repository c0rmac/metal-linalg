# Apple M5 Pro, 20 GPU cores — submission 20261007-8633ac

MacBook Pro (16-inch, M5 Pro) (Mac17,8), 18 CPU cores, 48 GB, macOS 27.0.1, MLX 0.32.1, metal-linalg 48af83a+changes. Measured 2026-10-07.

| decomposition | trustworthy | row for `kTuned[]` | minutes | report |
|---|---|---|---|---|
| qr | yes | `{"Apple M5 Pro", 20, 192, 576, 8,   320, 6144, 1, 0,   362, 0,   0},` | 4.7 | [qr/report.md](qr/report.md) |

Re-analysed the same day (`tune_qr.py --reanalyse`), as 2.16.0 ships it: the
kernel crossover on k = min(M, N) rather than M, fitted on the shapes a GPU
kernel takes from the CPU, its batch split adopted; and topped up with the
`unblocked` backend on the tall large shapes (two more passes of 12 points,
appended to `qr/raw.csv`), which the grid had timed with the blocked QR alone.
The first analysis gave `{"Apple M5 Pro", 20, 64, 64, 16,   128, 20480, 1, 0,   512, 64,   0}` (1.091x regret; this one 1.038x).
