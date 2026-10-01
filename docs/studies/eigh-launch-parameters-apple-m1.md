# `eigh` launch tuning on an Apple M1

Reference measurements behind the constants at the top of `src/eigh.mm` and
`eigh_block_jacobi.mm`. Regenerate with `./build/benchmark_eigh --tune` (or
`--tune 1|2|3|4|5` for one table). Every figure is the median of at least five
timed calls after two warm-ups, and the GPU clock ramps during a run, so
differences under ~15% are noise.

Device: Apple M1, 8 GPU cores, 32 KB threadgroup memory.

## What was learned

1. **Threads per matrix is the parameter that matters, and it depends on the
   batch, not just on N.** With one matrix per threadgroup and $N^2/2$ work
   items per phase, a lone matrix wants one item per thread up to the
   1024-thread cap (it has the core to itself). A large batch wants about six
   items per thread: smaller threadgroups co-reside on a core and hide one
   another's barriers, and at batch 256 the 1024-thread configuration is
   1.4-2.4x slower than the best one. The first rule tried, a flat four items
   per thread, was wrong in both regimes.

2. **Simd mode (one simdgroup per matrix) is not a win on this GPU.** It was
   designed for tiny N, where threadgroup barriers would dominate. But a
   32-thread threadgroup turns out to cost the same as a simdgroup, so once
   threadgroup mode gets the thread rule above the two modes are within noise
   for N ≤ 8 and threadgroup mode wins from N = 12 on (7x at N = 64, where a
   single simdgroup is simply too narrow for 2048 items per phase). The
   crossover `simd_max_n = 8` is a coin toss, kept because simd mode is the
   natural design for a GPU with cheaper threadgroup scheduling and a useful
   cross-check (the test suite runs with `EIGH_MODE=simd` as well).

3. **Matrices per threadgroup in simd mode barely matters** once there are at
   least a few threadgroups per core; the rule just avoids the degenerate
   case (8 matrices per threadgroup at batch 64 is one threadgroup per core).

4. **The interactivity watchdog is real and not purely a duration limit.** A
   single command buffer of 8 × 512² (4.4 s) survived while one of 1024 × 96²
   in simd mode (3.6 s, 128 threadgroups) and one of 256 × 128² on 32 threads
   (~9 s) were killed with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`.
   The host splits batches into chunks with a conservative cost model and
   never puts fewer matrices than cores in a chunk; the budget is
   `EIGH_CHUNK_MS` (default 750).

5. **Large N is memory-bound in the whole-matrix design.** One 512² matrix
   takes 1.0 s; eight concurrently take 4.4 s, not 1.0 s. Each round streams
   the whole matrix through device memory twice, ~$4N^3$ floats per sweep,
   and eight 2 MB working sets no longer fit in cache. This is what the block
   backend (table 4) fixes.

6. **The block backend crosses over near N = 100** (table 4, one pass). Below
   it the whole-matrix kernel wins by 2-7x: a block round is three launches
   with a handful of threadgroups each, and at N = 64 that is pure launch and
   barrier latency. From N = 192 block wins everywhere, by 3.5x at 256 and 9x
   at 512, where the whole-matrix kernel is one core streaming 19 GB and the
   block backend is eight cores doing tile products out of cache. This table
   put the crossover at 128; the two-pass routing study
   ([`eigh-routing-apple-m1.md`](eigh-routing-apple-m1.md)) put it at 96 and is what the policy's `block_min_n` follows.

7. **One inner sweep per subproblem is best** (table 5). More inner sweeps
   save at most one outer sweep and cost 10-100% more time, because the
   subproblem solve is the latency-bound part of a block round. One inner
   sweep on every block pair is a scalar cyclic sweep in a block-cyclic
   ordering, which is why the outer sweep count (6-10) matches the scalar
   kernel's.

8. **Neither backend beats Accelerate at large N on an M1**, and the block
   backend does not move the public CPU/GPU routing. At N = 512 the block
   backend needs 112 ms for one matrix against LAPACK's 18 ms, and it scales
   with batch about as LAPACK does. It is 9x better than what the GPU had,
   and it is the design that scales with GPU core count (a 32- or 64-core
   part runs the same launches with four to eight times the threadgroups in
   flight), but on eight cores it does not close a 6x gap.

## Table 4: whole-matrix (scalar) vs block Jacobi

```
     N   batch |  scalar ms   block ms | faster
    32       1 |      1.069      7.183 | scalar 6.72x
    32       4 |      1.433      7.811 | scalar 5.45x
    32      16 |      1.699      8.953 | scalar 5.27x
    32      64 |      3.015     12.391 | scalar 4.11x
    48       1 |      2.656     10.955 | scalar 4.12x
    48       4 |      3.071     12.047 | scalar 3.92x
    48      16 |      3.205     13.331 | scalar 4.16x
    48      64 |      7.819     19.611 | scalar 2.51x
    64       1 |      3.414     10.585 | scalar 3.10x
    64       4 |      3.577     11.701 | scalar 3.27x
    64      16 |      4.800     12.471 | scalar 2.60x
    64      64 |     14.874     19.092 | scalar 1.28x
    96       1 |      7.431     15.881 | scalar 2.14x
    96       4 |      7.719     14.680 | scalar 1.90x
    96      16 |     17.120     18.610 | scalar 1.09x
    96      64 |     60.779     44.922 | block  1.35x
   128       1 |     16.020     17.322 | scalar 1.08x
   128       4 |     15.415     16.032 | scalar 1.04x
   128      16 |     34.491     30.614 | block  1.13x
   128      64 |    129.052     91.371 | block  1.41x
   192       1 |     46.171     22.856 | block  2.02x
   192       4 |     53.633     32.270 | block  1.66x
   192      16 |    128.298     82.406 | block  1.56x
   192      64 |    447.837    278.485 | block  1.61x
   256       1 |    112.764     31.899 | block  3.53x
   256       4 |    145.404     46.610 | block  3.12x
   256      16 |    456.355    181.459 | block  2.51x
   384       1 |    467.550     69.849 | block  6.69x
   384       4 |    816.378    145.710 | block  5.60x
   384      16 |   3914.598    479.896 | block  8.16x
   512       1 |   1026.289    112.100 | block  9.16x
   512       4 |   2436.253    286.601 | block  8.50x
```

## Table 5: block backend, inner sweeps per subproblem

Median ms, with the number of outer sweeps to convergence in parentheses.

```
     N   batch |        inner=1        inner=2        inner=3        inner=4
    64       1 |    14.731 ( 6)    16.189 ( 5)    20.630 ( 5)    21.772 ( 5)
    64       8 |    15.551 ( 6)    19.236 ( 5)    18.839 ( 5)    24.755 ( 5)
   128       1 |    27.780 ( 7)    37.011 ( 7)    35.843 ( 7)    38.848 ( 7)
   128       8 |    31.106 ( 7)    37.227 ( 7)    37.253 ( 7)    40.178 ( 7)
   256       1 |    40.687 ( 8)    51.593 ( 8)    60.997 ( 8)    75.100 ( 8)
   256       8 |   102.822 ( 8)   121.146 ( 8)   151.017 ( 8)   176.100 ( 8)
   512       1 |   105.615 (10)   142.417 ( 9)   171.412 ( 9)   210.831 ( 9)
```

## Table 3: execution mode, each with its automatic parameters

Median ms per call. "faster" is the ratio of the slower to the faster mode.

```
     N   batch |    simd ms      tg ms | faster
     2      16 |      0.211      0.250 | simd 1.18x
     2      64 |      0.249      0.234 | tg   1.07x
     2    1024 |      0.297      0.307 | simd 1.03x
     2    8192 |      0.674      0.713 | simd 1.06x
     3      16 |      0.337      0.257 | tg   1.31x
     3      64 |      0.251      0.278 | simd 1.11x
     3    1024 |      0.476      0.639 | simd 1.34x
     3    8192 |      2.079      1.859 | tg   1.12x
     4      16 |      0.305      0.268 | tg   1.14x
     4      64 |      0.248      0.280 | simd 1.13x
     4    1024 |      0.535      0.534 | tg   1.00x
     4    8192 |      2.065      1.893 | tg   1.09x
     6      16 |      0.255      0.265 | simd 1.04x
     6      64 |      0.284      0.282 | tg   1.01x
     6    1024 |      0.649      0.749 | simd 1.15x
     6    8192 |      2.872      2.684 | tg   1.07x
     8      16 |      0.274      0.274 | simd 1.00x
     8      64 |      0.307      0.305 | tg   1.01x
     8    1024 |      0.689      0.717 | simd 1.04x
     8    8192 |      4.020      4.101 | simd 1.02x
    12      16 |      0.511      0.409 | tg   1.25x
    12      64 |      0.543      0.423 | tg   1.29x
    12    1024 |      3.050      2.678 | tg   1.14x
    12    8192 |     12.039     14.052 | simd 1.17x
    16      16 |      0.874      0.427 | tg   2.05x
    16      64 |      0.777      0.759 | tg   1.02x
    16    1024 |      3.387      3.581 | simd 1.06x
    16    8192 |     22.035     23.191 | simd 1.05x
    24      16 |      1.739      0.531 | tg   3.27x
    24      64 |      2.023      1.031 | tg   1.96x
    24    1024 |     10.634     10.353 | tg   1.03x
    24    8192 |     74.901     74.835 | tg   1.00x
    32      16 |      3.902      0.857 | tg   4.55x
    32      64 |      4.573      1.994 | tg   2.29x
    32    1024 |     29.203     23.244 | tg   1.26x
    32    8192 |    209.965    184.822 | tg   1.14x
    48      16 |     13.736      2.081 | tg   6.60x
    48      64 |     16.589      8.081 | tg   2.05x
    48    1024 |    162.770     85.265 | tg   1.91x
    64      16 |     31.757      4.160 | tg   7.63x
    64      64 |     42.946     13.074 | tg   3.28x
    64    1024 |    768.726    208.539 | tg   3.69x
```

## Table 1: simd mode, matrices per threadgroup

```
     N   batch |      g=1      g=2      g=4      g=8     g=16     auto
     4      16 |    0.272    0.275    0.246    0.362    0.292    0.352
     4      64 |    0.288    0.284    0.382    0.275    0.269    0.307
     4     256 |    0.313    0.461    0.331    0.500    0.328    0.448
     4    4096 |    1.414    1.516    1.467    1.429    1.238    1.356
     8      16 |    0.380    0.343    0.418    0.342    0.356    0.381
     8      64 |    0.368    0.493    0.370    0.367    0.447    0.389
     8     256 |    0.613    0.559    0.473    0.617    0.471    0.479
     8    4096 |    3.543    3.112    3.464    3.703    3.755    2.748
    16      16 |    0.842    0.990    1.105    1.415    1.369    1.138
    16      64 |    1.335    1.198    1.310    1.190    1.400    2.255
    16     256 |    3.319    2.526    2.423    2.259    1.950    1.997
    16    4096 |   12.549   13.309   13.940   13.281   12.701   12.704
    32      16 |    4.238    4.231    4.122    4.477    4.816    4.047
    32      64 |    4.399    4.823    4.545    4.665    4.970    4.588
    32     256 |    7.916    7.604    8.031    7.806    7.170    7.341
    32    4096 |  109.001  110.987  111.178  113.698  112.785  111.313
```

## Table 2: threadgroup mode, threads per matrix

The `auto` column is the rule in `eigh.mm`. `--` marks configurations skipped
because a single matrix would exceed the watchdog on its own.

```
     N   batch |     t=32     t=64    t=128    t=256    t=512   t=1024     auto
     8       1 |    0.304    0.285    0.380    0.381    0.414    0.364    0.423
     8      16 |    0.365    0.388    0.360    0.464    0.394    0.714    0.380
     8     256 |    0.624    0.610    1.056    1.394    2.644    3.338    0.418
    16       1 |    0.697    0.562    0.386    0.555    0.459    0.432    0.483
    16      16 |    1.197    0.709    0.636    0.641    0.641    1.090    0.632
    16     256 |    1.932    1.867    1.719    2.016    3.207    6.061    1.242
    24       1 |    1.552    1.107    0.888    0.813    0.766    0.774    0.762
    24      16 |    2.582    1.889    1.404    1.118    1.026    1.332    0.838
    24     256 |    4.698    3.402    4.322    5.622    6.203   13.562    3.245
    32       1 |    3.855    2.004    1.163    0.843    0.843    0.837    0.976
    32      16 |    5.066    2.414    1.629    1.187    1.107    1.799    1.458
    32     256 |    7.396    7.027    6.066    7.065    9.983   15.494    7.248
    48       1 |   12.989    7.122    3.907    2.294    1.815    1.513    1.725
    48      16 |   14.365    7.225    4.114    2.816    2.131    2.856    2.167
    48     256 |   50.847   34.477   25.950   25.607   33.151   43.561   22.564
    64       1 |   29.965   15.587    8.844    5.157    3.433    2.916    2.673
    64      16 |   31.774   17.910    9.508    5.567    4.138    5.067    4.154
    64     256 |  149.682  122.730   68.337   52.112   59.194   72.641   54.844
   128       1 |  259.382  136.214   69.770   36.620   26.071   15.234   15.766
   128      16 |  645.793  187.183   92.412   51.574   33.916   34.664   35.323
   128     256 | 8344.707 2540.753 1421.918  976.238  532.062  515.364  511.354
   256       1 |       --       --  532.139  260.600  156.930  108.570  113.342
   256      16 |       --       -- 1942.959 1079.167 1018.695  374.452  378.458
   512       1 |       --       --       --       -- 1811.007 1017.754 1026.077
```

## Earlier run of table 2, before the thread rule was fixed

Kept because it is the evidence for finding 1. The `auto` column here is the
*old* rule (a flat four items per thread), which is what these numbers argued
against. The sweep was killed by the watchdog at N = 128, batch 256, 32
threads, which is what motivated the thread-aware cost model.

```
     N   batch |     t=32     t=64    t=128    t=256    t=512   t=1024     auto
     8       1 |    0.302    0.278    0.293    0.303    0.287    0.308    0.281
     8      16 |    0.335    0.351    0.337    0.292    0.346    0.408    0.324
     8     256 |    0.377    0.435    0.553    0.900    1.988    3.331    0.468
    16       1 |    0.758    0.766    0.408    0.463    0.734    0.496    0.911
    16      16 |    1.340    0.923    0.609    0.565    0.557    1.091    1.137
    16     256 |    1.898    1.746    2.034    2.805    3.471    6.203    1.316
    24       1 |    1.508    1.083    0.919    0.952    0.754    0.885    1.291
    24      16 |    1.987    1.320    0.996    1.190    0.821    1.431    1.452
    24     256 |    4.399    3.819    3.286    4.588    6.336   10.651    3.384
    32       1 |    3.360    1.959    1.191    0.830    0.707    0.699    1.174
    32      16 |    4.033    2.446    1.728    1.085    1.128    1.948    2.095
    32     256 |    8.704    7.240    6.321    7.184    9.527   15.182    6.169
    48       1 |   13.583    7.237    3.896    2.625    1.875    1.599    2.187
    48      16 |   13.906    7.724    4.157    2.735    2.281    2.905    2.478
    48     256 |   32.361   28.702   24.016   25.422   31.934   43.711   27.187
    64       1 |   30.867   14.960    9.375    5.342    3.807    2.607    3.234
    64      16 |   32.539   17.563    9.889    5.751    4.196    5.545    4.614
    64     256 |  225.510  125.682   65.796   55.118   59.428   75.594   62.381
   128       1 |  274.676  138.744   71.189   38.718   23.028   16.312   15.992
   128      16 |  357.077  170.935   93.373   48.458   32.380   32.295   33.225
```
