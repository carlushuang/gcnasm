# pdl_bench / amd -- Software PDL emulation on AMD MI355X (gfx950)

A HIP port of the [PDL microbenchmark](../README.md) for AMD CDNA GPUs, which
have **no Programmatic Dependent Launch hardware**. We emulate the PDL
producer/consumer handshake in software with a **single global semaphore**.

## The idea

NVIDIA PDL lets a producer kernel signal completion
(`cudaTriggerProgrammaticLaunchCompletion()`) so a back-to-back consumer can
race ahead through independent work and only block
(`cudaGridDependencySynchronize()`) right before it touches the producer's
output.

We rebuild that edge with a `uint32` device buffer, `pdl_semaphore`:

| Step | CUDA PDL | This emulation (HIP) |
|------|----------|----------------------|
| init | (hardware) | `pdl_semaphore = producer_grid_size` |
| producer signal | `cudaTriggerProgrammaticLaunchCompletion()` | `__syncthreads()`; lead thread `__hip_atomic_fetch_add(sem, -1, __ATOMIC_RELEASE, AGENT)` |
| consumer wait | `cudaGridDependencySynchronize()` | `__syncthreads()`; lead thread spins on `__hip_atomic_load(sem, RELAXED, AGENT)` until `0`, then acquire fence; `__syncthreads()` |

The lead thread is the workgroup leader (`threadIdx.x == 0`), so the semaphore
is decremented exactly once per producer workgroup and reaches `0` only after
**every** producer workgroup has published its writes.

Memory model: the producer's per-thread writes are made workgroup-visible by
`__syncthreads()` and published agent-wide by the **release** decrement; the
consumer's lead thread pairs that with an **acquire** fence after the spin, and
the trailing `__syncthreads()` propagates that ordering to the rest of the
workgroup before it reads the producer's data.

## Same stream: `hipExtAnyOrderLaunch` (works on RDNA, NOT on CDNA/gfx950)

The natural same-stream approach is `hipExtLaunchKernel(..., hipExtAnyOrderLaunch)`,
which asks the runtime to relax the in-order stream guarantee so the consumer can
start before the producer finishes — the closest analogue to NVIDIA PDL. The
benchmark includes this path (`run_anyorder_pair`).

**It is a no-op on MI355X.** `hip/hip_ext.h` states the flag is *"not supported on
AMD GFX9xx boards,"* and gfx950 is in that family. Measured directly with two
*independent* long kernels on one stream (grid=256, lots of spare CUs):

```
normal in-order (1 stream): 40.74 ms
anyOrder flag   (1 stream): 40.76 ms  (1.00x  -> still serialized)
2 streams (control)       : 20.69 ms  (1.97x  -> overlap is real & detectable)
```

So on gfx950 the consumer does **not** begin before the producer completes on the
same stream, regardless of the flag.

### Why: it's hardware, not the API

The flag is **honored** by the runtime — `clr/rocclr/device/rocm/rocvirtual.cpp`
unconditionally clears the AQL barrier bit
(`aqlHeader &= ~(1 << HSA_PACKET_HEADER_BARRIER)`) when `getAnyOrderLaunchFlag()`
is set; there is no gfx9 software gate. Clearing the bit removes the *software*
ordering, but the **CDNA (gfx9xx) command processor still serializes dispatches
within one hardware queue** — it won't launch the next grid from a queue until
the in-flight grid retires. So clearing the barrier bit is necessary but not
sufficient on CDNA. RDNA's dispatch front-end *can* hold multiple grids in flight
per queue, which is why the flag works there (see below). Concurrency on CDNA
must come from multiple HW queues = multiple streams.

### RDNA4 (gfx1201) -- the flag DOES work

Same benchmark + probe on an RX 9070 XT (gfx1201, 32 WGPs). Here
`hipExtAnyOrderLaunch` gives genuine same-stream overlap, and because it clears
the barrier bit on *every* dispatch the whole launch pipeline overlaps (not just
a pair), so it beats two streams at small grids:

```
two independent kernels, same stream, sweep:
grid=8     normal 22.95 ms | anyOrder  2.43 ms (9.44x) | 2-stream 11.85 ms (1.94x)
grid=16    normal 22.96 ms | anyOrder  4.74 ms (4.85x) | 2-stream 11.88 ms (1.93x)
grid=32    normal 22.94 ms | anyOrder  9.26 ms (2.48x) | 2-stream 11.92 ms (1.92x)
grid=64    normal 23.04 ms | anyOrder 18.30 ms (1.26x) | 2-stream 18.79 ms (1.23x)
grid=256   normal 80.42 ms | anyOrder 72.81 ms (1.10x) | 2-stream 72.93 ms (1.10x)

pdl_bench (producer/consumer + semaphore), N=8192 grid=32:
Baseline                    : 0.363 ms
Same stream + anyOrder + sem: 0.191 ms (1.90x)   <- works on gfx1201
Software PDL (2 streams)    : 0.203 ms (1.78x)
```

The win shrinks toward 1x as the grid saturates the WGPs (no spare capacity to
overlap into). The binary now prints an architecture-aware note: `[anyOrder
active...]` on RDNA, `[anyOrder no-op on gfx9xx...]` on CDNA.

**Takeaway:** for a PDL-style *same-stream* producer/consumer overlap,
`hipExtAnyOrderLaunch` + the semaphore is the right tool on **RDNA (gfx10/11/12)**;
on **CDNA (gfx9xx)** it cannot work and you must fall back to the two-stream path.

## Profiler confirmation (rocprofv3)

Wall-clock speedup could in principle be a measurement artifact, so the overlap
is double-confirmed with a kernel-dispatch timeline. `trace_test.hip` launches a
fixed number of kernels on **one stream** (mode 0 = normal, 1 = anyOrder);
`parse_trace.py` reads the rocprofv3 CSV and checks whether the dispatch
intervals overlap *on the same hardware queue*.

```bash
hipcc --offload-arch=gfx1201 -O3 -std=c++17 trace_test.hip -o trace_test
rocprofv3 --kernel-trace -f csv -d tr_normal -o k -- ./trace_test 0 4 8 50000
rocprofv3 --kernel-trace -f csv -d tr_any    -o k -- ./trace_test 1 4 8 50000
python3 parse_trace.py            # reads tr_normal/ and tr_any/
```

All 4 dispatches land on the **same queue (Queue 1, Stream 1)** in every run, so
this is genuinely same-stream behaviour — not multiple HW queues:

```
gfx1201 (RDNA4)            start_us   end_us
  normal   disp 1..4        0 / 687 / 1359 / 2033   -> back-to-back   concurrency 0.98x
  anyOrder disp 1..4        0 / 2.6 / 4.8 / 7.0      -> all overlap    concurrency 3.94x

gfx950 (CDNA4)
  normal   disp 1..4        0 / 1186 / 2371 / 3556   -> back-to-back   concurrency 1.00x
  anyOrder disp 1..4        0 / 1187 / 2371 / 3556   -> back-to-back   concurrency 1.00x  (no effect)
```

`concurrency = sum(kernel durations) / wall_span`. On gfx1201 the four kernels
start within 7 us of each other and run simultaneously (3.94x ~= 4 concurrent);
on gfx950 each starts exactly when the previous ends, with or without the flag.
This matches the clr source analysis: the runtime clears the AQL barrier bit on
both, but only RDNA's dispatch front-end acts on it.

## Why two streams (CDNA fallback)

Since same-stream overlap is unavailable, we run the **producer on stream A and
the consumer on stream B**; the semaphore re-introduces the data dependency that
the separate streams dropped. Per-pair events ping-pong so the single shared
semaphore is never reused while a pair is still in flight.

## Build & run

```bash
make
./pdl_bench_amd.exe [N] [tail_iters] [head_iters] [iterations]
```

Requires ROCm with `hipcc` and `--offload-arch=gfx950` (MI350/MI355). Tested
with ROCm 7.1.1 on MI355X.

## Results (MI355X, 256 CUs, ROCm 7.1.1)

| Config | grid (blk) | baseline avg | same-stream anyOrder | software-PDL (2 stream) | speedup |
|--------|-----------:|-------------:|---------------------:|------------------------:|:-------:|
| `65536 16384 16384` (heavy)  | 256  | 0.669 ms | 0.675 ms (0.99x) | 0.406 ms | **1.65x** |
| `65536 1024 1024`  (light)   | 256  | 0.045 ms | ~baseline        | 0.053 ms | 0.86x |
| `262144 16384 16384` (heavy) | 1024 | 0.787 ms | ~baseline        | 0.946 ms | 0.83x |
| `1048576 16384 16384`        | 4096 | 2.361 ms | ~baseline        | *skipped* | — |

The `anyOrder` column tracks baseline (~1.0x) everywhere — the flag never
overlaps on gfx950. Only the two-stream path produces real overlap.

Correctness (`consumer sees producer writes`) **PASS** in every runnable case.

## When it helps, when it doesn't

The software handshake is **not free** and only the *heavy + small-grid* case
wins. Three regimes:

1. **Sweet spot (grid << GPU capacity, heavy kernels) -> ~1.6x.** Each kernel
   underutilizes the 256 CUs, so running producer and consumer concurrently
   packs the idle CUs. The semaphore overhead is tiny next to 16k-FMA kernels.
2. **Light kernels -> slower (0.86x).** The reset kernel + two events +
   cross-stream waits + spin cost more than the few microseconds of overlap.
3. **GPU-saturating grids -> slower (0.83x).** When the two grids already fill
   the machine there is no spare capacity to overlap into; concurrency just adds
   the handshake cost. This is a genuine gap vs. hardware PDL, which still
   overlaps the producer tail with the consumer head even for full grids.

## Hard limit: co-residency (deadlock guard)

Because a resident consumer workgroup **spins** (holding its CU), producer and
consumer workgroups must be able to live on the GPU **at the same time**. If
they can't, resident spinning consumers starve the unscheduled producers and the
kernel **deadlocks** — there is no hardware scheduler to break it, unlike real
PDL.

The benchmark queries `hipOccupancyMaxActiveBlocksPerMultiprocessor` and
**skips** the PDL run (rather than hanging) when

```
2 * grid  >  CUs * min(producer_blk_per_CU, consumer_blk_per_CU)
```

On MI355X both kernels hit 8 blocks/CU -> budget 2048 blocks, so the largest
safe grid is ~1024 blocks/kernel. The `N=1048576` row above (4096 blocks) is
correctly skipped.

## Files

| File | Description |
|------|-------------|
| `pdl_bench.hip` | HIP source: reset/producer/consumer kernels, semaphore handshake, occupancy guard, correctness check, timing |
| `trace_test.hip` | Minimal N-kernels-on-one-stream launcher (normal vs anyOrder) for profiler confirmation |
| `parse_trace.py` | Parses rocprofv3 `--kernel-trace` CSV and reports per-queue dispatch overlap / concurrency factor |
| `Makefile` | Build rules (`--offload-arch=gfx950`; override `HIP_ARCH` for gfx1201) |
| `README.md` | This file |
