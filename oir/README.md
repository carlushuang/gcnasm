# oir (opus-ir)

A typed, hardware-explicit schedule IR for AMD GPUs. It is structured like Cake IR (typed resources, a fixed op vocabulary, a verifier with stable diagnostic codes, canonical JSON with SHA-256 identity), but its vocabulary is [opus.hpp](https://github.com/ROCm/aiter/tree/main/csrc/include/opus): every resource and op lowers to exactly one opus construct, and every hardware fact the verifier checks is transcribed from an opus arch guard.

The full definition is in [docs/ir_spec.md](docs/ir_spec.md).

```python
from opus_ir import Access, Counter, DType, cdiv, kernel, verify


@kernel(name="gemm", target="gfx950", block_size=256)
def gemm(lm):
    k = lm.symbol("K", minimum=64, maximum=1 << 20, multiple_of=64)
    a = lm.param("A", (128, k), DType.BF16, Access.READ)
    s_a = lm.lds("s_a", (128, 64), DType.BF16, stages=3)
    mma = lm.mma("mma", "mfma", DType.BF16, DType.BF16, DType.FP32, wave=(16, 16, 32), expand=(4, 4, 2), tile=(2, 2, 1))
    ...
    with lm.loop("k", cdiv(k, 64), pipeline=lm.pipeline("kloop", 3)) as it:
        lm.wait(Counter.VM, outstanding=8)
        lm.sync_threads()
        ...


verify(gemm).require_ok()
print(gemm.canonical_json())
```

See [examples/gemm_gfx950.py](examples/gemm_gfx950.py) (3-stage `gmem::async_load` pipeline + tiled MFMA) and [examples/gemm_gfx1250_tdm.py](examples/gemm_gfx1250_tdm.py) (TDM producer wave + WMMA consumers with named-barrier handoff).

## Status

Done: IR schema, builder, LDS planner, target tables (gfx942, gfx950, gfx1250, gfx1200, gfx1201), and a verifier that covers structure, opus instruction admission, layout/register typing, and a wait-counter simulation of LDS hazards.

Not done: the opus C++ emitter, JIT/runtime, and an evidence harness. Lowering is specified op by op in the spec, but nothing generates code yet.

```bash
python3 -m pytest -q
python3 -m ruff check .
```

`src/opus_ir/diagnostics.py` is adapted from [cake-ir](https://github.com/huangzhilin-hzl/cake-ir) (Apache-2.0).
