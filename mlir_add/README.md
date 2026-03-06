# mlir_add — Python-based MLIR elementwise add kernel

Builds an AMD GPU `vector_add` kernel entirely from Python using the
upstream MLIR Python bindings (`mlir.ir`, `mlir.dialects`, `mlir.passmanager`).

## Pipeline overview

```
Python API          mlir-translate-20       llc-20
  (build IR)  ──►  MLIR (llvm+rocdl)  ──►  LLVM IR  ──►  AMDGPU asm
      │                   ▲
      │  PassManager      │
      └───────────────────┘
```

Dialects used: `gpu`, `memref`, `arith`, `scf`, `rocdl`.

## Environment setup (inside docker)

The host machine does **not** have the MLIR toolchain. Everything runs
inside a ROCm docker container.

### 1. Launch the docker container

```bash
# from the host (uses ~/launch_docker.sh image)
docker run -d --privileged --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /home/carhuang:/dockerx -v /mnt/raid0:/raid0 \
  --name mlir_dev \
  rocm/atom:nightly_202601190317 sleep infinity
```

### 2. Install system MLIR / LLVM packages

```bash
docker exec mlir_dev bash -c "\
  apt-get update -qq && \
  apt-get install -y -qq mlir-20-tools libmlir-20-dev llvm-20"
```

This gives you `mlir-opt-20`, `mlir-translate-20`, and `llc-20`.

### 3. Build MLIR Python bindings from source

The upstream MLIR Python bindings are **not** available as a pip package.
They must be built from the LLVM source tree (takes ~6 min on a beefy
machine).  No dependency on `iree-base-compiler` or `flydsl`.

```bash
docker exec mlir_dev bash -c "\
  pip install nanobind pybind11 && \
  cd /tmp && \
  wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.2/llvm-project-20.1.2.src.tar.xz \
       -O llvm-src.tar.xz && \
  tar xf llvm-src.tar.xz && \
  mkdir -p mlir-build && cd mlir-build && \
  cmake -G Ninja /tmp/llvm-project-20.1.2.src/llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=AMDGPU \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=\$(which python3) \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_BUILD_TESTS=OFF && \
  ninja -j\$(nproc) MLIRPythonModules"
```

### 4. Register the bindings on the Python path

```bash
docker exec mlir_dev bash -c "\
  SITE=\$(python3 -c 'import site; print(site.getsitepackages()[0])') && \
  echo /tmp/mlir-build/tools/mlir/python_packages/mlir_core > \$SITE/mlir-python.pth"
```

Quick smoke test:

```bash
docker exec mlir_dev python3 -c "from mlir.ir import Context; print('OK')"
```

## Running

```bash
docker exec mlir_dev python3 /raid0/carhuang/repo/gcnasm/mlir_add/add_kernel.py [chip]
```

Default target is `gfx942`. Pass a different chip name to override:

```bash
docker exec mlir_dev python3 /raid0/carhuang/repo/gcnasm/mlir_add/add_kernel.py gfx90a
```

The script prints four stages of the kernel:

| Stage | Description |
|---|---|
| High-level MLIR | `gpu.func` with `memref`, `arith`, `scf`, `rocdl` |
| Lowered MLIR | Pure `llvm` dialect + `rocdl` intrinsics |
| LLVM IR | Standard LLVM IR with `amdgpu_kernel` calling convention |
| AMDGPU Assembly | Native ISA for the target chip |

## Key MLIR Python API patterns

```python
from mlir.ir import Context, Module, InsertionPoint, ...
from mlir.dialects import gpu, arith, memref, rocdl, scf
from mlir.passmanager import PassManager

# Create context and module
ctx = Context()
with ctx, Location.unknown():
    module = Module.create()

    # Build a gpu.module + gpu.func kernel
    with InsertionPoint(module.body):
        gpu_mod = gpu.GPUModuleOp("my_module")
        block = gpu_mod.bodyRegion.blocks.append()
        with InsertionPoint(block):
            kf = gpu.GPUFuncOp(TypeAttr.get(func_type))
            kf.operation.attributes["sym_name"] = StringAttr.get("my_kernel")
            kf.operation.attributes["gpu.kernel"] = UnitAttr.get()

            entry = kf.body.blocks.append(*func_type.inputs)
            with InsertionPoint(entry):
                # Use rocdl.workitem_id_x(i32), rocdl.workgroup_id_x(i32)
                # Use arith.*, memref.*, scf.* for the kernel body
                # Every op that returns a value needs .result
                gpu.ReturnOp([])

# Lower with PassManager
pm = PassManager.parse("builtin.module( gpu.module(...), ... )")
pm.run(module.operation)
```

## Lowering pass pipeline

```
builtin.module(
  gpu.module(
    convert-gpu-to-rocdl{chipset=gfx942},
    convert-scf-to-cf,
    expand-strided-metadata,
    convert-arith-to-llvm,
    convert-index-to-llvm,
    reconcile-unrealized-casts
  ),
  finalize-memref-to-llvm,
  convert-cf-to-llvm,
  reconcile-unrealized-casts
)
```

Note: `finalize-memref-to-llvm` and `convert-cf-to-llvm` are
`builtin.module`-level passes and must run at the outer level, not inside
`gpu.module(...)`.
