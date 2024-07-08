# matrix core
this folder contains a simple example to showcase the usage of AMD's matrix core. we use `__builtin_amdgcn_mfma_f32_32x32x8f16` instruction with single wave to show case the layout requirement of this instruction. this instruction is supported for MI100/200/300 serious of AMDGPU (change arch number inside rebuild.sh to build on proper arch)

## 3 different way to play sith matrix core layout
```
matrix_core_kernel_standard:
    standard layout, each thread holding data along the column of C matrix
matrix_core_kernel_swap_a_b
    swap A/B pointer, each thread holding data along the row of C matrix
matrix_core_kernel_swap_swb
    swap A/B pointer plus swizzle B, each thread holding data along the row of C matrix
```