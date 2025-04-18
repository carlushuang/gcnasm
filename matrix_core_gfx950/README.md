# matrix core
this folder contains a simple example to showcase the usage of AMD's matrix core. we use `__builtin_amdgcn_mfma_f32_32x32x16_f16` instruction with single wave to show case the layout requirement of this instruction. this instruction is supported for 950 serious of AMDGPU (change arch number inside rebuild.sh to build on proper arch)

## 3 different way to play sith matrix core layout
```
matrix_core_kernel_standard:
    standard layout, each thread holding data along the column of C matrix
matrix_core_kernel_swap_a_b
    swap A/B pointer, each thread holding data along the row of C matrix, C matrix can use vector store(8x fp16, buffer_store_dwordx2)
matrix_core_kernel_swap_swb
    swap A/B pointer plus swizzle B, each thread holding data along the row of C matrix with larger vector size(16x fp16, buffer_store_dwordx4)
```
* note that transpose inside LDS before actually store out is another technique, but here omit this, just show case the layout of C matrix can be changed by A/B
