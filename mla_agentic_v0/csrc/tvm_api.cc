#include <cmath>
#include <tvm/ffi/tvm_ffi.h>
#include "flash_mla.hpp"

/// TVM FFI binding for flash_mla_decode (MODEL1 FP8, MFMA)
void flash_mla_decode_ffi(
    tvm::ffi::Tensor q,
    tvm::ffi::Tensor kv_nope,
    tvm::ffi::Tensor kv_rope,
    tvm::ffi::Tensor kv_scales,
    tvm::ffi::Tensor output,
    tvm::ffi::Tensor kv_indptr,
    int num_heads,
    float sm_scale)
{
    void* q_ptr    = static_cast<char*>(q.data_ptr())  + q.byte_offset();
    void* nope_ptr = static_cast<char*>(kv_nope.data_ptr()) + kv_nope.byte_offset();
    void* rope_ptr = static_cast<char*>(kv_rope.data_ptr()) + kv_rope.byte_offset();
    void* sc_ptr   = static_cast<char*>(kv_scales.data_ptr()) + kv_scales.byte_offset();
    void* o_ptr    = static_cast<char*>(output.data_ptr()) + output.byte_offset();
    void* indptr_ptr = static_cast<char*>(kv_indptr.data_ptr()) + kv_indptr.byte_offset();

    int batch_size = static_cast<int>(q.shape()[0]);

    if (sm_scale == 0.0f) {
        sm_scale = 1.0f / std::sqrt(512.0f);
    }

    flash_mla_decode(q_ptr, nope_ptr, rope_ptr, sc_ptr, o_ptr, indptr_ptr,
                     batch_size, num_heads, sm_scale, nullptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flash_mla_decode, flash_mla_decode_ffi);

/// TVM FFI binding for flash_mla_sparse_decode
void flash_mla_sparse_decode_ffi(
    tvm::ffi::Tensor q,
    tvm::ffi::Tensor kv_packed,
    tvm::ffi::Tensor indices,
    tvm::ffi::Tensor output,
    tvm::ffi::Tensor lse,
    int num_heads,
    int topk,
    float sm_scale,
    int has_topk_length,
    tvm::ffi::Tensor topk_length)
{
    void* q_ptr   = static_cast<char*>(q.data_ptr())   + q.byte_offset();
    void* kv_ptr  = static_cast<char*>(kv_packed.data_ptr()) + kv_packed.byte_offset();
    void* idx_ptr = static_cast<char*>(indices.data_ptr()) + indices.byte_offset();
    void* o_ptr   = static_cast<char*>(output.data_ptr()) + output.byte_offset();
    void* lse_ptr = static_cast<char*>(lse.data_ptr())  + lse.byte_offset();

    void* topk_len_ptr = nullptr;
    if (has_topk_length) {
        topk_len_ptr = static_cast<char*>(topk_length.data_ptr()) + topk_length.byte_offset();
    }

    int b   = static_cast<int>(q.shape()[0]);
    int s_q = static_cast<int>(q.shape()[1]);

    if (sm_scale == 0.0f) {
        sm_scale = 1.0f / std::sqrt(512.0f);
    }

    flash_mla_sparse_decode(
        q_ptr, kv_ptr, idx_ptr, topk_len_ptr,
        o_ptr, lse_ptr,
        b, s_q, num_heads, topk,
        sm_scale, nullptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flash_mla_sparse_decode,
                               flash_mla_sparse_decode_ffi);

/// TVM FFI binding for flash_mla_sparse_decode_splitk
void flash_mla_sparse_decode_splitk_ffi(
    tvm::ffi::Tensor q,
    tvm::ffi::Tensor kv_packed,
    tvm::ffi::Tensor indices,
    tvm::ffi::Tensor o_partial,
    tvm::ffi::Tensor lse_partial,
    tvm::ffi::Tensor output,
    tvm::ffi::Tensor lse,
    int num_heads,
    int topk,
    int num_splits,
    float sm_scale,
    int has_topk_length,
    tvm::ffi::Tensor topk_length)
{
    void* q_ptr   = static_cast<char*>(q.data_ptr())   + q.byte_offset();
    void* kv_ptr  = static_cast<char*>(kv_packed.data_ptr()) + kv_packed.byte_offset();
    void* idx_ptr = static_cast<char*>(indices.data_ptr()) + indices.byte_offset();
    void* op_ptr  = static_cast<char*>(o_partial.data_ptr()) + o_partial.byte_offset();
    void* lp_ptr  = static_cast<char*>(lse_partial.data_ptr()) + lse_partial.byte_offset();
    void* o_ptr   = static_cast<char*>(output.data_ptr()) + output.byte_offset();
    void* lse_ptr = static_cast<char*>(lse.data_ptr())  + lse.byte_offset();

    void* topk_len_ptr = nullptr;
    if (has_topk_length) {
        topk_len_ptr = static_cast<char*>(topk_length.data_ptr()) + topk_length.byte_offset();
    }

    int b   = static_cast<int>(q.shape()[0]);
    int s_q = static_cast<int>(q.shape()[1]);

    if (sm_scale == 0.0f) {
        sm_scale = 1.0f / std::sqrt(512.0f);
    }

    flash_mla_sparse_decode_splitk(
        q_ptr, kv_ptr, idx_ptr, topk_len_ptr,
        op_ptr, lp_ptr, o_ptr, lse_ptr,
        b, s_q, num_heads, topk, num_splits,
        sm_scale, nullptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flash_mla_sparse_decode_splitk,
                               flash_mla_sparse_decode_splitk_ffi);
