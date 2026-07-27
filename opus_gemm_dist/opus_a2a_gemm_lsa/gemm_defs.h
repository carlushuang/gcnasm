#pragma once

#include <cstdint>

#include "../opus_gemm_a2a_lsa/gemm_defs.h"
#include "mori/cco/cco.hpp"

enum opus_a2a_input_mode : int {
    OPUS_A2A_INPUT_BROADCAST = 0,
    OPUS_A2A_INPUT_GENERIC = 1,
};

struct opus_a2a_gemm_kargs {
    const void* __restrict__ local_a = nullptr;   // broadcast: [M,K_SHARD]; generic: [rank_count,M,K_SHARD]
    const void* __restrict__ ptr_b = nullptr;     // [N, K], bf16, replicated on every rank
    void* __restrict__ ptr_c = nullptr;           // [M, N], bf16
    void* __restrict__ workspace = nullptr;       // [M, N], fp32 accumulation workspace

    void* recv_a_win = nullptr;                   // LSA window: [rank_count, M, K_SHARD], bf16
    void* ready_win = nullptr;                    // LSA window: [rank_count], uint32 flags
    void* recv_a_local = nullptr;                 // local device pointer for this rank's recv_a window
    void* ready_local = nullptr;                  // local device pointer for this rank's ready flags
    const uint64_t* sdma_ready_local = nullptr;   // slot-local SDMA ready counters [rank_count]
    uint64_t sdma_ready_target = 0;                // expected per-source epoch for ready-aware GEMM

    unsigned int* wg_hw_records = nullptr;        // optional [wg_hw_record_count, 6]: bx,xcc,se,sh,cu,is_comm
    unsigned int* tile_counter = nullptr;         // persistent compute task counter

    int m = 2048;
    int n = 8192;
    int k = 8192;
    int k_shard = 1024;
    int rank_count = 8;
    int my_rank = 0;
    int input_mode = OPUS_A2A_INPUT_BROADCAST;

    int stride_a = 1024;      // local/recv A shard row stride
    int stride_b = 8192;      // full B row stride
    int stride_c = 8192;      // output C row stride
    int stride_ws = 8192;     // FP32 workspace row stride

    unsigned int recv_a_bytes = 0;
    unsigned int ready_bytes = 0;
    unsigned int output_bytes = 0;
    unsigned int workspace_bytes = 0;

    int comm_wgs = 32;             // active rotate-copy WGs; placed every 8th WG in fused mode
    int wg_hw_record_count = 0;
    int record_wg_hw = 0;
    int num_m_tiles = 0;
    int num_n_tiles = 0;
    int mode = 0;             // 0=fused, 1=compute-only-local full-K
};

struct opus_a2a_gemm_sdma_fused_comm_state {
    mori::cco::ccoWindow_t sdma_send_win = nullptr;
    mori::cco::ccoWindow_t sdma_recv_win = nullptr;
    mori::cco::ccoWindow_t sdma_ready_win = nullptr;
    const mori::cco::ccoDevComm* sdma_dev_comm = nullptr;
    uint64_t sdma_send_slot_offset = 0;
    uint64_t sdma_recv_slot_offset = 0;
    uint64_t sdma_ready_slot_offset = 0;
    uint64_t sdma_bytes_per_peer = 0;
};

struct opus_a2a_gemm_sdma_fused_kargs : opus_a2a_gemm_kargs {
    const opus_a2a_gemm_sdma_fused_comm_state* sdma_comm_state = nullptr;
};
