#include <hip/hip_runtime.h>

#include "mori/cco/cco.hpp"

using namespace mori::cco;

static constexpr int kWaveSize = 64;

// Retained optimization: one lane per destination submits one bulk no-signal
// PUT; data completion is deferred to the separate quiet/notify kernel.
__global__ void opus_sdma_a2a_post_kernel(
    ccoWindow_t staging_win,
    ccoWindow_t recv_win,
    ccoDevComm dev_comm,
    size_t staging_slot_offset,
    size_t bytes_per_peer) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int dst = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || dst >= dev_comm.lsaSize || dst == dev_comm.lsaRank) return;

    mori::cco::ccoSdma sdma{dev_comm};
    sdma.put<mori::cco::ccoCoopThread>(
        dst, recv_win, static_cast<size_t>(dev_comm.lsaRank) * bytes_per_peer,
        staging_win, staging_slot_offset + static_cast<size_t>(dst) * bytes_per_peer,
        bytes_per_peer, 0);
}

// Experimental split-SDMA-v2 baseline: submit multiple M-aligned no-signal
// PUTs per peer without changing the staging GEMM.
__global__ void opus_sdma_a2a_chunked_post_kernel(
    ccoWindow_t staging_win,
    ccoWindow_t recv_win,
    ccoDevComm dev_comm,
    size_t staging_slot_offset,
    size_t bytes_per_peer,
    size_t chunk_bytes) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int dst = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || dst >= dev_comm.lsaSize || dst == dev_comm.lsaRank) return;

    mori::cco::ccoSdma sdma{dev_comm};
    const size_t peer_src_offset =
        staging_slot_offset + static_cast<size_t>(dst) * bytes_per_peer;
    const size_t peer_dst_offset =
        static_cast<size_t>(dev_comm.lsaRank) * bytes_per_peer;
    for (size_t chunk_offset = 0; chunk_offset < bytes_per_peer;
         chunk_offset += chunk_bytes) {
        const size_t remaining = bytes_per_peer - chunk_offset;
        const size_t bytes = remaining < chunk_bytes ? remaining : chunk_bytes;
        sdma.put<mori::cco::ccoCoopThread>(
            dst, recv_win, peer_dst_offset + chunk_offset,
            staging_win, peer_src_offset + chunk_offset, bytes, 0);
    }
}

// Retained optimization: drain each peer queue from MORI rptr/wptr state and
// publish one release-ordered ready epoch, avoiding per-PUT signal atomics.
__global__ void opus_sdma_a2a_quiet_notify_kernel(
    ccoWindow_t ready_win, ccoDevComm dev_comm) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int peer = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || peer >= dev_comm.lsaSize || peer == dev_comm.lsaRank) return;

    mori::cco::ccoSdma{dev_comm}.quietQueue(peer, 0);
    auto* remote_ready = static_cast<uint64_t*>(ccoGetLsaPeerPtr(
        ready_win, peer,
        static_cast<size_t>(dev_comm.lsaRank) * sizeof(uint64_t)));
    __hip_atomic_fetch_add(
        remote_ready, 1ULL, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
