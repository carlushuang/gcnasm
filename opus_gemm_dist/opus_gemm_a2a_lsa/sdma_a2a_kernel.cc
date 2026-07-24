#include <hip/hip_runtime.h>

#include "mori/cco/cco.hpp"

using namespace mori::cco;

static constexpr int kWaveSize = 64;

__global__ void opus_sdma_a2a_post_kernel(
    ccoWindow_t staging_win,
    ccoWindow_t recv_win,
    ccoDevComm dev_comm,
    size_t staging_slot_offset,
    size_t bytes_per_peer) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int dst = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || dst >= dev_comm.lsaSize || dst == dev_comm.lsaRank) return;

    ccoSdma sdma{dev_comm};
    sdma.put<ccoCoopThread, true>(
        dst, recv_win, static_cast<size_t>(dev_comm.lsaRank) * bytes_per_peer,
        staging_win, staging_slot_offset + static_cast<size_t>(dst) * bytes_per_peer,
        bytes_per_peer, 0);
}

__global__ void opus_sdma_a2a_quiet_notify_kernel(
    ccoWindow_t ready_win, ccoDevComm dev_comm) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int peer = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || peer >= dev_comm.lsaSize || peer == dev_comm.lsaRank) return;

    ccoSdma{dev_comm}.quietQueue(peer, 0);
    auto* remote_ready = static_cast<uint64_t*>(ccoGetLsaPeerPtr(
        ready_win, peer,
        static_cast<size_t>(dev_comm.lsaRank) * sizeof(uint64_t)));
    __hip_atomic_fetch_add(
        remote_ready, 1ULL, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
