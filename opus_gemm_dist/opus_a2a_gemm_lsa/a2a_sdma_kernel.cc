#include <hip/hip_runtime.h>

#include "mori/cco/cco.hpp"

using namespace mori::cco;

static constexpr int kWaveSize = 64;

// Retained optimization: one lane per remote peer posts one bulk no-signal
// PUT into that peer's queue; generic A2A selects a distinct source chunk.
__global__ void a2a_sdma_post_kernel(
    ccoWindow_t send_win,
    ccoWindow_t recv_win,
    ccoDevComm dev_comm,
    size_t send_slot_offset,
    size_t recv_slot_offset,
    size_t bytes_per_peer,
    int input_mode) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int peer = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || peer >= dev_comm.lsaSize || peer == dev_comm.lsaRank) return;

    const size_t src_chunk = input_mode == 1 ? static_cast<size_t>(peer) : 0;
    ccoSdma{dev_comm}.put<ccoCoopThread>(
        peer,
        recv_win,
        recv_slot_offset + static_cast<size_t>(dev_comm.lsaRank) * bytes_per_peer,
        send_win,
        send_slot_offset + src_chunk * bytes_per_peer,
        bytes_per_peer,
        0);
}

// Retained optimization: quietQueue supplies transfer completion, then one
// release-ordered epoch wakes the destination without a PUT-tail signal.
__global__ void a2a_sdma_quiet_notify_kernel(
    ccoWindow_t ready_win,
    ccoDevComm dev_comm,
    size_t ready_slot_offset) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int peer = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || peer >= dev_comm.lsaSize || peer == dev_comm.lsaRank) return;

    ccoSdma{dev_comm}.quietQueue(peer, 0);
    auto* remote_ready = static_cast<uint64_t*>(ccoGetLsaPeerPtr(
        ready_win,
        peer,
        ready_slot_offset +
            static_cast<size_t>(dev_comm.lsaRank) * sizeof(uint64_t)));
    __hip_atomic_fetch_add(
        remote_ready, 1ULL, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Cross-epoch pipeline join: wait for every remote source's monotonic epoch.
// Sleeping failed polls reduces CU pressure while SDMA is still in flight.
__global__ void a2a_sdma_wait_ready_kernel(
    const uint64_t* ready_local,
    size_t ready_slot_offset,
    uint64_t target,
    int rank_count,
    int my_rank) {
    if (threadIdx.x != 0) return;

    const auto* slot_ready = reinterpret_cast<const uint64_t*>(
        reinterpret_cast<const char*>(ready_local) + ready_slot_offset);
    for (int source = 0; source < rank_count; ++source) {
        if (source == my_rank) continue;
        while (__hip_atomic_load(
                   slot_ready + source,
                   __ATOMIC_ACQUIRE,
                   __HIP_MEMORY_SCOPE_SYSTEM) < target) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
}
