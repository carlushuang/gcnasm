#include <hip/hip_runtime.h>

#include "mori/cco/cco.hpp"
#include "mori/core/transport/sdma/anvil_device.hpp"

using namespace mori::cco;

static constexpr int kWaveSize = 64;

__global__ void opus_sdma_a2a_post_kernel(
    const unsigned char* staging,
    unsigned char* const* recv_peers,
    ccoWindow_t ready_win,
    ccoDevComm dev_comm,
    uint64_t* completion_signal,
    size_t bytes_per_peer) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int dst = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || dst >= dev_comm.lsaSize || dst == dev_comm.lsaRank) return;

    const int num_queues = static_cast<int>(dev_comm.sdma.sdmaNumQueue);
    const void* src = staging + static_cast<size_t>(dst) * bytes_per_peer;
    void* dst_ptr =
        recv_peers[dst] + static_cast<size_t>(dev_comm.lsaRank) * bytes_per_peer;
    auto** handles = dev_comm.sdma.deviceHandles + dst * num_queues;
    auto* remote_ready = static_cast<uint64_t*>(ccoGetLsaPeerPtr(
        ready_win, dst,
        static_cast<size_t>(dev_comm.lsaRank) * sizeof(uint64_t)));

    anvil::SdmaQueueDeviceHandle handle = **handles;
    uint64_t offset = 0;
    uint64_t base =
        handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
    uint64_t pending_wptr = base;
    const uint64_t start_base = base;
    auto copy_packet =
        anvil::CreateCopyPacket(const_cast<void*>(src), dst_ptr, bytes_per_peer);
    handle.placePacket<SDMA_PKT_COPY_LINEAR>(
        copy_packet, pending_wptr, offset);

    base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
    pending_wptr = base;
    auto remote_packet = anvil::CreateAtomicIncPacket(remote_ready);
    handle.placePacket<SDMA_PKT_ATOMIC>(
        remote_packet, pending_wptr, offset);

    base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
    pending_wptr = base;
    auto completion_packet =
        anvil::CreateAtomicIncPacket(completion_signal);
    handle.placePacket<SDMA_PKT_ATOMIC>(
        completion_packet, pending_wptr, offset);
    handle.submitPacket(start_base, pending_wptr);
}
