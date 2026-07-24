#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "mori/cco/cco.hpp"
#include "mori/core/transport/sdma/anvil_device.hpp"

using namespace mori::cco;
static constexpr int kWaveSize = 64;

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                 \
        }                                                                                                 \
    } while (0)

#define CHECK_CCO(call)                                                                    \
    do {                                                                                   \
        int cco_status_ = (call);                                                          \
        if (cco_status_ != 0) {                                                            \
            fprintf(stderr, "cco error %d (%s:%d): %s\n", cco_status_, __FILE__, __LINE__, \
                    #call);                                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

__global__ void sdma_a2a_post_kernel(
    const unsigned char* send_local,
    unsigned char* const* recv_peers,
    ccoWindow_t ready_win,
    ccoDevComm dev_comm,
    uint64_t* completion_signal,
    size_t bytes_per_peer) {
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int dst = static_cast<int>(threadIdx.x) / kWaveSize;
    if (lane != 0 || dst >= dev_comm.lsaSize || dst == dev_comm.lsaRank) return;

    const int num_queues = static_cast<int>(dev_comm.sdma.sdmaNumQueue);
    if (num_queues <= 0) return;

    const void* src = send_local + static_cast<size_t>(dst) * bytes_per_peer;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t bytes_per_peer = 10ULL * 1024ULL * 1024ULL;
    int warmup = 5;
    int iters = 100;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bytes-per-peer") == 0 && i + 1 < argc) {
            bytes_per_peer = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 0));
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }
    if (nranks < 2 || nranks > 8 || bytes_per_peer == 0 || warmup < 0 ||
        iters <= 0) {
        if (rank == 0) {
            fprintf(stderr, "requires 2-8 ranks, positive bytes/iters, and nonnegative warmup\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const char* sdma_env = std::getenv("MORI_ENABLE_SDMA");
    if (!sdma_env || std::strcmp(sdma_env, "1") != 0) {
        if (rank == 0) fprintf(stderr, "set MORI_ENABLE_SDMA=1 before launch\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int ndev = 0;
    CHECK_HIP(hipGetDeviceCount(&ndev));
    CHECK_HIP(hipSetDevice(rank % ndev));

    ccoUniqueId uid;
    if (rank == 0) CHECK_CCO(ccoGetUniqueId(&uid));
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    ccoComm* comm = nullptr;
    CHECK_CCO(ccoCommCreate(uid, nranks, rank, 256ULL * 1024ULL * 1024ULL, &comm));
    ccoWindow_t ready_win = nullptr;
    void* ready_local = nullptr;
    CHECK_CCO(ccoWindowRegister(
        comm, static_cast<size_t>(nranks) * sizeof(uint64_t),
        &ready_win, &ready_local));

    ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
    reqs.gdaSignalCount = 0;
    reqs.gdaCounterCount = 0;
    reqs.sdmaQueueCount = 1;
    ccoDevComm dev_comm{};
    CHECK_CCO(ccoDevCommCreate(comm, &reqs, &dev_comm));
    if (dev_comm.sdma.sdmaNumQueue == 0 || !dev_comm.sdma.deviceHandles) {
        if (rank == 0) fprintf(stderr, "MORI did not materialize SDMA queues\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t buffer_bytes = static_cast<size_t>(nranks) * bytes_per_peer;
    void* send_local_void = nullptr;
    void* recv_local_void = nullptr;
    CHECK_HIP(hipExtMallocWithFlags(
        &send_local_void, buffer_bytes, hipDeviceMallocUncached));
    CHECK_HIP(hipExtMallocWithFlags(
        &recv_local_void, buffer_bytes, hipDeviceMallocUncached));
    auto* send_local = static_cast<unsigned char*>(send_local_void);
    auto* recv_local = static_cast<unsigned char*>(recv_local_void);

    hipIpcMemHandle_t recv_handle{};
    CHECK_HIP(hipIpcGetMemHandle(&recv_handle, recv_local));
    std::vector<hipIpcMemHandle_t> recv_handles(static_cast<size_t>(nranks));
    MPI_Allgather(
        &recv_handle, sizeof(recv_handle), MPI_BYTE,
        recv_handles.data(), sizeof(recv_handle), MPI_BYTE, MPI_COMM_WORLD);
    std::vector<unsigned char*> recv_peers(static_cast<size_t>(nranks), nullptr);
    recv_peers[rank] = recv_local;
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == rank) continue;
        void* mapped = nullptr;
        CHECK_HIP(hipIpcOpenMemHandle(
            &mapped, recv_handles[peer], hipIpcMemLazyEnablePeerAccess));
        recv_peers[peer] = static_cast<unsigned char*>(mapped);
    }
    unsigned char** recv_peers_device = nullptr;
    CHECK_HIP(hipMalloc(
        &recv_peers_device, static_cast<size_t>(nranks) * sizeof(unsigned char*)));
    CHECK_HIP(hipMemcpy(
        recv_peers_device, recv_peers.data(),
        static_cast<size_t>(nranks) * sizeof(unsigned char*),
        hipMemcpyHostToDevice));
    uint64_t* completion_signal = nullptr;
    CHECK_HIP(hipExtMallocWithFlags(
        reinterpret_cast<void**>(&completion_signal),
        sizeof(uint64_t), hipMallocSignalMemory));
    CHECK_HIP(hipMemset(completion_signal, 0, sizeof(uint64_t)));

    CHECK_HIP(hipMemset(send_local, rank + 1, buffer_bytes));
    CHECK_HIP(hipMemset(recv_local, 0, buffer_bytes));
    CHECK_HIP(hipMemset(
        ready_local, 0, static_cast<size_t>(nranks) * sizeof(uint64_t)));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_CCO(ccoBarrierAll(comm));
    (void)hipGetLastError();

    const dim3 block(static_cast<unsigned>(nranks * kWaveSize));
    uint64_t epoch = 0;
    auto launch_once = [&]() {
        ++epoch;
        sdma_a2a_post_kernel<<<1, block>>>(
            send_local, recv_peers_device, ready_win, dev_comm,
            completion_signal, bytes_per_peer);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipStreamWaitValue64(
            nullptr, completion_signal,
            epoch * static_cast<uint64_t>(nranks - 1), hipStreamWaitValueGte));
    };

    for (int i = 0; i < warmup; ++i) launch_once();
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_CCO(ccoBarrierAll(comm));

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch_once();
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
    const double local_ms = static_cast<double>(total_ms) / iters;
    double max_ms = 0.0;
    MPI_Allreduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    CHECK_CCO(ccoBarrierAll(comm));

    int mismatches = 0;
    std::vector<unsigned char> recv_host(buffer_bytes);
    std::vector<uint64_t> ready_host(static_cast<size_t>(nranks));
    uint64_t signal_host = 0;
    CHECK_HIP(hipMemcpy(
        recv_host.data(), recv_local, buffer_bytes, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(
        &signal_host, completion_signal, sizeof(uint64_t), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(
        ready_host.data(), ready_local,
        static_cast<size_t>(nranks) * sizeof(uint64_t),
        hipMemcpyDeviceToHost));
    const auto* recv = recv_host.data();
    const uint64_t expected_ready =
        static_cast<uint64_t>(warmup + iters) * (nranks - 1);
    const size_t samples[] = {0, bytes_per_peer / 2, bytes_per_peer - 1};
    for (int src = 0; src < nranks; ++src) {
        if (src == rank) continue;
        for (size_t offset : samples) {
            const unsigned char got =
                recv[static_cast<size_t>(src) * bytes_per_peer + offset];
            const unsigned char expected = static_cast<unsigned char>(src + 1);
            if (got != expected) {
                if (mismatches < 8) {
                    printf("[rank %d] data mismatch src=%d offset=%zu got=%u expected=%u\n",
                           rank, src, offset, static_cast<unsigned>(got),
                           static_cast<unsigned>(expected));
                }
                ++mismatches;
            }
        }
    }
    if (signal_host != expected_ready) {
        if (mismatches < 8) {
            printf("[rank %d] completion mismatch got=%llu expected=%llu\n",
                   rank, static_cast<unsigned long long>(signal_host),
                   static_cast<unsigned long long>(expected_ready));
        }
        ++mismatches;
    }
    const uint64_t expected_remote = static_cast<uint64_t>(warmup + iters);
    for (int src = 0; src < nranks; ++src) {
        if (src == rank || ready_host[src] == expected_remote) continue;
        if (mismatches < 8) {
            printf("[rank %d] remote-ready mismatch src=%d got=%llu expected=%llu\n",
                   rank, src,
                   static_cast<unsigned long long>(ready_host[src]),
                   static_cast<unsigned long long>(expected_remote));
        }
        ++mismatches;
    }
    int total_mismatches = 0;
    MPI_Allreduce(
        &mismatches, &total_mismatches, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        const double remote_bytes =
            static_cast<double>(nranks - 1) * static_cast<double>(bytes_per_peer);
        const double per_rank_gib_s =
            remote_bytes / (max_ms * 1.0e-3) / double(1ULL << 30);
        printf("sdma_a2a ranks=%d bytes_per_peer=%zu queues=%u "
               "max_rank_time=%.4f ms per_rank_bw=%.2f GiB/s "
               "aggregate_bw=%.2f GiB/s %s\n",
               nranks, bytes_per_peer, dev_comm.sdma.sdmaNumQueue,
               max_ms, per_rank_gib_s, per_rank_gib_s * nranks,
               total_mismatches == 0 ? "SUCCESS" : "FAILED");
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipFree(completion_signal));
    CHECK_HIP(hipFree(recv_peers_device));
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer != rank) CHECK_HIP(hipIpcCloseMemHandle(recv_peers[peer]));
    }
    CHECK_HIP(hipFree(recv_local));
    CHECK_HIP(hipFree(send_local));
    CHECK_CCO(ccoDevCommDestroy(comm, &dev_comm));
    CHECK_CCO(ccoWindowDeregister(comm, ready_win));
    CHECK_CCO(ccoCommDestroy(comm));
    MPI_Finalize();
    return total_mismatches == 0 ? 0 : 1;
}
