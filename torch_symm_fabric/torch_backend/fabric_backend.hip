// fabric_backend.hip -- a torch SymmetricMemory backend built on HIP fabric handles.
//
// The other approaches in this directory work *around* torch. This one plugs *into* it:
// c10d::symmetric_memory::register_allocator() is a documented extension point, so a
// backend that implements SymmetricMemoryAllocator + SymmetricMemory gets driven by
// torch's own entry points.
//
//     import fabric_backend                       # registers "FABRIC"
//     symm_mem.set_backend("FABRIC")
//     buf = symm_mem.empty(...)                   # -> FabricAllocator::alloc
//     hdl = symm_mem.rendezvous(buf, group)       # -> FabricAllocator::rendezvous
//
// The catch, and the reason the rest of the directory exists: registering means
// *supplying* rendezvous, not reusing torch's. torch calls ours. What it buys is that
// torch.ops.symm_mem::* dispatch through this object, so the collectives can run on
// fabric memory without waiting for upstream to fix the ROCm branches.
//
// Peers are laid out in one flat span, so buffer_ptrs()[r] == flat_base + r*stride and a
// kernel can address a rank arithmetically instead of walking buffer_ptrs_dev.

#include <hip/hip_runtime.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/extension.h>

#include <map>
#include <mutex>
#include <vector>

namespace {

using namespace c10d::symmetric_memory;

#define HIPCHECK(expr)                                                    \
  do {                                                                    \
    hipError_t _e = (expr);                                               \
    TORCH_CHECK(                                                          \
        _e == hipSuccess, #expr, " failed: ", hipGetErrorString(_e));     \
  } while (0)

constexpr size_t kSignalPadBytes = 9216;  // match torch's, so layouts stay comparable

// What each rank publishes so peers can map it.
struct FabricRendezvous {
  unsigned char handle[64];  // hipMemFabricHandle_t
  size_t alloc_size;
  int device_idx;
};

size_t round_up(size_t v, size_t m) {
  return ((v + m - 1) / m) * m;
}

hipMemAllocationProp fabric_prop(int device_idx) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypeFabric;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device_idx;
  return prop;
}

void set_rw_access(void* ptr, size_t size, int device_idx) {
  hipMemAccessDesc desc = {};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = device_idx;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  HIPCHECK(hipMemSetAccess(ptr, size, &desc, 1));
}

// One local allocation, before any rendezvous has happened.
struct Block {
  void* ptr = nullptr;
  size_t buffer_size = 0;  // what the user asked for
  size_t alloc_size = 0;   // buffer + signal pad, rounded to granularity
  int device_idx = 0;
  hipMemGenericAllocationHandle_t handle{};
  std::optional<std::string> default_group_name;
  c10::intrusive_ptr<SymmetricMemory> symm;  // cached after rendezvous
};

class FabricSymmetricMemory : public SymmetricMemory {
 public:
  FabricSymmetricMemory(
      char* flat_base,
      size_t span,
      size_t stride,
      size_t buffer_size,
      std::vector<hipMemGenericAllocationHandle_t> handles,
      int rank,
      int world_size,
      int device_idx)
      : flat_base_(flat_base),
        span_(span),
        stride_(stride),
        buffer_size_(buffer_size),
        handles_(std::move(handles)),
        rank_(rank),
        world_size_(world_size),
        device_(c10::DeviceType::CUDA, device_idx) {
    buffers_.reserve(world_size);
    signal_pads_.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      buffers_.push_back(flat_base_ + r * stride_);
      signal_pads_.push_back(flat_base_ + r * stride_ + buffer_size_);
    }
    // device-side copies of both arrays, for kernels that want the torch contract
    HIPCHECK(hipMalloc(&buffers_dev_, world_size * sizeof(void*)));
    HIPCHECK(hipMalloc(&signal_pads_dev_, world_size * sizeof(void*)));
    HIPCHECK(hipMemcpy(buffers_dev_, buffers_.data(), world_size * sizeof(void*), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(signal_pads_dev_, signal_pads_.data(), world_size * sizeof(void*), hipMemcpyHostToDevice));
  }

  ~FabricSymmetricMemory() override {
    if (buffers_dev_ != nullptr) {
      (void)hipFree(buffers_dev_);
    }
    if (signal_pads_dev_ != nullptr) {
      (void)hipFree(signal_pads_dev_);
    }
    for (int r = 0; r < world_size_; ++r) {
      (void)hipMemUnmap(flat_base_ + r * stride_, stride_);
      (void)hipMemRelease(handles_[r]);
    }
    (void)hipMemAddressFree(flat_base_, span_);
  }

  std::vector<void*> get_buffer_ptrs() override { return buffers_; }
  std::vector<void*> get_signal_pad_ptrs() override { return signal_pads_; }
  void** get_buffer_ptrs_dev() override { return buffers_dev_; }
  void** get_signal_pad_ptrs_dev() override { return signal_pads_dev_; }
  size_t get_buffer_size() override { return buffer_size_; }
  size_t get_offset() override { return 0; }

  // No NVLS equivalent here.
  bool has_multicast_support() override { return false; }
  void* get_multicast_ptr() override { return nullptr; }

  int get_rank() override { return rank_; }
  int get_world_size() override { return world_size_; }
  c10::Device get_device() override { return device_; }

  // Every peer is mapped for direct load/store -- that is the whole point.
  bool world_within_direct_access() override { return true; }

  // Host-side, deliberately: correct, but it drains the device and round-trips the store
  // rather than spinning on the signal pad the way torch's kernels do.
  void barrier(int channel, size_t timeout_ms) override {
    HIPCHECK(hipDeviceSynchronize());
    TORCH_CHECK(store_barrier_ != nullptr, "barrier requires a group store");
    store_barrier_();
  }

  void put_signal(int dst_rank, int channel, size_t timeout_ms) override {
    TORCH_CHECK(false, "fabric backend: put_signal is not implemented yet");
  }

  void wait_signal(int src_rank, int channel, size_t timeout_ms) override {
    TORCH_CHECK(false, "fabric backend: wait_signal is not implemented yet");
  }

  void set_store_barrier(std::function<void()> fn) { store_barrier_ = std::move(fn); }

  // extras beyond the torch interface, for arithmetic peer addressing
  uintptr_t flat_base() const { return reinterpret_cast<uintptr_t>(flat_base_); }
  size_t stride() const { return stride_; }

 private:
  char* flat_base_;
  size_t span_;
  size_t stride_;
  size_t buffer_size_;
  std::vector<hipMemGenericAllocationHandle_t> handles_;
  int rank_;
  int world_size_;
  c10::Device device_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_ = nullptr;
  void** signal_pads_dev_ = nullptr;
  std::function<void()> store_barrier_;
};

class FabricAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(size_t size, int device_idx, const std::optional<std::string>& group_name)
      override {
    HIPCHECK(hipSetDevice(device_idx));

    auto prop = fabric_prop(device_idx);
    size_t granularity = 0;
    HIPCHECK(hipMemGetAllocationGranularity(
        &granularity, &prop, hipMemAllocationGranularityRecommended));
    // buffer and signal pad share one allocation, as torch does
    size_t alloc_size = round_up(size + kSignalPadBytes, granularity);

    hipMemGenericAllocationHandle_t handle{};
    HIPCHECK(hipMemCreate(&handle, alloc_size, &prop, 0));

    void* ptr = nullptr;
    HIPCHECK(hipMemAddressReserve(&ptr, alloc_size, 0, nullptr, 0));
    HIPCHECK(hipMemMap(ptr, alloc_size, 0, handle, 0));
    set_rw_access(ptr, alloc_size, device_idx);
    HIPCHECK(hipMemset(ptr, 0, alloc_size));

    auto block = std::make_shared<Block>();
    block->ptr = ptr;
    block->buffer_size = size;
    block->alloc_size = alloc_size;
    block->device_idx = device_idx;
    block->handle = handle;
    block->default_group_name = group_name;

    std::lock_guard<std::mutex> lock(mutex_);
    blocks_[ptr] = std::move(block);
    return ptr;
  }

  void free(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(ptr);
    if (it == blocks_.end()) {
      return;
    }
    auto block = it->second;
    blocks_.erase(it);
    // the SymmetricMemory object owns the flat span; this only drops our own mapping
    (void)hipMemUnmap(block->ptr, block->alloc_size);
    (void)hipMemAddressFree(block->ptr, block->alloc_size);
    (void)hipMemRelease(block->handle);
  }

  size_t get_alloc_size(void* ptr) override {
    auto block = find_block(ptr);
    TORCH_CHECK(block != nullptr, "fabric backend: pointer is not a fabric allocation");
    return block->buffer_size;
  }

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    auto block = find_block(ptr);
    TORCH_CHECK(block != nullptr, "fabric backend: pointer is not a fabric allocation");
    if (block->symm != nullptr) {
      return block->symm;  // rendezvous is once per allocation
    }

    auto name = group_name.has_value() ? group_name : block->default_group_name;
    TORCH_CHECK(
        name.has_value(),
        "fabric backend: group_name was given neither at allocation nor at rendezvous");
    auto& info = get_group_info(*name);
    int rank = info.rank;
    int world_size = info.world_size;

    HIPCHECK(hipSetDevice(block->device_idx));

    // Publish our handle and collect everyone else's.
    FabricRendezvous local{};
    HIPCHECK(hipMemExportToShareableHandle(
        local.handle, block->handle, hipMemHandleTypeFabric, 0));
    local.alloc_size = block->alloc_size;
    local.device_idx = block->device_idx;

    auto peers = store_exchange_.all_gather(info.store, rank, world_size, local);
    for (int r = 0; r < world_size; ++r) {
      TORCH_CHECK(
          peers[r].alloc_size == local.alloc_size,
          "fabric backend: rank ", r, " allocated ", peers[r].alloc_size,
          " bytes, this rank ", local.alloc_size, " -- sizes must match");
    }

    // One reservation for every rank, so peer(r) == flat_base + r*stride.
    size_t stride = local.alloc_size;
    size_t span = static_cast<size_t>(world_size) * stride;
    void* flat = nullptr;
    auto prop = fabric_prop(block->device_idx);
    size_t granularity = 0;
    HIPCHECK(hipMemGetAllocationGranularity(
        &granularity, &prop, hipMemAllocationGranularityRecommended));
    HIPCHECK(hipMemAddressReserve(&flat, span, granularity, nullptr, 0));

    std::vector<hipMemGenericAllocationHandle_t> handles(world_size);
    for (int r = 0; r < world_size; ++r) {
      char* slot = static_cast<char*>(flat) + r * stride;
      if (r == rank) {
        // our own allocation gets a second alias inside the span, so the stride is
        // uniform across all ranks and peer(rank) is an ordinary slot
        HIPCHECK(hipMemRetainAllocationHandle(&handles[r], block->ptr));
      } else {
        HIPCHECK(hipMemImportFromShareableHandle(
            &handles[r], peers[r].handle, hipMemHandleTypeFabric));
      }
      HIPCHECK(hipMemMap(slot, stride, 0, handles[r], 0));
      set_rw_access(slot, stride, block->device_idx);
    }

    auto symm = c10::make_intrusive<FabricSymmetricMemory>(
        static_cast<char*>(flat), span, stride, block->buffer_size,
        std::move(handles), rank, world_size, block->device_idx);

    auto store = info.store;
    auto exchange = std::make_shared<StoreExchange>("fabric_backend_barrier");
    symm->set_store_barrier([store, exchange, rank, world_size]() {
      exchange->all_gather(store, rank, world_size, static_cast<int>(1));
    });

    block->symm = symm;
    return symm;
  }

  bool has_multicast_support(int /*device_idx*/) override { return false; }
  c10::DeviceType supported_device_type() override { return c10::DeviceType::CUDA; }
  std::string name() override { return "FABRIC"; }

  // exposed to python so the flat layout is inspectable
  std::shared_ptr<Block> find_block(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(ptr);
    return it == blocks_.end() ? nullptr : it->second;
  }

 private:
  std::mutex mutex_;
  std::map<void*, std::shared_ptr<Block>> blocks_;
  StoreExchange store_exchange_{"fabric_backend"};
};

c10::intrusive_ptr<FabricAllocator>& allocator_singleton() {
  static c10::intrusive_ptr<FabricAllocator> inst = c10::make_intrusive<FabricAllocator>();
  return inst;
}

// (flat_base, stride) for a rendezvous'd allocation -- the arithmetic addressing that
// torch's own interface has no place for.
std::tuple<uintptr_t, int64_t> flat_layout(const at::Tensor& tensor) {
  auto block = allocator_singleton()->find_block(tensor.data_ptr());
  TORCH_CHECK(block != nullptr, "not a fabric allocation");
  TORCH_CHECK(block->symm != nullptr, "rendezvous() has not been called on this tensor");
  auto* symm = static_cast<FabricSymmetricMemory*>(block->symm.get());
  return {symm->flat_base(), static_cast<int64_t>(symm->stride())};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // register_availability makes the name selectable via symm_mem.set_backend("FABRIC")
  register_availability("FABRIC", allocator_singleton());
  m.def("flat_layout", &flat_layout, "(flat_base, stride) of a rendezvous'd tensor");
  m.attr("backend_name") = "FABRIC";
}
