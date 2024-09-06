#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "mem_stream.h"

#ifdef RAND_INT
#define PER_PIXEL_CHECK
#endif

template<typename T>
int valid_vector_integer(const T* lhs, const T * rhs, size_t len){
    int err_cnt = 0;
    for(size_t i = 0;i < len; i++){
        if(lhs[i] == rhs[i])
            ;
        else{
            printf(" diff at %d, lhs:%d, rhs:%d\n", (int)i, lhs[i], rhs[i]);
            err_cnt++;
        }
    }
    return err_cnt;
}

template<typename T>
void rand_vector(T* v, int pixels)
{
    for(auto i = 0; i < pixels; i++) {
        v[i] =  ((T)(rand() % 10)) - 5;
    }
}

static inline void b2s(size_t bytes, char * str){
	if(bytes<1024){
		sprintf(str, "%luB", bytes);
	}else if(bytes<(1024*1024)){
		double b= (double)bytes/1024.0;
		sprintf(str, "%.2fKB", b);
	}else if(bytes<(1024*1024*1024)){
		double b= (double)bytes/(1024.0*1024);
		sprintf(str, "%.2fMB", b);
	}else{
		double b= (double)bytes/(1024.0*1024*1024);
		sprintf(str, "%.2fGB", b);
	}
}

template<typename K>
float bench_kernel(K k, int warmup, int loop)
{
    hipEvent_t start_ev, stop_ev;
    HIP_CALL(hipEventCreate(&start_ev));
    HIP_CALL(hipEventCreate(&stop_ev));

    for(int i=0;i<warmup;i++)
        k();

    HIP_CALL(hipEventRecord(start_ev, 0));
    for(int i=0;i<loop;i++)
        k();
    HIP_CALL(hipEventRecord( stop_ev, 0 ));
    HIP_CALL(hipEventSynchronize(stop_ev));

    float ms;
    HIP_CALL(hipEventElapsedTime(&ms,start_ev, stop_ev));
    ms = ms / loop;
    return ms;
}

#define INIT_MEMORY(pixels, host_a, host_b, dev_a, dev_b) { \
    host_a = (int*)malloc((pixels) * sizeof(int)); \
    host_b = (int*)malloc((pixels) * sizeof(int)); \
    rand_vector(host_a, (pixels)); \
    HIP_CALL(hipMalloc(&dev_a, (pixels) * sizeof(int))); \
    HIP_CALL(hipMalloc(&dev_b, (pixels) * sizeof(int))); \
    HIP_CALL(hipMemcpy(dev_a, host_a, (pixels) * sizeof(int), hipMemcpyHostToDevice)); \
    char str[64]; \
    b2s((pixels)*sizeof(float), str); \
    printf("%s ", str); fflush(stdout); \
}

#define FREE_MEMORY() { \
    printf("\n"); \
    fflush(stdout); \
    free(host_a); \
    free(host_b); \
    HIP_CALL(hipFree(dev_a)); \
    HIP_CALL(hipFree(dev_b)); \
}

int main(int argc, char ** argv)
{
    int *host_a, *host_b;
    void *dev_a, *dev_b;

    auto get_gbps = [](double bytes, float ms_){
        return  ((double)bytes)/((double)ms_/1000)/1000000000.0;
    };

#if ENABLE_MEMCPY_PERSISTENT == 1
    {

        int64_t pixels = CodegenSettings<MEMCPY_PERSISTENT_mode>::PIXELS_MB * 1024 * 1024;
        INIT_MEMORY(pixels, host_a, host_b, dev_a, dev_b);
        
        auto k = [=](){
            auto k = memcpy_persistent<
                CodegenSettings<MEMCPY_PERSISTENT_mode>::BS, 
                CodegenSettings<MEMCPY_PERSISTENT_mode>::GS, 
                CodegenSettings<MEMCPY_PERSISTENT_mode>::UNROLL>{{dev_a, dev_b, pixels * 4}};
            return k;
        }();

        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
#if MEMCPY_PERSISTENT_VERIFY == 1
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
#endif
        printf("memcpy_persistent -->> %.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);

        FREE_MEMORY();
    }
#endif

#if ENABLE_MEMREAD_STREAM == 1
    {
        int64_t pixels = CodegenSettings<READ_mode>::PIXELS_MB * 1024 * 1024;
        INIT_MEMORY(pixels, host_a, host_b, dev_a, dev_b);

        auto k = [=](){
            auto k = memread_stream<
                CodegenSettings<READ_mode>::BS, 
                CodegenSettings<READ_mode>::GS, 
                CodegenSettings<READ_mode>::UNROLL>{{dev_a, dev_b, pixels * 4}};
            return k;
        }();
        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
#if READ_VERIFY == 1
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
        printf("memread_stream -->> %.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);
#else
        printf("memread_stream -->> %.3f(GB/s) ", get_gbps(pixels * sizeof(int), ms)); fflush(stdout);
#endif
        FREE_MEMORY();
    }
#endif

#if ENABLE_MEMCPY_STREAM == 1
    {
        int64_t pixels = CodegenSettings<MEMCPY_mode>::PIXELS_MB * 1024 * 1024;
        INIT_MEMORY(pixels, host_a, host_b, dev_a, dev_b);
        
        auto k = [=](){
            auto k = memcpy_stream<
                CodegenSettings<MEMCPY_mode>::BS, 
                CodegenSettings<MEMCPY_mode>::GS, 
                CodegenSettings<MEMCPY_mode>::UNROLL>{{dev_a, dev_b, pixels * 4}};
            return k;
        }();

        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
#if MEMCPY_STREAM_VERIFY == 1
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
#endif
        printf("memcpy_stream -->> %.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);
        FREE_MEMORY();
    }
#endif

#if ENABLE_MEMCPY_STREAM_ASYNC == 1
    {
        int64_t pixels = CodegenSettings<MEMCPY_ASYNC_mode>::PIXELS_MB * 1024 * 1024;
        INIT_MEMORY(pixels, host_a, host_b, dev_a, dev_b);
        
        auto k = [=](){
            auto k = memcpy_stream_async<
                CodegenSettings<MEMCPY_ASYNC_mode>::BS, 
                CodegenSettings<MEMCPY_ASYNC_mode>::GS, 
                CodegenSettings<MEMCPY_ASYNC_mode>::UNROLL>{{dev_a, dev_b, pixels * 4}};
            return k;
        }();
        
        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
#if MEMCPY_STREAM_ASYNC_VERIFY == 1
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
#endif
        printf("memcpy_stream_async -->> %.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);
        FREE_MEMORY();
    }
#endif

#if ENABLE_MEMCPY_STREAM_SWIZZLED == 1
    {
        int64_t pixels = CodegenSettings<MEMCPY_SWIZZLED_mode>::PIXELS_MB * 1024 * 1024;
        INIT_MEMORY(pixels, host_a, host_b, dev_a, dev_b);
        
        auto k = [=](){
            auto k = memcpy_stream_swizzled<
                CodegenSettings<MEMCPY_SWIZZLED_mode>::BS, 
                CodegenSettings<MEMCPY_SWIZZLED_mode>::GS, 
                CodegenSettings<MEMCPY_SWIZZLED_mode>::CHUNKS,
                CodegenSettings<MEMCPY_SWIZZLED_mode>::INNER>{{dev_a, dev_b, pixels * 4}};
            return k;
        }();
        
        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
#if MEMCPY_STREAM_SWIZZLED_VERIFY == 1
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
#endif
        printf("memcpy_stream_swizzled -->> %.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);
        FREE_MEMORY();
    }
#endif

}
