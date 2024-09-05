#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CALL(call)                                      \
    do {                                                    \
        hipError_t err = call;                              \
        if (err != hipSuccess) {                            \
            printf("[hiperror](%d) fail to call %s\n",      \
                   (int)err, #call);                        \
            exit(0);                                        \
        }                                                   \
    } while (0)

int main() {
    hipDevice_t device;
    HIP_CALL(hipGetDevice(&device));

    int numCUs = [&]() {
        hipDeviceProp_t devProp;
        HIP_CALL(hipGetDeviceProperties(&devProp, device));
        return devProp.multiProcessorCount;
    }();

    std::cout << numCUs <<  std::endl;

    return 0;
}
