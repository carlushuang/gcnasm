#ifndef CONFIG
#define CONFIG

#include <hip/hip_runtime.h>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

template <typename T>
class Matrix_2d
{
public:
    int rows;
    int cols;
    int padding;
    int length;
    bool type; //Host: 1; Device: 0
    T *data;
public:
    Matrix_2d(): rows(0), cols(0), padding(0), length(0), type(0), data(nullptr) {}
    Matrix_2d(int r, int c, int p, bool t): rows(r), cols(c), padding(p), type(t), length(r * (c + p)), data(new T[r * (c + p)]) {}
    ~Matrix_2d() {
        if (data != nullptr && type) {
            delete [] data;
        }
    }
    void initMem() {
        if (!data)
            data = new T[rows * (cols + padding)];
    }
};

typedef struct Result
{
    bool isValid;
    int bdx;
    int gdx;
    float gbps;
} Result;

#endif
