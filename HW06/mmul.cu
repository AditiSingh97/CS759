#include <cublas_v2.h>
#include <cuda.h>
#include "mmul.h"

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n){
    const float alpha = 1.0;
    const float beta = 1.0;
    const float *ptr1 = &alpha;
    const float *ptr2 = &beta;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, ptr1, A, n, B, n, ptr2, C, n);
    cudaDeviceSynchronize();

}
