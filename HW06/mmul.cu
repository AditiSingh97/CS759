#include <cublas_v2.h>
#include <cuda.h>
#include "mmul.h"

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n){
    const float alpha = 1.0;
    const float beta = 1.0;
    const float *alpha_ptr = &alpha;
    const float *beta_ptr = &beta;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha_ptr, A, n, B, n, beta_ptr, C, n);
    cudaDeviceSynchronize();
}
