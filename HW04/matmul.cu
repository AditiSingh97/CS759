#include <cmath>
#include <cstdio>
#include <cuda.h>

#include "matmul.cuh"


__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    if((blockIdx.x *blockDim.x + threadIdx.x) <= (n*n)){
        //elem: absolute position of new element in C pointer
        long unsigned int elem = blockIdx.x *blockDim.x + threadIdx.x;
        //row: row of matrix corresponding to C which has elem
        long unsigned row = elem/n;
        //col: column of matrix corresponding to C which has elem
        long unsigned col = elem%n;
        //initializing C[elem] to be 0
        C[elem] = 0.0;
        //for loop to run through the appropriate row and column of A and B respectively
        for(long unsigned i = 0; i < n; i++){
            C[elem] += A[row * n + i] * B[i * n + col];
        }
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    //computing the correct number of blocks
    double q = ((double)(n*n))/((double)threads_per_block);
    long unsigned int num_blocks = ceil(q);
    //kernel call
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
