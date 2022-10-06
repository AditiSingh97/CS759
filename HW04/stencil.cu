#include <cmath>
#include <cstdio>
#include <cuda.h>

#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern _shared_ float shMem[];
    if(threadIdx.x < (2*R + 1)){
        shMem[threadIdx.x] = mask[threadIdx.x];
    }
    if((threads_per_block * blockIdx.x + threadIdx.x) < n){
        shMem[2*R + 1 + threadIdx.x] = image[threadIdx.x];
        sh_output[2*R + 1 + thread_per_block + threadIdx.x] = 0.0;
    }

    __syncthreads();
    for(long int j = (-1)*R; j <= R; j++){
        float elem = 0.0;
        if((threadIdx.x + j < 0) || (threadIdx.x + j > n-1)){
            elem = 1;
        }else{
            elem = sh_image[threadIdx.x];
        }
        sh_output[threadIdx.x] += elem * mask[j + R];
    }

    output[threads_per_block * blockIdx.x + threadIdx.x] = sh_output[threadIdx.x];
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block){
    //computing the correct number of blocks
    double q = ((double)(n))/((double)threads_per_block);
    long unsigned int num_blocks = ceil(q);
    //kernel call
    stencil_kernel<<<num_blocks, threads_per_block>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}