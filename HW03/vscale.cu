#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n){
    if(threadIdx.x < n){
        //checking if thread ID is valid before performing operation
        b[blockIdx.x*512 + threadIdx.x] = a[blockIdx.x*512 + threadIdx.x] * b[blockIdx.x*512 + threadIdx.x];
    }
}