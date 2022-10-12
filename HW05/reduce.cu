#include <cuda.h>
#include <cmath>
#include <cstdio>

#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float sdata[];
    // perform first level of reduction upon reading from 
    // global memory and writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    if(i < n){
        if(i+blockDim.x < n){
            sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
        }
        else{
            sdata[tid] = g_idata[i];
        }
    }
    __syncthreads();

    //to construct g_odata
    //if we have single block
    if(gridDim.x > 1){
        g_odata[blockIdx.x * blockDim.x + threadIdx.x] = sdata[tid];
    }
    else{
        //perform reduction
        if(tid > 0){
            s_data[0] += s_data[tid];
        }
    }
}
__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){
    //computing the correct number of blocks
    double q = ((double)(n))/((double)threads_per_block);
    long unsigned int num_blocks = ceil(q);
    //kernel call
    reduce_kernel<<<num_blocks, threads_per_block, N*sizeof(float)>>>(*(input), *(output), N);
    unsigned int new_N = ceil((double)N/(double)2.0);
    reduce_kernel<<<ceil((double)num_blocks/(double)threads_per_block), threads_per_block, new_N*sizeof(float)>>>(*(input), *(output), new_N);
}
