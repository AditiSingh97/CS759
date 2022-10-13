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
    g_odata[blockIdx.x] = 0;
    __syncthreads();

    //to construct g_odata
    //if we have single block

    unsigned int new_n = 0;
    if(n <= blockDim.x){
	    new_n = n;
    }else{
	    new_n = blockDim.x;
    } 

    if(((blockIdx.x * blockDim.x + threadIdx.x) < n) && (threadIdx.x > 0)){
	     float t = atomicAdd(&sdata[0], sdata[threadIdx.x]);
	    __syncthreads();
    }

    if(threadIdx.x == 0){
	    g_odata[blockIdx.x] = sdata[0];
    }


}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){
    //computing the correct number of blocks
    double q = ((double)(N))/((double)(2.0 * threads_per_block));
    long unsigned int num_blocks = ceil(q);
    //kernel call
    float *temp_ip = *input;
    float *temp_op = *output;
    do{
	    reduce_kernel<<<num_blocks, threads_per_block, num_blocks*threads_per_block*sizeof(float)>>>(temp_ip, temp_op, N);
	    float *temp = temp_ip;
	    temp_ip = temp_op;
	    temp_op = temp;
	    float *deref = new float[num_blocks];
	    *input = temp_ip;
	    *output = temp_op;
	    N = num_blocks;
	    num_blocks = ceil((double)(N)/(double)(2.0 * threads_per_block));

    }while(N > 1);
}
