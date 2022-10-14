#include <cuda.h>
#include <cmath>
#include <cstdio>

#include "reduce.cuh"

//code taken from CS759 lecture 12
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
    else{
	    sdata[tid] = 0.0;
    }
    //initializing to 0
    __syncthreads();

    //to construct g_odata
    //reduction #3
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	    if ((i < n) && (tid < s)) {
		    sdata[tid] += sdata[tid + s];
	    }
	    __syncthreads();
    }
    
    //copying data to global memory
    if(threadIdx.x == 0){
	    g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){
    //computing the correct number of blocks
    double q = ((double)(N))/((double)(2.0 * threads_per_block));
    long unsigned int num_blocks = ceil(q);
    //kernel call
    //dereferecing for simpler code
    float *temp_ip = *input;
    float *temp_op = *output;
    do{
	    reduce_kernel<<<num_blocks, threads_per_block, threads_per_block*sizeof(float)>>>(temp_ip, temp_op, N);
	    //swaping pointers before passing to next call
	    float *temp = temp_ip;
	    temp_ip = temp_op;
	    temp_op = temp;
	    //updating N and num_blocks before next call
	    N = num_blocks;
	    num_blocks = ceil((double)(N)/(double)(2.0 * threads_per_block));
    }while(N > 1);

    *input = temp_ip;
    *output = temp_op;
    cudaDeviceSynchronize();
}
