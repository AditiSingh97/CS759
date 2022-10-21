#include <cuda.h>
#include <cstdio>
#include <cmath>

#include "scan.cuh"

//some code taken from CS759 Lecture 15
__global__ void hillis_steele(const float *g_idata, float *g_odata, float *block_sum, unsigned int n, bool record_block_sum = false) {
    extern volatile __shared__  float temp[]; // allocated on invocation

    int thid = threadIdx.x;
    unsigned int idx = thid + blockDim.x * blockIdx.x;
    if(idx < n){
    unsigned int n_block = 0;
    if((blockIdx.x+1) * blockDim.x < n){
	    n_block = blockDim.x;
    }else{
	    n_block = n-(blockIdx.x*blockDim.x);
    }
    int pout = 0, pin = 1;
    // load input into shared memory. 
    // **exclusive** scan: shift right by one element and set first output to 0
    temp[thid] = g_idata[idx];
    __syncthreads();
    unsigned int threshold = ceil(log2((double)n_block));

    for(unsigned int power = 0; power<threshold; power++) {
        unsigned offset = pow(2, power);
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pin;

        if (thid >= offset){
		temp[pout*n_block+thid] = temp[pin*n_block+thid] + temp[pin*n_block+thid - offset];
        }
        else{
		temp[pout*n_block+thid] = temp[pin*n_block+thid];
        }

        __syncthreads(); // I need this here before I start next iteration 
    }
    
    g_odata[idx] = temp[pout*n_block+thid]; // write output
    __syncthreads();
    if(record_block_sum)
    {
        if((thid == blockDim.x-1)||(idx == n-1)){
            block_sum[blockIdx.x] = g_odata[idx];
        }
    }
    }
}

//helper kernel to apply scanned sum of previous blocks to current block
__global__ void apply_sum_to_blocks(float *g_odata, float *apply_sum, unsigned int n){
    if(blockIdx.x > 0){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n){
            g_odata[idx] += apply_sum[blockIdx.x-1];
        }
    }

    __syncthreads();
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block){
	//kernel calls happen here
	int num_blocks = ceil((double)n/(double)threads_per_block);
	float *block_sum, *apply_sum;
	cudaMallocManaged(&block_sum, num_blocks*sizeof(float));
	cudaMallocManaged(&apply_sum, num_blocks*sizeof(float));
	for(int i = 0; i < num_blocks; i++){
		block_sum[i] = (float)0.0;
		apply_sum[i] = (float)0.0;
	}
	hillis_steele<<<num_blocks, threads_per_block, 2*threads_per_block*sizeof(float)>>>(input, output, block_sum, n, true);
	cudaDeviceSynchronize();
	
	//scanned each block internally and stored their full sum in block_sum, now perform inclusive scan on block_sum
    	//n < threads_per_block * threads_per_block, so in next step we only need 1 block
    	hillis_steele<<<1, num_blocks, 2*num_blocks*sizeof(float)>>>(block_sum, apply_sum, (float *)nullptr, n, false);
    	cudaDeviceSynchronize();
    	
	//computed extra sum to be added to each block respectively, now apply to the blocks
    	apply_sum_to_blocks<<<num_blocks, threads_per_block>>>(output, apply_sum, n);
    	cudaDeviceSynchronize();

    	cudaFree(block_sum);
    	cudaFree(apply_sum);
}

