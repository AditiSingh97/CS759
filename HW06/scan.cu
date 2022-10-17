#include <cuda.h>
#include <cstdio>
#include <cmath>

#include "scan.cuh"

__global__ void scan_kernel(const float *g_idata, float *g_odata, unsigned int n) {
    extern volatile __shared__  float temp[]; // allocated on invocation

    int thid = threadIdx.x;
    int pout = 0, pin = 1;
    // load input into shared memory. 
    // **exclusive** scan: shift right by one element and set first output to 0
    temp[thid] = g_idata[thid];
    __syncthreads();
    unsigned int threshold = ceil(log2((double)n));

    for(unsigned int power = 0; power<threshold; power++) {
        unsigned offset = pow(2, power);
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pin;

        if (thid >= offset){
		temp[pout*n+thid] = temp[pin*n+thid] + temp[pin*n+thid - offset];
        }
        else{
		temp[pout*n+thid] = temp[pin*n+thid];
	}

        __syncthreads(); // I need this here before I start next iteration 
    }
    
    g_odata[thid] = temp[pout*n+thid]; // write output
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block){
	//kernel calls happen here
	scan_kernel<<<1,threads_per_block, 2*threads_per_block*sizeof(float)>>>(input, output, n);
	cudaDeviceSynchronize();
}

