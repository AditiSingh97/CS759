#include <cmath>
#include <cstdio>
#include <cuda.h>

#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern __shared__ float shMem[];
    //variable for mask size for easy referring
    unsigned int M = 2*R + 1;

    //first M threads populate mask elements
    if(threadIdx.x < M){
        shMem[threadIdx.x] = mask[threadIdx.x];
    }
    //blockDimx.x + 2 * R
    if((blockIdx.x  * blockDim.x + threadIdx.x) < n){
	    //computing global thread id
        unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//padding left side of image window + first element of window
        if(threadIdx.x == 0){
            for(long int i = M+R; i >= M; i--){
                if((threadid - (M + R - i)) < 0){
                    shMem[i] = 1.0;
                }else{
                    shMem[i] = image[threadid - (M + R - i)];
                }
            }
        }else{
            //M (for mask) ; R (left padding) ; blockDim.x (==threads_per_block) (for each output) ; R (right padding)
		//padding right side of image window + last element of window
            if((threadIdx.x == blockDim.x - 1) || (threadid == n-1)){
                if(threadid != n-1)
                {
                    unsigned long offset = M + blockDim.x + R - 1;
                    for(long int i = offset; i <= offset + R; i++){
                        if(threadid + (i - offset) > (n-1)){
                            shMem[i] = 1.0;
                        }else{
                            shMem[i] = image[threadid + (i - offset)];
                        }
                    }
                }else{
                    shMem[M + R + threadIdx.x] = image[threadid];
                    for(long int i = 1; i <= R; i++){
                        shMem[M + threadIdx.x + R + i] = 1.0;
                    }
                }
            }else{
		    //elements between the boundaries of windows
                shMem[M + R + threadIdx.x] = image[threadid];
            }
        }
    }

    //synchronize to ensure all threads see this shared memory write
    __syncthreads();
    //output computation
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadid < n){
	    //position in shared memory where this output will go
        int op_off = M + blockDim.x + 2*R + threadIdx.x;
	//computing position of central element of this window
        int pos = M + R + threadIdx.x;
	//resetting to 0.0
        shMem[op_off] = 0.0;
	//lower limit of window
        signed int neg = (-1) * (signed)R;
	//for loop for computing convolution
        for(signed int j = neg; j <= (signed)R; j++){
            shMem[op_off] += shMem[((signed)pos+j)] * shMem[(j+(signed)R)];
        }
	//copying to global memory
        output[threadid] = shMem[op_off];
    }
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block){
    //computing the correct number of blocks
    double q = ((double)(n))/((double)threads_per_block);
    long unsigned int num_blocks = ceil(q);
    //kernel call
    stencil_kernel<<<num_blocks, threads_per_block, (4*R+1 + 2*threads_per_block) * sizeof(float)>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}
