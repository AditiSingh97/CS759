#include <cmath>
#include <cstdio>
#include <cuda.h>

#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern _shared_ float shMem[];
    unsigned int M = 2*R + 1;
    if(threadIdx.x < (M)){
        shMem[threadIdx.x] = mask[threadIdx.x];
    }
    //blockDimx.x + 2 * R
    if(blockIdx.x * blockDim.x + threadIdx.x < n){
        unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
        if(threadIdx.x == 0){
            for(long int i = M; i <= M + R; i++){
                if((threadid - (i - M)) < 0){
                    shMem[i] = 1.0;
                }else{
                    shMem[i] = image[threadid - (i - M)];
                }
            }
        }else{
            //M (for mask) ; R (left padding) ; blockDim.x (==threads_per_block) (for each output) ; R (right padding)
            if(threadIdx.x == blockDim.x - 1){
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
            }
        }
    }


    __syncthreads();
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long op_off = M + blockDim.x + 2*R + threadIdx.x;
    unsigned long pos = M + R + threadIdx.x;
    shMem[op_off] = 0.0;
    for(long int j = (-1)*R; j <= R; j++){
        shMem[op_off] += shMem[pos + j] * shMem[j + R];
    }

    output[threadid] = shMem[op_off];
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block){
    //computing the correct number of blocks
    double q = ((double)(n))/((double)threads_per_block);
    long unsigned int num_blocks = ceil(q);
    //kernel call
    stencil_kernel<<<num_blocks, threads_per_block>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}