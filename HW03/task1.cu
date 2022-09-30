#include <cstdio>
#include <cuda.h>

const unsigned int nThreads = 8;

//kernel to compute factorial
__global__ void factorial() {
    if (threadIdx.x < nThreads) {
        int fact = 1;
        //increment thread ID by 1 because IDs go from 0-7 and we need factorial from 1-8
        for(int i = 1; i <= threadIdx.x+1; i++){
            fact = fact*i;
        }
        std::printf("%d!=%d\n", threadIdx.x+1, fact);
    }
}

int main(int argc, char* argv[]) {
    //kernel call
    factorial<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}