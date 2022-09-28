#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <random>

const unsigned int nThreads = 8;

//kernel to compute the result
__global__ void series(int *dA, int a) {
    if(threadIdx.x < nThreads)
    {
        dA[blockIdx.x * nThreads + threadIdx.x] = a * threadIdx.x + blockIdx.x;
    }
}

int main(int argc, char* argv[]) {

    // Allocate vectors in device memory
    int* dA;
    cudaMalloc(&dA, 16*sizeof(int));

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(10,20);

    int a = dist(gen);
    //kernel call
    series<<<2, 8>>>(dA, a);
    cudaDeviceSynchronize();

    //result array creation in host memory
    int * hA = new int[16];

    //copy from device memory to host memory
    cudaMemcpy(hA, dA, 16*sizeof(int), cudaMemcpyDeviceToHost);

    //printing output
    for(int i = 0; i < 16; i++){
        std::printf("%d ", hA[i]);
    }
    std::printf("\n");
    return 0;
}