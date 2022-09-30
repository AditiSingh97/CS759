#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <random>
#include "vscale.cuh"

int main(int argc, char* argv[]) {
    if(argc != 2){
        exit(-1);
    }

    unsigned int n = atoi(argv[1]);
    //Allocate vectors in host memory
    float *a = new float[n];
    float *b = new float[n];

    // Allocate vectors in device memory
    float *da, *db;
    cudaMalloc(&da, n*sizeof(float));
    cudaMalloc(&db, n*sizeof(float));

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist1(-10.0,10.0);
    std::uniform_real_distribution<> dist2(0.0,1.0);

    for(unsigned int i = 0; i < n; i++){
        a[i] = dist1(gen);
        b[i] = dist2(gen);
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(da, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n*sizeof(float), cudaMemcpyHostToDevice);

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call
    vscale<<<n/512, 512>>>(da, db, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    //copy from device memory to host memory
    cudaMemcpy(b, db, n*sizeof(float), cudaMemcpyDeviceToHost);

    //printing results
    std::printf("%f\n%f\n%f\n", ms, b[0], b[n-1]);

    //freeing memory
    delete [] a;
    delete [] b;
    cudaFree(da);
    cudaFree(db);

    return 0;
}