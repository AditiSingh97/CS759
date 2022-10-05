#include <cstdio>
#include <cuda.h>
#include <random>

#include "matmul.cuh"


int main(int argc, char* argv[]) {

    if(argc != 3){
        exit(-1);
    }

    long unsigned int n = atoi(argv[1]);
    long unsigned int threads_per_block = atoi(argv[2]);
    long unsigned int SIZE = n*n;

    //allocating host memory for input matrices
    float *h_A, *h_B;
    h_A = new float[SIZE];
    h_B = new float[SIZE];

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    for(long unsigned int i = 0; i < SIZE; i++){
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    // Allocate vectors in device memory
    float *A, *B, *C;
    cudaMalloc(&A, SIZE*sizeof(float));
    cudaMalloc(&B, SIZE*sizeof(float));
    cudaMalloc(&C, SIZE*sizeof(float));

    //copy from host memory to device memory
    cudaMemcpy(A, h_A, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, SIZE*sizeof(float), cudaMemcpyHostToDevice);

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call
    matmul(A, B, C, n, threads_per_block);
    cudaEventRecord(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    //allocate host memory for product matrix
    float *h_C;
    h_C = new float[SIZE];

    //copying output from device to host memory
    cudaMemcpy(h_C, C, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    //printing output
    
    std::printf("%f\n%f\n", h_C[SIZE - 1], ms);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    delete [] h_A;
    delete [] h_B;
    delete [] h_C;

    return 0;
}