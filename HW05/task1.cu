#include <cstdio>
#include <random>
#include <cuda.h>

#include "reduce.cuh"

int main(int argc, char* argv[]) {
    if(argc != 3){
        exit(-1);
    }

    unsigned int N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    //allocating host memory for input matrices
    unsigned int num_blocks = ceil((double)N/(double)(2.0*threads_per_block));
    float *h_A, *h_B;
    h_A = new float[N];
    h_B = new float[num_blocks];

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    for(long unsigned int i = 0; i < N; i++){
        h_A[i] = (float)i;
    }

    // Allocate vectors in device memory
    float *A, *B;
    cudaMalloc(&A, N*sizeof(float));
    cudaMalloc(&B, num_blocks*sizeof(float));

    //copy from host memory to device memory
    cudaMemcpy(A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call
    reduce(&A, &B, N, threads_per_block);
    cudaEventRecord(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_A, A, 1*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, B, 1*sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("final answer A %f\n", h_A[0]);

    return 0;
}
