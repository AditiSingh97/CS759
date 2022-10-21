#include <cstdio>
#include <random>
#include <cuda.h>
#include <cublas_v2.h>

#include "mmul.h"

int main(int argc, char* argv[]) {
    if(argc != 3){
        exit(-1);
    }

    unsigned int n = atoi(argv[1]);
    unsigned int n_tests = atoi(argv[2]);

    unsigned int SIZE = n*n;
    //allocating host memory for input matrices
    float *A, *B, *C;

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms = 0.0;
    for(unsigned int i = 0; i < n_tests; i++){
	cudaMallocManaged(&A, SIZE*sizeof(float));
        cudaMallocManaged(&B, SIZE*sizeof(float));
        cudaMallocManaged(&C, SIZE*sizeof(float));
        for(unsigned int j = 0; j < SIZE; j++){
            A[j] = dist(gen);
            B[j] = dist(gen);
            C[j] = dist(gen);
        }
        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaEventRecord(start);
        //kernel call
        mmul(handle, A, B, C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cublasDestroy(handle);
    

        // Get the elapsed time in milliseconds
        float ms1 = 0.0;
        cudaEventElapsedTime(&ms1, start, stop);
        ms += ms1;
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }
    
    printf("Average time: %.6f\n", ms/float(n_tests));
    return 0;
}
