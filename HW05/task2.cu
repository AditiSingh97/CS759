#include <cstdio>
#include <random>
#include <cuda.h>

#include "matmul.cuh"

int main(int argc, char* argv[]) {
    if(argc != 3){
        exit(-1);
    }

    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);

    unsigned int SIZE = n*n;
    //allocating host memory for input matrices
    float *A, *B, *C;
    cudaMallocManaged(&A, SIZE*sizeof(float));
    cudaMallocManaged(&B, SIZE*sizeof(float));
    cudaMallocManaged(&C, SIZE*sizeof(float));

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    for(long unsigned int i = 0; i < SIZE; i++){
        A[i] = (float)i;
    }
    for(unsigned int i = 0; i < n; i++){
	    B[i*n + i] = 1.0;
    }

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call
    matmul_2(A, B, C, n, block_dim);
    cudaEventRecord(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("final answer C\n");
    for(unsigned int i = 0; i < n; i++){
	    for(unsigned int j = 0; j < n; j++){
		    std::printf("C[%u][%u]=%f\n",i,j,C[i*n+j]);
	    }
    }

    return 0;
}
