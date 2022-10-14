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
    double *A, *B, *C;
    cudaMallocManaged(&A, SIZE*sizeof(double));
    cudaMallocManaged(&B, SIZE*sizeof(double));
    cudaMallocManaged(&C, SIZE*sizeof(double));

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    for(unsigned int i = 0; i < SIZE; i++){
        A[i] = (double)i;
    }
    for(unsigned int i = 0; i < n; i++){
	    B[i*n + i] = (double)1.0;
    }
    for(unsigned int i = 0; i < n; i++){
	    for(unsigned int j = 0; j < n; j++){
		    std::printf("A[%u][%u]: %f, ", i, j, A[i * n + j]);
	    }
	    std::printf("\n");
    }
    
    for(unsigned int i = 0; i < n; i++){
	    for(unsigned int j = 0; j < n; j++){
		    std::printf("B[%u][%u]: %f, ", i, j, B[i * n + j]);
	    }
	    std::printf("\n");
    }
    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call
    matmul_3(A, B, C, n, block_dim);
    cudaEventRecord(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("final answer C\n");
    for(unsigned int i = 0; i < n; i++){
	    for(unsigned int j = 0; j < n; j++){
		    std::printf("C[%u][%u]=%f, ",i,j,C[i*n+j]);
	    }
	    std::printf("\n");
    }

    return 0;
}
