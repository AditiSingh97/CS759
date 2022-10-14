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
    int *A_int, *B_int, *C_int;
    cudaMallocManaged(&A_int, SIZE*sizeof(int));
    cudaMallocManaged(&B_int, SIZE*sizeof(int));
    cudaMallocManaged(&C_int, SIZE*sizeof(int));

    //allocating host memory for input matrices
    //running int first but allocating float first for simpler casting of data
    float *A_float, *B_float, *C_float;
    cudaMallocManaged(&A_float, SIZE*sizeof(float));
    cudaMallocManaged(&B_float, SIZE*sizeof(float));
    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    for(unsigned int i = 0; i < SIZE; i++){
        A_float[i] = dist(gen);
	B_float[i] = dist(gen);
    }
    
    for(unsigned int i = 0; i < SIZE; i++){
        A_int[i] = (int)A_float[i];
	B_int[i] = (int)B_float[i];
	C_int[i]  = (int)0.0;
    }
    
    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call - int
    matmul_1(A_int, B_int, C_int, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    

    // Get the elapsed time in milliseconds
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);
    printf("%d\n%d\n%f\n", C_int[0], C_int[SIZE-1], ms1);
    
    cudaFree(A_int);
    cudaFree(B_int);
    cudaFree(C_int);

    cudaMallocManaged(&C_float, SIZE*sizeof(float));
    for(unsigned int i = 0; i < SIZE; i++){
	    C_float[i] = 0.0;
    }
    
    //creating cuda timing variables
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call - float
    matmul_2(A_float, B_float, C_float, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds

    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);
    printf("%f\n%f\n%f\n", C_float[0], C_float[SIZE-1], ms2);

    cudaFree(C_float);

    double *A_double, *B_double, *C_double;
    cudaMallocManaged(&A_double, SIZE*sizeof(double));
    cudaMallocManaged(&B_double, SIZE*sizeof(double));
    cudaMallocManaged(&C_double, SIZE*sizeof(double));

    for(unsigned int i = 0; i < SIZE; i++){
        A_double[i] = (double)A_float[i];
	B_double[i] = (double)B_float[i];
	C_double[i]  = (double)0.0;
    }
    
    cudaFree(A_float);
    cudaFree(B_float);
    //creating cuda timing variables
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call - double
    matmul_3(A_double, B_double, C_double, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms3;
    cudaEventElapsedTime(&ms3, start, stop);
    printf("%f\n%f\n%f\n", C_double[0], C_double[SIZE-1], ms3);

    cudaFree(A_double);
    cudaFree(B_double);
    cudaFree(C_double);

    return 0;
}
