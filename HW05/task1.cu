#include <cstdio>
#include <random>
#include <cuda.h>

#include "reduce.cuh"

void verifier(float *A, float *result, unsigned int n){
	double threshold = 1e-5;
	float ground_truth = 0.0;
	for(unsigned int i = 0; i < n; i++){
		ground_truth += A[i];
	}
//	printf("ground_truth: %f, result: %f\n", ground_truth, result[0]);
//	printf("difference between result and ground_truth: %f\n", result[0] - ground_truth);
}

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
        h_A[i] = dist(gen);
    }
    for(unsigned int i = 0; i < num_blocks; i++){
	   h_B[i] = (float)0.0;
    } 

    // Allocate vectors in device memory
    float *A, *B;
    cudaMalloc(&A, N*sizeof(float));
    cudaMalloc(&B, num_blocks*sizeof(float));

    //copy from host memory to device memory
    cudaMemcpy(A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, num_blocks*sizeof(float), cudaMemcpyHostToDevice);

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call
    reduce(&A, &B, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms3;
    cudaEventElapsedTime(&ms3, start, stop);
    printf("%f\n%f\n%f\n", C_double[0], C_double[SIZE-1], ms3);

    cudaFree(A_double);
    cudaFree(B_double);
    cudaFree(C_double);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float *result = new float;
    cudaMemcpy(result, A, 1*sizeof(float), cudaMemcpyDeviceToHost);
    printf("%.6f\n%.6f\n", result[0], ms);
    cudaFree(A);
    cudaFree(B);
    delete [] h_A;
    delete [] h_B;
    delete result;
    return 0;
}
