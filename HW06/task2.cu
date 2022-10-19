#include <cstdio>
#include <random>
#include <cuda.h>

#include "scan.cuh"

int main(int argc, char* argv[]) {
    if(argc != 3){
        exit(-1);
    }

    unsigned int n = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    //allocating host memory for input matrices
    float *input, *output;
    cudaMallocManaged(&input, n*sizeof(float));
    cudaMallocManaged(&output, n*sizeof(float));

    for(unsigned int i = 0; i < n; i++){
	    input[i] = (float)i+1;
	    output[i] = 0.0;
    }

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    scan(input, output, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get the elapsed time in milliseconds
    float ms = 0.0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n%f", output[n-1], ms);

    cudaFree(input);
    cudaFree(output);
    return 0;
}

