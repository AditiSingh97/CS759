#include <cstdio>
#include <cuda.h>
#include <random>

#include "stencil.cuh"


int main(int argc, char* argv[]) {

    if(argc != 4){
        exit(-1);
    }

    unsigned int n = atoi(argv[1]);
    unsigned int R = atoi(argv[2]);
    unsigned int threads_per_block = atoi(argv[3]);

    //allocating host memory for input matrices
    float *h_image, *h_output, *h_mask;
    h_image = new float[n];
    h_output = new float[n];
    h_mask = new float[2 * R + 1];

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    //random number generation
    //image initialization
    for(long unsigned int i = 0; i < n; i++){
        h_image[i] = dist(gen);
    }
    //mask initialization
    for(long unsigned int i = 0; i < 2*R + 1; i++){
        h_mask[i] = dist(gen);
    }

    // Allocate vectors in device memory
    float *image, *output, *mask;
    cudaMalloc(&image, n*sizeof(float));
    cudaMalloc(&output, n*sizeof(float));
    cudaMalloc(&mask, (2*R+1)*sizeof(float));

    //copy from host memory to device memory
    cudaMemcpy(image, h_image, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mask, h_mask, (2*R + 1)*sizeof(float), cudaMemcpyHostToDevice);

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //kernel call
    stencil(image, mask, output, n, R, threads_per_block);
    cudaEventRecord(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    //copying output from device to host memory
    cudaMemcpy(h_output, output, n*sizeof(float), cudaMemcpyDeviceToHost);

    //printing output
    
    std::printf("%f\n%f\n", h_output[n - 1], ms);

    cudaFree(image);
    cudaFree(output);
    cudaFree(mask);

    delete [] h_image;
    delete [] h_mask;
    delete [] h_output;

    return 0;
}
