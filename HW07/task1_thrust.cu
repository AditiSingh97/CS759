#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <iostream>
#include <cuda.h>
#include <random>

int main(int argc, char *argv[]) {
    if(argc != 2){
        exit(-1);
    }
    unsigned n = atoi(argv[1]);
    // generate n random float numbers on the host
    thrust::host_vector<float> h_vec(n);
    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for(unsigned i = 0; i< n; i++){
	    h_vec[i] = dist(gen);
    }
    
    // transfer data to the device
    thrust::device_vector<float> d_vec(n);
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin()); 
    
    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //thrust function call
    float result = thrust::reduce(d_vec.begin(), d_vec.end(), (float)0.0, thrust::plus<float>());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    std::cout << result << std::endl << ms << std::endl;
    return 0;
}

