#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <iostream>
#include <cuda.h>

#include "count.cuh"

int main(int argc, char *argv[]) {
    if(argc != 2){
        exit(-1);
    }
    unsigned n = atoi(argv[1]);
    // generate n random int numbers on the host
    // create a minstd_rand object to act as our source of randomness
    thrust::minstd_rand rng;
    // create a uniform_real_distribution to produce ints
    thrust::uniform_real_distribution<float> dist(0,500);
    thrust::host_vector<int> h_vec(n);
    for(unsigned int i = 0; i < n; i++){
	    h_vec[i] = (int)dist(rng);
    }
    
    // transfer data to the device
    thrust::device_vector<int> d_in(n);
    thrust::copy(h_vec.begin(), h_vec.end(), d_in.begin()); 
    thrust::device_vector<int> values(n), counts(n);
    for(unsigned int i = 0; i < n; i++){
	    values[i] = 0;
	    counts[i] = 0;
    }
    
    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //thrust function call
    count(d_in, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    std::cout << values[values.size()-1] << std::endl << counts[counts.size()-1] << std::endl << ms << std::endl;
    return 0;
}

