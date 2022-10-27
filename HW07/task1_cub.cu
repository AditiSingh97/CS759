#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
#include <random>
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char *argv[]) {
	if(argc != 2){
		exit(-1);
	}

    const size_t num_items = atoi(argv[1]);
    // Set up host arrays
    float *h_in = new float[num_items];

    //random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    for(long unsigned int i = 0; i < num_items; i++){
        h_in[i] = dist(gen);
    }

    float sum = 0.0;
    for (unsigned int i = 0; i < num_items; i++)
        sum += h_in[i];

    // Set up device arrays
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * num_items));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));
    // Setup device output array
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    //creating cuda timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Do the actual reduce operation using CUB
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));

    std::cout << gpu_sum << std::endl << ms << std::endl;
    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}
