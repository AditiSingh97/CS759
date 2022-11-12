#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cstddef>
#include <omp.h>
#include <chrono>

#include "cluster.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;

int main(int argc, char **argv){
    if(argc != 3){
        exit(-1);
    }

    //parsing command line args
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);

    //array declarations
    float *arr = new float[n];
    float *centers = new float[t];
    float *dists = new float[t];

    //generating random floats
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, n);

    for(int i = 0; i < n; i++){
        arr[i] = dist(gen);
    }

    std::sort(arr, arr + n);
    
    for(int i = 0; i < t; i++){
        centers[i] = ((float)(2*(i+1) - 1)*n)/((float)t*2);
        dists[i] = 0.0;
    }

    //setting number of threads
    omp_set_num_threads(t);
    //initializing timer objects
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    //function call
    cluster(n, t, arr, centers, dists);
    end = high_resolution_clock::now();

    float max_distance = 0.0;
    float max_partition = -1;
    for(int j = 0; j < t; j++){
        if(dists[j] > max_distance){
            max_distance = dists[j];
            max_partition = j;
        }
    }
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << max_distance << std::endl << max_partition << std::endl << duration_msec.count() << std::endl; 

    delete [] arr;
    delete [] centers;
    delete [] dists;
    return 0;
}
