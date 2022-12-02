#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cstddef>
#include <omp.h>
#include <chrono>

#include "reduce.h"

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

    //generating random floats
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for(int i = 0; i < n; i++){
        arr[i] = dist(gen);
    }

    //setting number of threads
    omp_set_num_threads(t);
    //initializing timer objects
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    //function call
    float res = reduce(arr, 0, n);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << res << std::endl << duration_msec.count() << std::endl; 

    delete [] arr;

    return 0;
}
