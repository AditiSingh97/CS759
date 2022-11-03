#include <iostream>
#include <random>
#include <chrono>
#include <cstddef>
#include <omp.h>

#include "msort.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;

int main(int argc, char **argv){
    if(argc != 4){
        exit(-1);
    }
    size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);
    size_t ts = atoi(argv[3]);

    int *arr = new int[n];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1000, 1000);

    for(std::size_t i = 0; i < n; i++){
            arr[i] = (int)dist(gen);
        }

    omp_set_num_threads(t);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    msort(arr, n, ts);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << *arr << std::endl << *(arr + n - 1) << std::endl << duration_msec.count() << std::endl;

    return 0;
}
