#include <iostream>
#include <random>
#include <chrono>
#include "scan.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv){
    if(argc != 2){
        exit(-1);
    }

    int n = atoi(argv[1]);
    float *arr = new float[n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    for(int i = 0; i < n; i++){
        *(arr + i) = dist(gen);
    }

    float *output = new float[n];

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    scan(arr, output, n);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_msec.count() << std::endl << *output << std::endl << *(output + n -1) << std::endl;

    delete arr;
    delete output;
}