#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include "convolution.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;

int main(int argc, char **argv){
    if(argc != 3){
        exit(-1);
    }

    std::size_t n = atoi(argv[1]);
    std::size_t m = atoi(argv[2]);

    float *image = new float[n*n];
    float *mask = new float[m*m];
    float *output = new float[n*n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist1(-10.0, 10.0);
    std::uniform_real_distribution<> dist2(-1.0, 1.0);

    for(std::size_t i = 0; i < n; i++){
        for(std::size_t j = 0; j < n; j++){
            *(image + i*n + j) = dist1(gen);
        }
    }

    for(std::size_t i = 0; i < m; i++){
        for(std::size_t j = 0; j < m; j++){
            *(mask + i*m + j) = dist2(gen);
        }
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_msec.count() << std::endl << *output << std::endl << *(output + (n*n) - 1) << std::endl;

    delete image;
    delete mask;
    delete output;
}