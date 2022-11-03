#include <iostream>
#include <random>
#include <vector>
#include <cstddef>
#include <omp.h>
#include <chrono>

#include "matmul.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

//verifier function to check correctness
void mmul_verif(const float* A, const float* B, float* C, const unsigned int n){
    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < n; j++){
            *(C + i*n + j) = 0.0;
        }
    }
    for(size_t i = 0; i < n; i++){
        for(size_t k = 0; k < n; k++){
            for(size_t j = 0; j < n; j++){
                *(C + i*n + j) += *(A + i*n + k) * (*(B + k*n + j));
            }
        }
    }
}

int main(int argc, char **argv){
    if(argc != 3){
        exit(-1);
    }

    //parsing command line args
    unsigned int n = atoi(argv[1]);
    int t = atoi(argv[2]);

    unsigned int SIZE = n*n;
    //array declarations
    float *A = new float[SIZE];
    float *B = new float[SIZE];
    float *C = new float[SIZE];
//    float *C_gt = new float[SIZE];

    //generating random floats
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < n; j++){
            *(A + i*n + j) = dist(gen);
            *(B + i*n + j) = dist(gen);
            *(C + i*n + j) = 0;
        }
    }

    //setting number of threads
    omp_set_num_threads(t);
    //initializing timer objects
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    //function call
    mmul(A, B, C, n);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << *C << std::endl << *(C + SIZE - 1) << std::endl << duration_msec.count() << std::endl;
    /*
    mmul_verif(A, B, C_gt, n);

    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < n; j++){
            if((*(C_gt+i*n+j) - *(C+i*n+j)) > 1e-5){
                std::cout << i <<"; " << j << " diff = " << *(C_gt+i*n+j)-*(C+i*n+j) << std::endl;
            }
        }
    }
    */

    return 0;
}
