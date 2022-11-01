#include <iostream>
#include <random>
#include <chrono>
#include <cstddef>
#include <omp.h>

#include "convolution.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;

void convolve_verif(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    std::size_t mm = (m-1)/2;
    for(std::size_t i = 0; i < n; i++){
        for(std::size_t j = 0; j < n ; j++){
            float ans = 0.0;
            for(std::size_t k = 0; k < m; k++){
                for(std::size_t l = 0; l < m; l++){
                    std::size_t t1 = i + k - mm;
                    std::size_t t2 = j + l -mm;
                    if(((t1 < 0) || (t1 >= n)) && ((t2 < 0) || (t2 >= n))){
                        ans += 0.0;
                    }else{
                        if(((t1 >= 0) && (t1 < n)) && ((t2 >= 0) && (t2 < n))){
                            ans += *(mask + k*m + l) * (*(image + n*t1 + t2));
                        }else{
                            ans += *(mask + k*m + l);
                        }
                    }
                }
            }
            *(output + i * n + j) = ans;
        }
    }
}

int main(int argc, char **argv){
    if(argc != 3){
        exit(-1);
    }

    std::size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);
    size_t SIZE = n*n;

    float *image = new float[SIZE];
    float *mask = new float[9];
    float *output = new float[SIZE];
//    float *gt = new float[SIZE];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist1(-10.0, 10.0);
    std::uniform_real_distribution<> dist2(-1.0, 1.0);

    for(std::size_t i = 0; i < n; i++){
        for(std::size_t j = 0; j < n; j++){
            *(image + i*n + j) = dist1(gen);
            *(output + i*n + j) = 0.0;
//            *(gt + i*n + j) = 0.0;
        }
    }

    for(std::size_t i = 0; i < 3; i++){
        for(std::size_t j = 0; j < 3; j++){
            *(mask + i*3 + j) = dist2(gen);
        }
    }

    omp_set_num_threads(t);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    convolve(image, output, n, mask, 3);
    end = high_resolution_clock::now();
    /*
    convolve_verif(image, gt, n, mask, 3);

    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < n; j++){
            if((*(gt+i*n+j)-*(output+i*n+j)) > 1e-5){
                std::cout << "diff at " << i << " , " << j << " is = " << *(gt+i*n+j) - *(output + i*n+j) << std::endl; 
            }
        }
    }
    */
    
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << *output << std::endl << *(output + SIZE - 1) << std::endl << duration_msec.count() << std::endl;

    return 0;
}
