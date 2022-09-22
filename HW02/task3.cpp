#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#include "matmul.h"

using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv){
    unsigned int n = 1000;
    double *A = new double[n*n];
    double *B = new double[n*n];
    double *C = new double[n*n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < n; j++){
            *(A + i*n + j) = dist(gen);
            *(B + i*n + j) = dist(gen);
        }
    }

    std::cout << n << std::endl;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    mmul1(A, B, C, n);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_msec.count() << std::endl << *(C + (n*n) - 1) << std::endl;

    start = high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_msec.count() << std::endl << *(C + (n*n) - 1) << std::endl;

    start = high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_msec.count() << std::endl << *(C + (n*n) - 1) << std::endl;

    vector<double> a;
    vector<double> b;

    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < n; j++){
            a.push_back(*(A+ i*n + j));
            b.push_back(*(B + i*n + j));
        }
    }
    start = high_resolution_clock::now();
    mmul4(a, b, C, n);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_msec.count() << std::endl << *(C + (n*n) - 1) << std::endl;

}
