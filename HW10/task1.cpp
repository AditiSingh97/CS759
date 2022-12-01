#include <iostream>
#include <chrono>
#include <random>

#include "optimize.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;


int main(int argc, char **argv){
    if(argc != 2){
        exit(-1);
    }

    //parsing command line args
    int n = atoi(argv[1]);

    //vec declarations
    vec v(n);
    v.data = new data_t[n];

    //generating random floats
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for(int i = 0; i < n; i++){
        if(dist(gen) <= 0.5)
            v.data[i] = (data_t)(-1.0);
        else
            v.data[i] = (data_t)(1.0);
    } 

    data_t *dest = new data_t;
    // start timing t0
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    *dest = 0;
    start = high_resolution_clock::now();
    optimize1(&v, dest);
    end = high_resolution_clock::now();
    //end timing
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << *dest << std::endl << duration_msec.count() << std::endl;

    *dest = 0;
    start = high_resolution_clock::now();
    optimize2(&v, dest);
    end = high_resolution_clock::now();
    //end timing
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << *dest << std::endl << duration_msec.count() << std::endl;

    *dest = 0;
    start = high_resolution_clock::now();
    optimize3(&v, dest);
    end = high_resolution_clock::now();
    //end timing t0
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << *dest << std::endl << duration_msec.count() << std::endl;

    *dest = 0;
    start = high_resolution_clock::now();
    optimize4(&v, dest);
    end = high_resolution_clock::now();
    //end timing t0
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << *dest << std::endl << duration_msec.count() << std::endl;

    *dest = 0;
    start = high_resolution_clock::now();
    optimize5(&v, dest);
    end = high_resolution_clock::now();
    //end timing t0
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << *dest << std::endl << duration_msec.count() << std::endl;

    delete [] v.data;
    delete dest;

    return 0;
}
