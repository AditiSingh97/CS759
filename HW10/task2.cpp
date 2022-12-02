#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cstddef>
#include <omp.h>
#include <chrono>

#include "mpi.h"
#include "reduce.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;

int main(int argc, char **argv){
    if(argc != 3){
        exit(-1);
    }

    int my_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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

    //ensuring both processes start at the same time from this point onwards
    MPI_Barrier(MPI_COMM_WORLD);
    //setting number of threads
    omp_set_num_threads(t);
    //initializing timer objects
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    start = high_resolution_clock::now();
    //function call
    float res = reduce(arr, 0, n);

    float global_res;
    MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    end = high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    if(my_rank == 0)
        std::cout << global_res << std::endl << duration_msec.count() << std::endl; 

    delete [] arr;

    MPI_Finalize();

    return 0;
}
