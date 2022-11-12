#include <iostream>
#include "mpi.h"
#include <chrono>
#include <random>

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace std;

int main(int argc, char **argv){
    if(argc != 2){
        exit(-1);
    }

    int my_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //parsing command line args
    int n = atoi(argv[1]);

    //array declarations
    float *a = new float[n];
    float *b = new float[n];

    //generating random floats
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for(int i = 0; i < n; i++){
        a[i] = dist(gen);
        b[i] = dist(gen);
    }
    
    duration<double, std::milli> duration_msec_0, duration_msec_1;
    double t1;
    MPI_Status status; 
    if (my_rank == 0) {
        // start timing t0
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;

        start = high_resolution_clock::now();
        //MPI send and recv
        MPI_Send(a, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD) ;
        MPI_Recv(b, n, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status) ;
        end = high_resolution_clock::now();
        //end timing t0
        duration_msec_0 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

        MPI_Recv(&t1, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);
        std::cout << duration_msec_0.count() + t1 << std::endl;

    }else if(my_rank == 1){
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;

        start = high_resolution_clock::now();
        //MPI send and recv
        MPI_Recv(b, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status) ;
        MPI_Send(a, n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD) ;
        end = high_resolution_clock::now();
        //end timing t0
        duration_msec_1 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        t1 = duration_msec_1.count();
        MPI_Send(&t1, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
