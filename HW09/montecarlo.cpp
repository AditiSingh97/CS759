#include <cstddef>
#include <omp.h>

// this function returns the number of points that lay inside
// a circle using OpenMP parallel for. 
// You also need to use the simd directive.

// x - an array of random floats in the range [-radius, radius] with length n.
// y - another array of random floats in the range [-radius, radius] with length n.

int montecarlo(const size_t n, const float *x, const float *y, const float radius){
      int incircle_count = 0;
#pragma omp parallel
    {
#pragma omp for simd reduction(+:incircle_count)
        for (size_t i = 0; i < n; i++){
             float x_tmp = x[i];
             float y_tmp = y[i];

             float distance_from_center = x_tmp * x_tmp + y_tmp * y_tmp;
             if(distance_from_center <= radius*radius){
                 incircle_count+= 1;
             }
         }
    }
    return incircle_count;
}
