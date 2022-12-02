#include <cstddef>
#include <omp.h>

#include "reduce.h"

// this function should do a parallel reduction with OpenMP to get
// the summation of elements in array "arr" in the range [l, r)
// do as much as you can to improve performance, 
// i.e. use simd directive

float reduce(const float* arr, const size_t l, const size_t r)
{
    float reduced_sum = 0.0;
#pragma omp parallel
    {
#pragma omp for simd reduction(+:reduced_sum)
        for(size_t i = l; i < r; i++){
            reduced_sum += arr[i];
        }
    }
    return reduced_sum;
}
