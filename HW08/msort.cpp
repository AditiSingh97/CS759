#include <cstddef>
#include <omp.h>
#include <cstring>
#include <iostream>

// This function does a merge sort on the input array "arr" of length n. 
// You can add more functions as needed to complete the merge sort,
// but do not change this file. Declare and define your addtional
// functions in the msort.cpp file, but the calls to your addtional functions
// should be wrapped in the "msort" function.

// "threshold" is the lower limit of array size where your function would 
// start making parallel recursive calls. If the size of array goes below
// the threshold, a serial sort algorithm will be used to avoid overhead
// of task scheduling

using namespace std;

void merge(int *arr, size_t n) {
    int *tmp = new int[n];
    memset(tmp, 0, n*sizeof(int));
    size_t i = 0;
    size_t j = n/2;
    size_t ti = 0;

    while ((i < n/2) && (j < n)) {
        if (arr[i] < arr[j]) {
            tmp[ti] = arr[i];
            ti++;
            i++;
        }
        else {
            tmp[ti] = arr[j];
            ti++;
            j++;
        }
    }
    while (i < (n/2)) { /* finish up lower half */
        tmp[ti] = arr[i];
        ti++;
        i++;
    }
    
    while (j < n) { /* finish up upper half */
        tmp[ti] = arr[j];
        ti++;
        j++;
    }
    /*
     * for(size_t k = 0; k < n; k++){
        std::cout << "tmp[" << k << "]=" << tmp[k] ;
    }
    std::cout << std::endl;
    */

    memcpy(arr, tmp, n*sizeof(int));
} 

void msort(int* arr, const std::size_t n, const std::size_t threshold){
    //if n < threshold, perform tasks serially
    //if n>= threshold, perform tasks parallely
    if (n < 2) return;
    
#pragma omp task shared(arr) if (n >= threshold)
    //msort(arr, n/2, tmp);
    msort(arr, n/2, threshold);
#pragma omp task shared(arr) if (n >= threshold)
   //msort(arr+(n/2), n-(n/2), tmp + n/2);
   msort(arr+(n/2), n-(n/2), threshold);
#pragma omp taskwait
   merge(arr, n);

}
