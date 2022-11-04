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
    //temporary array to copy merged results
    int *tmp = new int[n];
    memset(tmp, 0, n*sizeof(int));
    size_t p1 = 0;
    size_t p2 = n/2;
    size_t curr_tmp = 0;

    //compare and cppy elements from arr to tmp, increment appropriate pointers
    while ((p1 < n/2) && (p2 < n)) {
        if (arr[p1] < arr[p2]) {
            tmp[curr_tmp] = arr[p1];
            curr_tmp++;
            p1++;
        }
        else {
            tmp[curr_tmp] = arr[p2];
            curr_tmp++;
            p2++;
        }
    }
    while (p1 < (n/2)) { /* finish up lower half */
        tmp[curr_tmp] = arr[p1];
        curr_tmp++;
        p1++;
    }
    
    while (p2 < n) { /* finish up upper half */
        tmp[curr_tmp] = arr[p2];
        curr_tmp++;
        p2++;
    }
    
    //copy results back to arr
    memcpy(arr, tmp, n*sizeof(int));
} 

void msort(int* arr, const std::size_t n, const std::size_t threshold){
    //if n < threshold, perform tasks serially
    //if n>= threshold, perform tasks parallely
    if (n < 2)
    {
        return;
    }
    if(n < threshold){
        int i, j;
        int key = 0;
        for (i = 0; i < n; i++)
        {
            key = *(arr + i);
            j = i - 1;

            // Move elements of arr[0..i-1], 
            // that are greater than key, to one
            // position ahead of their
            // current position
            while (j >= 0 && *(arr + j) > key)
            {
                *(arr + j + 1) = *(arr + j);
                j = j - 1;
            }
            *(arr + j + 1) = key;
        }
    }
    else{
        //sort first half

#pragma omp task
        msort(arr, n/2, threshold);
        //sort second half
#pragma omp task
        msort(arr+(n/2), n-(n/2), threshold);
    }
        //wait for both sorts to finish, then merge
#pragma omp taskwait
        merge(arr, n);

}
