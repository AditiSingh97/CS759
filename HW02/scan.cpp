#include "scan.h"

void scan(const float *arr, float *output, std::size_t n){
    float sum = 0;
    for(unsigned int i = 0; i < n; i++){
        sum = sum + *(arr + i);
        *(output + i) = sum;
    }
}