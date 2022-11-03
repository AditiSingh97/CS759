#include <cstddef>
#include <omp.h>

// This function produces a parallel version of matrix multiplication C = A B using OpenMP. 
// The resulting C matrix should be stored in row-major representation. 
// Use mmul2 from HW02 task3. You may recycle the code from HW02.

// The matrices A, B, and C have dimension n by n and are represented as 1D arrays.

void mmul(const float* A, const float* B, float* C, const std::size_t n){
    size_t i = 0, j = 0, k = 0;
//all code below pragma directive taken directly from HW02
////i = row, j = column, k = running variable
#pragma omp parallel for private(i,j,k) shared(A,B,C)
    for(size_t i = 0; i < n; i++){
        for(size_t k = 0; k < n; k++){
            for(size_t j = 0; j < n; j++){
                //element computation
                *(C + i*n + j) += *(A + i*n + k) * (*(B + k*n + j));
            }
        }
    }
}

