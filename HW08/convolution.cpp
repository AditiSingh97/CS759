#include <cstddef>
#include <omp.h>

// This function does a parallel version of the convolution process in HW02 task2
// using OpenMP. You may recycle your code from HW02.

// "image" is an n by n grid stored in row-major order.
// "mask" is an m by m grid stored in row-major order.
// "output" stores the result as an n by n grid in row-major order.

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    std::size_t mm = (m-1)/2;
//the following pragma directive is sufficient to parallelize the outermost for loop
//code directly takn from HW02 code for convolution
#pragma omp parallel for
    for(std::size_t i = 0; i < n; i++){
        for(std::size_t j = 0; j < n ; j++){
            //intermeditae result ans
            float ans = 0.0;
            //starting loops for mask
            for(std::size_t k = 0; k < m; k++){
                for(std::size_t l = 0; l < m; l++){
                    //calculating limits
                    std::size_t t1 = i + k - mm;
                    std::size_t t2 = j + l -mm;
                    //accounting for corner cases
                    if(((t1 < 0) || (t1 >= n)) && ((t2 < 0) || (t2 >= n))){
                        ans += 0.0;
                    }else{
                        if(((t1 >= 0) && (t1 < n)) && ((t2 >= 0) && (t2 < n))){
                            ans += *(mask + k*m + l) * (*(image + n*t1 + t2));
                        }else{
                            ans += *(mask + k*m + l);
                        }
                    }
                }
            }
            //copying intermediate result to output
            *(output + i * n + j) = ans;
        }
    }
}
