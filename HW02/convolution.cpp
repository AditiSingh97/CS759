#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    std::size_t mm = (m-1)/2;
    for(std::size_t i = 0; i < n; i++){
        for(std::size_t j = 0; j < n ; j++){
            float ans = 0.0;
            for(std::size_t k = 0; k < m; k++){
                for(std::size_t l = 0; l < m; l++){
                    std::size_t t1 = i + k - mm;
                    std::size_t t2 = j + l -mm;
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
            *(output + i * n + j) = ans;
        }
    }
}