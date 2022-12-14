#include "qr.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

void qr(const thrust::device_vector<thrust::device_vector<float>>& a, thrust::device_vector<thrust::device_vector<float>>& q, thrust::device_vector<thrust::device_vector<float>>& r, unsigned int n){
    r = a
    for(unsigned int i = 0; i < n-1; i++){
        thrust::device_vector<thrust::device_vector<float>> x(n);
        for(unsigned int j = 0; j < n; j++){
            x[j] = thrust::device_vector<float>(1);
            if(j >= i)
            {
                x[j][0] = r[j][i];
            }else{
                x[j][0] = 0.0f;
            }
        }
        float norm = thrust::inner_product(thrust::device, x.begin(), x.begin() + x.size(), x.begin(), 0.0f);
        thrust::device_vector<thrust::device_vector<float>> e(n);
        for(unsigned int j = 0; j < n; j++){
            e[j] = thrust::device_vector<float>(1);
            e[j][0] = 0.0f;
        }
        e[i][0] = norm * (-1.0);

    }
}
