#ifndef QR_CUH
#define QR_CUH

#include <thrust/device_vector.h>
//performs qr factorization of a using household reflections
//Q is in q and R is places in r
//a is a square matrix of dimension n*n
//q*r = a

void qr(const thrust::device_vector<float>& a, thrust::device_vector<float>& q, thrust::device_vector<float>& r, unsigned int n){

#endif
