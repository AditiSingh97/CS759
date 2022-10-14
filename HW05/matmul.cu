#include <cuda.h>
#include <cstdio>
#include <cmath>

#include "matmul.cuh"


//code taken from CS 759 lecture 12
template <typename T>
__global__ void matmul(const T *A, const T *B, T *C, unsigned int n){
    // Shared memory for the sub-matrices (tiles) of  A and B
    extern __shared__ char smem[];
    T * sdata = reinterpret_cast<T *>(smem);
    T *As = sdata;
    T *Bs = sdata + blockDim.x * blockDim.x;
    
    //could copy elements of A, followed by elements of B

   // Block index
   int bx = blockIdx.x; //the B (and C) matrix sub-block column index
   int by = blockIdx.y; //the A (and C) matrix sub-block row index

   // Thread index
   int tx = threadIdx.x; //the column index in the sub-block
   int ty = threadIdx.y; //the row index in the sub-block

   // Index of the first sub-matrix of A processed by the block
   int aBegin = n * blockDim.x * by;

   // Index of the last sub-matrix of A processed by the block
   int aEnd = aBegin + n - 1;

   // Step size used to iterate through the sub-matrices of A
   int aStep = blockDim.x;

   // Index of the first sub-matrix of B processed by the block
   int bBegin = blockDim.x * bx;

   // Step size used to iterate through the sub-matrices of B
   int bStep = blockDim.x * n;

   // The element of the block sub-matrix that is computed
   // by the thread
   // Loop over all the sub-matrices (tiles) of A and B required to
   // compute the block sub-matrix; moving in A left to right in
   // a row, and in B from top to bottom in a column
   T Csub = 0;
   for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
	   // Load tiles from global memory into shared memory; each
	   // thread loads one element of the two tiles from A & B
	   As[ty * blockDim.x + tx] = (T)0.0;
	   Bs[ty * blockDim.x + tx] = (T)0.0;
	   if((ty < n) && (tx < n)){
		   As[ty * blockDim.x + tx] = A[a + n * ty + tx];
	   }else{
		   As[ty * blockDim.x + tx] = (T)0.0;
	   }

	   if((ty < n) && (tx < n)){
		   Bs[ty * blockDim.x + tx] = B[b + n * ty + tx];
	   }
	   else{
		   Bs[ty * blockDim.x + tx] = (T) 0.0;
	   }

           // Synchronize to make sure the matrices are loaded
           __syncthreads();

           // Each thread in this block computes one element 
           // of the block sub-matrix (tile).  Thread with indexes
           // ty and tx computes in this tile the entry [ty][tx].  
          for (int k = 0; k < blockDim.x; ++k){
		  Csub += As[ty * blockDim.x + k] * Bs[k * blockDim.x + tx];
	  }
	  // Synchronize to make sure that the preceding
          // computation is done before loading two new
          // sub-matrices of A and B in the next iteration
	  __syncthreads();
   }
   
   // Write the block sub-matrix to global memory;
   // each thread writes one element
   
   unsigned int Row = blockIdx.y*blockDim.x + threadIdx.y;
   unsigned int Col = blockIdx.x*blockDim.y + threadIdx.x;
   int c = n * blockDim.y * by + blockDim.x * bx;
   if((Row < n) && (Col < n))
   {
	   C[c + n * ty + tx] = Csub;
   }
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
	    // Compute the execution configuration *assuming*
	    // the matrix dimensions are multiples of BLOCK_SIZE
	    dim3 dimBlock(block_dim, block_dim);
	    unsigned q = ceil((double)n/(double)dimBlock.x);
	    dim3 dimGrid(q, q);
	    // Launch the device computation
	    matmul<int><<<dimGrid, dimBlock, 2*block_dim*block_dim*sizeof(int)>>>(A, B, C, n);
	    cudaDeviceSynchronize();
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
	    // Compute the execution configuration *assuming*
	    // the matrix dimensions are multiples of BLOCK_SIZE
	    dim3 dimBlock(block_dim, block_dim);
	    unsigned q = ceil((double)n/(double)dimBlock.x);
	    dim3 dimGrid(q, q);
	    // Launch the device computation
	    matmul<float><<<dimGrid, dimBlock, 2*block_dim*block_dim*sizeof(float)>>>(A, B, C, n);
	    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
	    // Compute the execution configuration *assuming*
	    // the matrix dimensions are multiples of BLOCK_SIZE
	    dim3 dimBlock(block_dim, block_dim);
	    unsigned q = ceil((double)n/(double)dimBlock.x);
	    dim3 dimGrid(q, q);
	    // Launch the device computation
	    matmul<double><<<dimGrid, dimBlock, 2*block_dim*block_dim*sizeof(double)>>>(A, B, C, n);
	    cudaDeviceSynchronize();
}
