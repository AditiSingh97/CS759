#include <cuda.h>
#include <cstdio>
#include <cmath>

#include "matmul.cuh"

__global__ void matmul(const float *A, const float *B, float *C, unsigned int n){
    // Shared memory for the sub-matrices (tiles) of  A and B
    extern __shared__ float sdata[];
    float *As = sdata;
    float *Bs = sdata + blockDim.x * blockDim.x;
    
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
   float Csub = 0;
   // Loop over all the sub-matrices (tiles) of A and B required to
   // compute the block sub-matrix; moving in A left to right in
   // a row, and in B from top to bottom in a column
   for (int a = aBegin, b = bBegin;
      a <= aEnd;
      a += aStep, b += bStep) {

         // Load tiles from global memory into shared memory; each
         // thread loads one element of the two tiles from A & B
         As[ty * blockDim.x + tx] = A[a + n * ty + tx];
         Bs[ty * blockDim.x + tx] = B[b + n * ty + tx];

         // Synchronize to make sure the matrices are loaded
         __syncthreads();

         // Each thread in this block computes one element 
         // of the block sub-matrix (tile).  Thread with indexes
         // ty and tx computes in this tile the entry [ty][tx].  
        for (int k = 0; k < blockDim.x; ++k)
	{
		Csub += As[ty * blockDim.x + k] * Bs[k * blockDim.x + tx];
	}

         // Synchronize to make sure that the preceding
         // computation is done before loading two new
         // sub-matrices of A and B in the next iteration
         __syncthreads();
   }
   // Write the block sub-matrix to global memory;
   // each thread writes one element
   int c = n * blockDim.x * by + blockDim.x * bx;
   C[c + n * ty + tx] = Csub;
   printf("C[%u]:%f\n",c + n * ty + tx, C[c + n * ty + tx]);
}


__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
	    // Compute the execution configuration *assuming*
	    // the matrix dimensions are multiples of BLOCK_SIZE
	    dim3 dimBlock(block_dim, block_dim);
	    unsigned q = ceil((double)n/(double)dimBlock.x);
	    std::printf("grid dimension: %u\n", q);
	    dim3 dimGrid(q, q);
	    // Launch the device computation
	    matmul<<<dimGrid, dimBlock, 2*block_dim*block_dim*sizeof(float) >>>(A, B, C, n);
	    cudaDeviceSynchronize();
}
