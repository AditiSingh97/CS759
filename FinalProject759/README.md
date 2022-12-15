# dynamic-embeddings

### Euler Machine Instructions
1. To run **[qr_cublas.py](https://git.doit.wisc.edu/SINGH273/repo759/-/blob/245e508431cd7ffacdd14445fe276f5012856c0d/FinalProject759/qr_cublas.py)**, **[qr_cupy.py](https://git.doit.wisc.edu/SINGH273/repo759/-/blob/245e508431cd7ffacdd14445fe276f5012856c0d/FinalProject759/qr_cupy.py)**, and **[qr_np.py](https://git.doit.wisc.edu/SINGH273/repo759/-/blob/245e508431cd7ffacdd14445fe276f5012856c0d/FinalProject759/qr_np.py)** machine should have python 3.10 and compatible Numpy and CuPy installed. For Euler's GPU and driver, CuPy eeds to be installed fro CUDA 11.4
2. All three files qr_cublas.py, qr_cupy.py and qr_np.py can be called as `python <file> <n: size of square matrix>`. A matrix of n*n elements will be used initialized with random floats and factorized using QR factorization.
3. The files **[qr.cuh](https://git.doit.wisc.edu/SINGH273/repo759/-/blob/245e508431cd7ffacdd14445fe276f5012856c0d/FinalProject759/qr.cuh)** and **[qr.cu](https://git.doit.wisc.edu/SINGH273/repo759/-/blob/245e508431cd7ffacdd14445fe276f5012856c0d/FinalProject759/qr.cu)** contain some code for Thrust implementation of QR-factorization but the code is not complete yet. To build it, we need to load modules in euler as follows `module load nvidia/cuda gcc/9.4.0`
4. In **[dyn_embed.py](https://git.doit.wisc.edu/SINGH273/repo759/-/blob/245e508431cd7ffacdd14445fe276f5012856c0d/FinalProject759/dyn_embed.py)**, the function `GenSvdUpdate()` implements end-to-end incremental SVD using QR-factorization at some steps. This is however currently semantically buggy towards the end.

### Repository Internals
The repository contains the following implementations. The code in the github repository is linked for each.
The links currently lead to an external GitHub repository which is the origin of this code and safe to visit.

1. Generate [static HOPE embeddings](https://github.com/marius-team/dynamic-embeddings/blob/0c1b89e3b0dc4b45ac208af808a050bb864db74a/dyn_embed.py#L569) for a graph supplied as a bin file (specific format : __(source node id)__ __(#edge originating from preceding node)__ __(the destinations of these edges…)__). Embeddings can be generated using three proximity metrics - [Katz](https://github.com/marius-team/dynamic-embeddings/blob/0c1b89e3b0dc4b45ac208af808a050bb864db74a/dyn_embed.py#L379), [Common Neighbors](https://github.com/marius-team/dynamic-embeddings/blob/0c1b89e3b0dc4b45ac208af808a050bb864db74a/dyn_embed.py#L455), and [Adamic Adar](https://github.com/marius-team/dynamic-embeddings/blob/0c1b89e3b0dc4b45ac208af808a050bb864db74a/dyn_embed.py#L486). 
  
    i. Further work: With Katz, there are two possible implementations - Full HOPE and [Partial HOPE](https://github.com/marius-team/dynamic-embeddings/blob/0c1b89e3b0dc4b45ac208af808a050bb864db74a/dyn_embed.py#L391). Partial HOPE factorizes the main proximity matrix into two relatively sparser matrices to speed up SVD. Partial SVD uses a special type of SVD called JDGSVD but is not working correctly as of now.
2. Evaluate HOPE embeddings given a train and test dataset of edges in the form of a bin file. (specific format : __(source node id)__ __(destination node id)__ __(line break)__).
3. Generate [singular value decomposition](https://github.com/marius-team/dynamic-embeddings/blob/0c1b89e3b0dc4b45ac208af808a050bb864db74a/dyn_embed.py#L104) (SVD) of a matrix dynamically. When the initial matrix’s SVD components are available, and the modifications made to it are only additive and only in the form of low-rank matrices, the implementation quickly returns the SVD components of the new modified matrix. The additive matrix must be factorizable as two matrices having rows equalling the size of the initial matrix.
_This method works correctly when the additive matrices have only a single column i.e., they are vectors. There are still some implementation issues with larger matrices to be further addressed._
4. Generate dynamic HOPE embeddings when using Katz and Common Neighbors metrics. The implementation can relay changes pertaining to a single source node (i.e., edges added and/or deleted to a single node and originating from it.) in a single run.

    i. Further work : With Katz, only the Full version can be used with Dynamic HOPE. We do not know of a methodology to perform JGDSVD incrementally. 
5. Given a graph and its smaller subgraph, run a series of dynamic HOPE embeddings on the small graph, continuously growing it through edge and vertex additions to become the larger graph. The embeddings can be optionally saved after each round.
A tool to generate random subgraphs of a given graph. The input is a simple list of edges and the output is a bin format of the new graph (in the aforementioned format). It also adds and deleted edges and vertices to an input graph.
6. Some utilities to convert .bin to .txt and vice versa, convert .csv to .bin


