import math
import numpy as np
import logging
import pygsvd
import struct
import sys
import time
import os
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import inv
from sortedcontainers import SortedSet

mapping_f = {}
mapping_b = {}
num_vertices = 0
count = 0
np.set_printoptions(threshold=20000, precision=10)

def svdUpdate(U, S, V, a, b, rank):
    """
    Update SVD of an (m x n) matrix `X = U * S * V^T` so that
    `[X + a * b^T] = U' * S' * V'^T`
    and return `U'`, `S'`, `V'`.
    
    The original matrix X is not needed at all, so this function implements one-pass
    streaming rank-1 updates to an existing decomposition. 
    
    `a` and `b` are (m, 1) and (n, 1) matrices.
    
    You can set V to None if you're not interested in the right singular
    vectors. In that case, the returned V' will also be None (saves memory).
    
    The blocked merge algorithm in LsiModel.addDocuments() is much faster; I keep this fnc here
    purely for backup reasons.
    This is the rank-1 update as described in
    **Brand, 2006: Fast low-rank modifications of the thin singular value decomposition**,
    but without separating the basis from rotations.
    """
    # convert input to matrices (no copies of data made if already numpy.ndarray or numpy.matrix)
    S = S[:rank]
    
    U = U[:,:rank]
    if V is not None:
        V = V[:,:rank]

    # eq (6)
    m = np.dot(U.T, a)
    p = a - np.dot(U, m)
    Ra = np.linalg.norm(p, 2)
    if float(Ra) < 1e-10:
        print("input already contained in a subspace of U; skipping update")
        return U, S, V
    P = (1.0 / float(Ra)) * p

    if V is not None:
        # eq (7)
        n = np.dot(V.T, b)
        q = b - np.dot(V, n)
        Rb = np.linalg.norm(q, 2)
        if float(Rb) < 1e-10:
            print("input already contained in a subspace of V; skipping update")
            return U, S, V
        Q = (1.0 / float(Rb)) * q
    else:
        n = np.zeros(rank, 1)
        Rb = 1.0
    
    if float(Ra) > 1.0 or float(Rb) > 1.0:
        print("insufficient target rank (Ra=%.3f, Rb=%.3f); this update will result in major loss of information"
                      % (float(Ra), float(Rb)))
    
    # eq (8)
    # form the new matrix
    K_1 = np.diag(np.append(S, 0.0))
    Raa = np.reshape(Ra, (1,1))
    print(np.shape(Raa))
    print(np.shape(m))
    #K_2_1 = np.append(m, Raa)
    K_2_1 = np.bmat([[m],[Raa]])
    Rbb = np.reshape(Ra, (1,1))
    print(np.shape(Rbb))
    #K_2_2 = np.append(n, Rbb)
    K_2_2 = np.bmat([[n],[Rbb]])
    K_2 = np.outer(K_2_1, K_2_2)
    
    K = K_1 + K_2
    
    # eq (5)
    uP, sP, vtP = np.linalg.svd(K, full_matrices = True)
    p_l = len(P)
    q_l = len(Q)
    
    Up = np.dot(np.bmat([U, np.reshape(P, (p_l, 1))]), uP)
    if V is not None:
        Vp = np.dot(np.bmat([V, np.reshape(Q, (q_l, 1))]), vtP.T)
    else:
        Vp = None

    print("IncrSVD completed")
    return Up[:,0:rank], sP[0:rank], Vp[:,0:rank]

def GenSvdUpdate(U, S, V, a, b, rank):
    """
    Update SVD of an (m x n) matrix `X = U * S * V^T` so that
    `[X + a * b^T] = U' * S' * V'^T`
    and return `U'`, `S'`, `V'`.
    
    The original matrix X is not needed at all, so this function implements one-pass
    streaming rank-c updates to an existing decomposition. 
    
    `a` and `b` are (m, 1) and (n, 1) matrices.
    """

    # convert input to matrices (no copies of data made if already numpy.ndarray or numpy.matrix)
    S = S[:rank]
    
    U = U[:,:rank]
    if V is not None:
        V = V[:,:rank]

    """
    print("Inside low-rank incr-SVD")
    print("U shape : ", U.shape)
    print("S shape : ", S.shape)
    print("V shape : ", V.shape)
    print("a shape : ", a.shape)
    print("b shape : ", b.shape)
    """

    I = np.eye(len(U))

    QR1 = np.matmul(I-(np.matmul(U,U.T)),a)
    print("QR1 :", QR1.shape)
    P,R = np.linalg.qr(QR1)
    print("P :", P.shape)
    print("R :", R.shape)
    Ra = np.matmul(P.T, QR1)
    print("Ra :", Ra.shape)


    QR2 = np.matmul(I-(np.matmul(V,V.T)),b)
    Q,_ = np.linalg.qr(QR2)
    Rb = np.matmul(Q.T, QR2)
    print("Rb :", Rb.shape)

    # eq (8)
    # form the new matrix
    K_1 = np.diag(np.append(S, 0.0))

    K_2_1_1 = np.matmul(U.T,a)
    print("K_2_1_1 : ", K_2_1_1.shape)
    K_2_1 = np.bmat([[K_2_1_1], [Ra]])
    print("K_2_1 : ", K_2_1.shape)

    K_2_2_1 = np.matmul(b.T,V)
    print("K_2_2_1 : ", K_2_2_1.shape)

    K_2_2 = np.bmat([K_2_2_1, Rb.T])
    print("K_2_2 : ", K_2_2.shape)

    K_2 = np.matmul(K_2_1, K_2_2)
    print("K_2 : ", K_2.shape)
    
    K = K_1 + K_2
    
    # eq (5)
    s = time.time()
    uP, sP, vtP = np.linalg.svd(K, full_matrices = True)
    print("Incr-SVD time :" , time.time()-s)
    p_l = len(P)
    q_l = len(Q)
    
    Up = np.dot(np.bmat([U, P]), uP)
    if V is not None:
        Vp = np.dot(np.bmat([V, Q]), vtP.T)
    else:
        Vp = None

    """
    print("Up shape : ", Up.shape)
    print("sP shape : ", sP.shape)
    print("Vp shape : ", Vp.shape)
    print("IncrSVD completed")
    """
    
    return Up[:,0:rank], sP[0:rank], Vp[:,0:rank]

def map_node_ids(old_ids):
    global mapping_f
    global mapping_b
    i = int(0)
    for node in old_ids:
        mapping_f[node]=i
        mapping_b[i]=node
        i += 1

def load_mutation_data(filename):
    raw_data = np.fromfile(filename, dtype=np.uint64, count = -1)
    length = int(len(raw_data)/2)
    A = raw_data[0:length]
    B = raw_data[length:len(raw_data)]
    return A, B

#add comments on functionality
def load_graph(filename):
    global num_vertices
    raw_data = np.fromfile(filename, dtype=np.uint64, count = -1)
    
    total_len = len(raw_data)

    graph_data = {}
    old_ids = SortedSet()

    i = int(0);
    while i < total_len:
        begin = int(i+2)
        added = int(raw_data[i+1])
        end = int(i) + int(2) + added

        graph_data[int(raw_data[i])] = raw_data[begin:end]
        old_ids.add(int(raw_data[i]))
        for j in graph_data[int(raw_data[i])]:
            old_ids.add(j)
        
        i = int(end)
        if(i >= total_len):
            break;

    map_node_ids(old_ids)
    num_vertices = len(old_ids)

    adj_matrix = np.zeros((num_vertices,num_vertices) , dtype=np.float64)
    #use csr/csc format - scipy svd
    for vertex in graph_data.keys():
        for nghbr in graph_data[vertex]:
            adj_matrix[mapping_f[vertex]][mapping_f[nghbr]]=1;

    return adj_matrix

def load_graph_with_mapping(filename):
    global num_vertices
    global mapping_b
    global mapping_f

    raw_data = np.fromfile(filename, dtype=np.uint64, count = -1)
    total_len = len(raw_data)

    graph_data = {}
    i = int(0);
    while i < total_len:
        begin = int(i+2)
        added = int(raw_data[i+1])
        end = int(i) + int(2) + added

        graph_data[int(raw_data[i])] = raw_data[begin:end]
        i = int(end)
        if(i >= total_len):
            break;

    adj_matrix = np.zeros((num_vertices,num_vertices) , dtype=np.float64)
    #use csr/csc format - scipy svd
    for vertex in graph_data.keys():
        for nghbr in graph_data[vertex]:
            adj_matrix[mapping_f[vertex]][mapping_f[nghbr]]=1;

    return adj_matrix

#Load the graph. Map the vertices in adj_matrix using already available mapping(created while loading a superset graph). Create dict of in-nghbrs called src_dict
def load_graph_with_mapping_in_nghbrs(filename):
    global num_vertices
    global mapping_b
    global mapping_f

    raw_data = np.fromfile(filename, dtype=np.uint64, count = -1)
    total_len = len(raw_data)

    graph_data = {}
    src_dict = {}
    i = int(0);
    while i < total_len:
        begin = int(i+2)
        added = int(raw_data[i+1])
        end = int(i) + int(2) + added

        graph_data[int(raw_data[i])] = raw_data[begin:end]
        i = int(end)
        if(i >= total_len):
            break;

    adj_matrix = np.zeros((num_vertices,num_vertices) , dtype=np.float64)
    #use csr/csc format - scipy svd
    for vertex in graph_data.keys():
        for nghbr in graph_data[vertex]:
            if nghbr not in src_dict.keys():
                src_dict[nghbr] = []
            src_dict[nghbr].append(vertex)
            adj_matrix[mapping_f[vertex]][mapping_f[nghbr]]=1;

    return adj_matrix, src_dict

def calculate_vectors_alt(U,S,VT,k):
    """
    source = np.empty((len(U),k),dtype=np.float64)
    target = np.empty((len(U),k),dtype=np.float64)
    """
    V = VT.T
    source = np.dot(U, np.diag(np.sqrt(S)))
    target = np.dot(V, np.diag(np.sqrt(S)))
    return source,target

def calculate_vectors(V1,S1,V2,S2,k):
    print("Received SVDs of proximity matrices, computing HOPE vectors now \n")
    source = np.empty((len(V1[0]),k),dtype=np.float64)
    target = np.empty((len(V2[0]),k),dtype=np.float64)
    
    v1 = V1.T
    v2 = V2.T
    print("Computed final v1 and v2 \n")

    sigmas = []
    for i in range(0,k):
        sigmas.append(np.float64(math.sqrt(S2[i]/S1[i])))
    
    source = np.dot(v1, np.diag(sigmas))
    target = np.dot(v2, np.diag(sigmas))
    return source,target

def ret_prox_param(X):
    eigvalue,eigvecs = scipy.sparse.linalg.eigs(X,k=1,which='LM')
    return abs(eigvalue)

def return_prox_mat_katz_full(X, prox_param):
    n = len(X)
    id_m = np.eye(n)

    Mg = csr_matrix(id_m - (prox_param*X))
    Ml = csr_matrix(prox_param*X)

    return Mg,Ml

def return_prox_mat_common(X):
    n = len(X)
    id_m = np.eye(n)

    Ml = csr_matrix(np.matmul(X,X))

    return Ml

def return_prox_mat_aa(X):
    n = len(X)
    id_m = np.eye(n)

    Mg = csr_matrix(id_m)
    D = np.zeros((n,n), dtype=np.float64)

    for i in range(0,n):
        ele = np.float64(0.0)
        for j in range(0,n):
            ele = ele + X[i][j] + X[j][i]
        D[i][i] = 1/ele

    D_csr = csr_matrix(D)
    X_csr = csr_matrix(X)
    Ml = X_csr@(D_csr@X_csr)

    return Mg,Ml

def return_prox_mat_katz_partial(X, prox_param):
    n = len(X)
    id_m = np.eye(n)

    Mg = id_m - (prox_param*X)
    Ml = prox_param*X

    return Mg,Ml

def katz_full(X, dim, prox_param):
    Mg,Ml = return_prox_mat_katz_full(X, prox_param)
    Mg_inv =  inv(Mg)
    """Mg_inv_sparse = csr_matrix(Mg_inv)
    Ml_sparse = csr_matrix(Ml)
    S = Mg_inv_sparse.multiply(Ml_sparse)
    """
    S_sparse=Mg_inv@Ml

    U,S,VT = svds(S_sparse, k=dim, which='LM')
    return calculate_vectors_alt(U,S,VT,dim)

def katz_partial(X,dim,prox_param):
    Mg,Ml = return_prox_mat_katz_partial(X, prox_param)

    print("Inside katz_partial, returned proximity matrices of size : Mg : ", Mg.shape, " and Ml : ", Ml.shape)
    s = time.time()
    c,s,x,u,v = pygsvd.gsvd(np.linalg.inv(Mg), Ml, full_matrices=True,extras='uv')
    print("Time taken for GSVD :" , time.time() - s)

    print("c :", c.shape)
    print("s :", s.shape)
    print("u :", u.shape)
    print("v :", v.shape)
    c_list = c.tolist()
    s_list = s.tolist()

    return calculate_vectors(u[0:dim,:],c_list[0:dim],v[0:dim,:],s_list[0:dim],dim)

def incr_katz_full(X,A,B,dim,prox_param):
    Mg,Ml = return_prox_mat_katz_full(X, prox_param)
    Mg_inv =  inv(Mg)
    S_sparse  = Mg_inv@Ml
    U, S, VT = svds(S_sparse, k=dim, which='LM')
    print("S_dense :" ,S_sparse.shape)
    Up,Sp,Vp = svdUpdate(U, S, VT.T, ((-1)*prox_param)*A, B, dim)
    print("Computed svd updates for S_sparse")

    return calculate_vectors_alt(Up,Sp,Vp.T,dim)

def svd_hope_full(X,dim, prox_param):
    Mg,Ml = return_prox_mat_katz_full(X, prox_param)
    Mg_inv =  inv(Mg)
    S_dense  = Mg_inv@Ml
    S_sparse = csr_matrix(S_dense)
    
    U, S, VT = svds(S_sparse, k=dim, which='LM')
    return U, S, VT.T

def incr_katz_full_series(U, S, V,A,B,dim,prox_param):
    Up,Sp,Vp = svdUpdate(U, S, V, ((-1)*prox_param)*A, B, dim)
    
    print("Up :",Up.shape)
    print("Sp : ", Sp.shape)
    print("Vp :" ,Vp.shape)
    print("Computed svd updates for S_dense")

    src, trgt = calculate_vectors_alt(Up,Sp,Vp.T,dim)

    return src, trgt, Up, Sp, Vp

def incr_hope_partial(X,A,B,dim,prox_param):
    Mg,Ml = return_prox_mat_katz_partial(X, prox_param)

    U1, S1, VT1 = svds(Mg, k=dim, which='LM')
    U2, S2, VT2 = svds(Ml, k=dim, which='LM')

    print("Computed full SVD of Mg and Ml\n")

    U1p,S1p,V1p = svdUpdate(U1, S1, VT1.T, ((-1)*prox_param)*A, B, len(S1))
    print("Computed svd updates for Mg")
    U2p,S2p,V2p = svdUpdate(U2, S2, VT2.T, (prox_param)*A, B, len(S2))
    print("Computed svd updates for Ml")

    return calculate_vectors(V1p.T,S1p,V2p.T,S2p,k)

def common_nghbrs(X, dim):
    Ml = return_prox_mat_common(X)
    U,S,VT = svds(Ml, k=dim, which='LM')
    return calculate_vectors_alt(U,S,VT,dim)

def incr_common_nghbrs_series(U, S, V, A, B, dim):
    Up,Sp,Vp = GenSvdUpdate(U, S, V, A, B, dim)
    
    print("Up :",Up.shape)
    print("Sp : ", Sp.shape)
    print("Vp :" ,Vp.shape)

    src, trgt = calculate_vectors_alt(Up,Sp,Vp.T,dim)

    return src, trgt, Up, Sp, Vp

def common_nghbrs_factorize(X,Y, non_zero_idx):
    """
    factorize the term (X*Y.T)**2 to the form of (P*Q.T) - given X, Y; find P and Q
    assumption : X and Y are both n-by-1 vectors
    """
    P = X
    n = len(X)
    Z = np.outer(X, Y)
    ZZ = np.matmul(Z,Z)
    Q = np.zeros((n, 1), dtype=np.float64)
    for i in range(0, n):
        Q[i][0] = ZZ[non_zero_idx][i] / P[non_zero_idx][0]

    return P, Q

def adamic_adar(X, dim):
    Mg,Ml = return_prox_mat_aa(X)
    S_sparse = Mg@Ml
    U,S,VT = svds(S_sparse, k=dim, which='LM')
    return calculate_vectors_alt(U,S,VT,dim)

#write any N by d matrix to a text file
def save_matrix(output_graph_file, m):
    N = len(m)
    d = len(m[0])
    for i in range(0,N):
        m_st = ""
        for j in range(0,d):
            m_st = m_st + str(m[i][j]) + " "
        m_st = m_st + "\n"
        with open(output_graph_file, "a") as f:
            f.write(m_st)

def save_embeddings_bin(output_embs_prefix, source, target):
    source_as_list = source.flatten()
    target_as_list = target.flatten()
    
    if os.path.exists(output_embs_prefix + "_source.bin"):
        os.remove(output_embs_prefix + "_source.bin")
    if os.path.exists(output_embs_prefix + "_target.bin"):
        os.remove(output_embs_prefix + "_target.bin")
    
    source.tofile(output_embs_prefix+"_source.bin", sep='')
    target.tofile(output_embs_prefix+"_target.bin", sep='')

def save_mappings(output_embs_prefix):
    global mapping_b
    global num_vertices
    if os.path.exists(output_embs_prefix + "_mapping.bin"):
        os.remove(output_embs_prefix + "_mapping.bin")
    
    mapping_b_mat = np.asarray(list(mapping_b.values()) ,  dtype=np.uint64).reshape(num_vertices, 1)
    mapping_b_mat.tofile(output_embs_prefix + "_mapping.bin",sep='')

def compare_prox_mat_svd(X, A, B, dim, prox_param):
    X_hat = X + np.outer(A, B)
    Mg_up, Ml_up = return_prox_mat_katz_partial(X_hat, prox_param)
    print("Computed proximity matrices for updated matrix\n")

    U1_up, S1_up, VT1_up = svds(Mg_up, k=dim, which='LM')
    U2_up, S2_up, VT2_up = svds(Ml_up, k=dim, which='LM')
    print("Computed full SVD on proximity matrices of updated matrix\n")
    with open("S1_up", "w+") as f:
        f.write(str(S1_up))
    with open("S2_up", "w+") as f:
        f.write(str(S2_up))

    Mg_incr, Ml_incr = return_prox_mat_katz_partial(X, prox_param)
    print("Computed proximity matrices for initial matrix\n")
    U1, S1, VT1 = svds(Mg_incr, k=dim, which='LM')
    U2, S2, VT2 = svds(Ml_incr, k=dim, which='LM')
    print("Computed full SVD on proximity matrices of initial matrix\n")
    
    U1_incr,S1_incr,V1_incr = GenSvdUpdate(U1, S1, VT1.T, (-1)*prox_param*A, B, dim)
    U2_incr,S2_incr,V2_incr = GenSvdUpdate(U2, S2, VT2.T, prox_param*A, B, dim)
    print("Completed incr_svd on proximity matrices of updated matrix\n")
    with open("S1_incr", "w+") as f:
        f.write(str(S1_incr))
    with open("S2_incr", "w+") as f:
        f.write(str(S2_incr))
    print("Done comparison and saved results to file")

def compare_svd(X, A, B, dim):
    X_hat = X + np.outer(A, B)
    U_hat, S_hat, VT_hat = svds(X_hat, k=dim, which='LM')
    print("Computed full SVD on updated matrix\n")
    with open("S_hat", "w+") as f:
        f.write(str(S_hat))
    
    U, S, VT = svds(X, k=dim, which='LM')
    print("Computed full SVD on initial matrix\n")
    
    U_incr,S_incr,V_incr = GenSvdUpdate(U, S, VT.T, A, B, dim)
    print("Completed incr_svd on updated matrix\n")
    with open("S_incr", "w+") as f:
        f.write(str(S_incr))
    print("Done comparison and saved results to file")

def static_embs():
    # arguments : input graph as bin, file name prefix for output embedding files, embedding dim
    input_graph_file = str(sys.argv[1])
    # describe graph format
    output_embs_prefix = str(sys.argv[2])
    dimensions = int(sys.argv[3])
    
    X = load_graph(input_graph_file)
    beta = ret_prox_param(X)
    beta = 7
    print("beta :", beta)

    #enter desired embedding function in the following line
    s = time.time()
    source, target = katz_partial(X,dimensions,beta)
    print(time.time()-s)
    save_embeddings_bin(output_embs_prefix, source, target)
    save_mappings(output_embs_prefix)

def incr_svd_test():
    """
    arguments : input smaller graph as bin, input larger graph with added edges as bin , embedding dimensions
    """
    input_small_graph_file = str(sys.argv[1])
    input_large_graph_file = str(sys.argv[2])
    dim = int(sys.argv[3])
    
    X_hat = load_graph(input_large_graph_file)
    X = load_graph_with_mapping(input_small_graph_file)

    global num_vertices
    A = np.zeros((num_vertices,1), dtype=np.float64)
    B = np.zeros((num_vertices,1), dtype=np.float64)

    #assuming only single row has changed
    X_diff = X_hat - X
    diff_vertex = 0
    for i in range(0,num_vertices):
        for j in range(0,num_vertices):
            if(X_diff[i][j] != 0):
                diff_vertex = i
                break

    A[diff_vertex][0] = np.float64(1.0)
    Y = X_hat[diff_vertex]-X[diff_vertex]
    B = Y.reshape(num_vertices,1)

    compare_svd(X, A, B, dim)

def incr_hope_test():
    """
    arguments : input smaller graph as bin, input larger graph with added edges as bin 
    file name prefix for output embedding files, embedding dim
    """
    input_small_graph_file = str(sys.argv[1])
    output_embs_prefix = str(sys.argv[2])
    dimensions = int(sys.argv[3])
    
    X = load_graph(input_small_graph_file)
    X_hat = np.copy(X)

    #delete some edges in a single row of X to create a finite diff
    for i in range(0,len(X[0])):
        if(i%2!=0):
            X[0][i] = 0

    A = np.zeros((len(X),1), dtype=np.float64)
    B = np.zeros((len(X),1), dtype=np.float64)
    A[0][0] = 1
    Y = X_hat[0]-X[0]
    B = Y.reshape(len(X),1)
    #beta = ret_prox_param(X)
    beta = 0.0001

    source, target = incr_katz_full(X,A,B,dimensions,beta)
    save_embeddings_bin(output_embs_prefix, source, target)
    save_mappings(output_embs_prefix)

def incr_hope():
    """
    arguments : input smaller graph as bin, input larger graph with added edges as bin 
    file name prefix for output embedding files, embedding dim
    """
    input_small_graph_file = str(sys.argv[1])
    input_large_graph_file = str(sys.argv[2])
    output_embs_prefix = str(sys.argv[3])
    dimensions = int(sys.argv[4])
    
    X_hat = load_graph(input_large_graph_file)
    X = load_graph_with_mapping(input_small_graph_file)

    global num_vertices
    A = np.zeros((num_vertices,1), dtype=np.float64)
    B = np.zeros((num_vertices,1), dtype=np.float64)

    #assuming only single row has changed
    X_diff = X_hat - X
    diff_vertex = 0
    for i in range(0,num_vertices):
        for j in range(0,num_vertices):
            if(X_diff[i][j] != 0):
                diff_vertex = i
                break

    A[diff_vertex][0] = np.float64(1)
    Y = X_hat[diff_vertex]-X[diff_vertex]
    B = Y.reshape(num_vertices,1)
    #beta = ret_prox_param(X)
    beta = 0.000001

    source, target = incr_katz_full(X,A,B,dimensions,beta)
    save_embeddings_bin(output_embs_prefix, source, target)
    save_mappings(output_embs_prefix)

# calculate simple diff between adj matrices corresponding to initial and final graphs, also keep a record of which vertices already had some edges in initial graph and which have been added in the final one
def calc_diff(X1,X2):
    global num_vertices
    vert_added = []
    edge_added = []
    X_diff = X2-X1

    for r in range(0,num_vertices):
        for c in range(0,num_vertices):
            if X_diff[r][c] != 0:
                if np.any(X1[r]):
                    edge_added.append(r)
                    break
                else:
                    vert_added.append(r)
                    break

    return X_diff, edge_added, vert_added

# the static SVD needs to be performed only once. Subsequently, the results of the previous incr-SVD are used
def saturate_edges_katz(X1, X_diff, edge_added, output_embs_prefix, dimensions):
    X_s = X1
    beta = 0.0001
    Ui, Si, Vi = svd_hope_full(X_s, dimensions, beta)
    global num_vertices
    global count

    for x in range(0,len(edge_added)):
        i = edge_added[x]
        A = np.zeros((num_vertices,1), dtype=np.float64)
        B = np.zeros((num_vertices,1), dtype=np.float64)
        A[i][0] = np.float64(1)
        B = X_diff[i].reshape(num_vertices,1)
        source, target, Up, Sp, Vp = incr_katz_full_series(Ui, Si, Vi, A, B, dimensions, beta)
        np.reshape(Up, np.shape(Ui))
        Ui = Up
        np.reshape(Sp, np.shape(Si))
        Si = Sp
        np.reshape(Vp, np.shape(Vi))
        Vi = Vp
        if(x == len(edge_added) - 1):
            save_embeddings_bin(output_embs_prefix + "_" + str(count), source, target)
        #save_embeddings_bin(output_embs_prefix + "_" + str(count), source, target)
        count = count + 1
        X_s = X_s + np.outer(A,B)

    return X_s

def saturate_vertices_katz(X1, X_diff, vert_added, output_embs_prefix, dimensions):
    X_s = X1
    global num_vertices
    global count
    beta = 0.0001
    Ui, Si, Vi = svd_hope_full(X_s, dimensions, beta)

    for x in range(0,len(vert_added)):
        i = vert_added[x]
        A = np.zeros((num_vertices,1), dtype=np.float64)
        B = np.zeros((num_vertices,1), dtype=np.float64)
        A[i][0] = np.float64(1)
        B = X_diff[i].reshape(num_vertices,1)
       # source, target = incr_katz_full(X_s,A,B,dimensions,beta)
        source, target, Up, Sp, Vp = incr_katz_full_series(Ui, Si, Vi, A, B, dimensions, beta)
        Ui = Up
        Si = Sp
        Vi = Vp
        if(x == len(vert_added) - 1):
            save_embeddings_bin(output_embs_prefix + "_" + str(count), source, target)
        count = count + 1
        X_s = X_s + np.outer(A,B)
    return X_s

def update_common_nghbrs(X1, X_diff, edge_added, output_embs_prefix, dim):
    X_s = X1
    global num_vertices
    global count
    Ml = return_prox_mat_common(X_s)
    s = time.time()
    Ui,Si,VTi = svds(Ml, k=dim, which='LM')
    print("Full SVD time : ", time.time()-s)
    Vi = VTi.T

    for x in range(0,len(edge_added)):
        print("count :", count)
        i = edge_added[x]
        A = np.zeros((num_vertices,1), dtype=np.float64)
        B = np.zeros((num_vertices,1), dtype=np.float64)
        A[i][0] = np.float64(1)
        B = X_diff[i].reshape(num_vertices,1)
        A1 = np.matmul(X_s,A)
        B1 = B
        A2 = A
        B2 = np.matmul(X_s.T, B)
        A3,B3 = common_nghbrs_factorize(A,B,i)

        U1,S1,V1 = GenSvdUpdate(Ui, Si, Vi, A1, B1, dim)
        U2, S2, V2 = GenSvdUpdate(U1, S1, V1, A2, B2, dim)
        source, target, Up, Sp, Vp = incr_common_nghbrs_series(U2, S2, V2, A, B, dim)
        Ui = Up
        Si = Sp
        Vi = Vp
        if(x == len(edge_added) - 1):
            save_embeddings_bin(output_embs_prefix + "_" + str(count), source, target)
        count = count + 1
        X_s = X_s + np.outer(A,B)
        break
    return X_s    

def update_common_nghbrs(X1, X_diff, src_dict, edge_added, output_embs_prefix, dim):
    X_s = X1
    global num_vertices
    global count
    Ml = return_prox_mat_common(X_s)
    s = time.time()
    Ui,Si,VTi = svds(Ml, k=dim, which='LM')
    print("Full SVD time : ", time.time()-s)
    Vi = VTi.T

    for x in range(0,len(edge_added)):
        print("count :", count)
        u = edge_added[x]
        A = np.zeros((num_vertices,num_vertices), dtype=np.float64)
        B = np.zeros((num_vertices,num_vertices), dtype=np.float64)
        A[u][u] = 1.0
        for s in src_dict[u]:
            A[s][s] = 1.0
            for v in range(0,len(X_diff[u])):
                if(X_diff[u][v] != 0):
                    B[s][v] = 1.0
        for v in range(0,len(X_diff[u])):
            if(X_diff[u][v] != 0):
                if v not in src_dict.keys():
                    src_dict[v] = []
                src_dict[v].append(u)
                for d in range(0,len(X1[v])):
                    if(X1[v][d] != 0):
                        B[u][d] += 1.0
        source, target, Up, Sp, Vp = incr_common_nghbrs_series(Ui,Si,Vi,A,B,dim,)
        Ui = Up
        Si = Sp
        Vi = Vp
        if(x == len(edge_added) - 1):
            save_embeddings_bin(output_embs_prefix + "_" + str(count), source, target)
        count = count + 1
        X_s = X_s + np.outer(A,B)
    return X_s 

    """
    for x in range(0,len(edge_added)):
        print("count :", count)
        i = edge_added[x]
        A = np.zeros((num_vertices,1), dtype=np.float64)
        B = np.zeros((num_vertices,1), dtype=np.float64)
        A[i][0] = np.float64(1)
        B = X_diff[i].reshape(num_vertices,1)
        A1 = np.matmul(X_s,A)
        B1 = B
        A2 = A
        B2 = np.matmul(X_s.T, B)
        A3,B3 = common_nghbrs_factorize(A,B,i)

        U1,S1,V1 = GenSvdUpdate(Ui, Si, Vi, A1, B1, dim)
        U2, S2, V2 = GenSvdUpdate(U1, S1, V1, A2, B2, dim)
        source, target, Up, Sp, Vp = incr_common_nghbrs_series(U2, S2, V2, A, B, dim)
        Ui = Up
        Si = Sp
        Vi = Vp
        if(x == len(edge_added) - 1):
            save_embeddings_bin(output_embs_prefix + "_" + str(count), source, target)
        count = count + 1
        X_s = X_s + np.outer(A,B)
    return X_s
    """    

# runs a series of rank-1 svdUpdates and calculates embeddings after each update (currently : does not save any except the final stage, uses full hope)
def incr_hope_series():
    """
    arguments : input smaller graph as bin, input larger graph with added edges as bin 
    file name prefix for output embedding files, embedding dim
    """
    input_small_graph_file = str(sys.argv[1])
    input_large_graph_file = str(sys.argv[2])
    output_embs_prefix = str(sys.argv[3])
    dimensions = int(sys.argv[4])
    
    # load final graph first to prepare a mapping of all vertices
    X2 = load_graph(input_large_graph_file)
    # initial(smaller) graph is loaded afterwards but padded with zero rows and columns to enable calculation of diff with final graph
    X1 = load_graph_with_mapping(input_small_graph_file)
    global num_vertices
    print("num_vertices :",num_vertices)
    print("X2 :" ,X2.shape)
    print("X1 :",X1.shape)

    X_diff, edge_added, vert_added = calc_diff(X1,X2)

    print("edge_added :",len(edge_added))
    print("vert_added :", len(vert_added))

    edge_added.extend(vert_added)
    save_mappings(output_embs_prefix)
    # first update edge additions.
    X_hat = saturate_edges_katz(X1, X_diff, edge_added, output_embs_prefix, dimensions)
    # NOTE : currently, vertex additions also behave like edge additions, a row which was entirely zero in initial graph is populated with ones
    #X_hat = saturate_vertices_katz(X_intmdt, X_diff, vert_added, output_embs_prefix, dimensions)

    print("diff between X_hat and X2 :" ,np.any(X_hat-X2))

def incr_common_nghbrs():
    """
    arguments : input smaller graph as bin, input larger graph with added edges as bin 
    file name prefix for output embedding files, embedding dim
    """
    input_small_graph_file = str(sys.argv[1])
    input_large_graph_file = str(sys.argv[2])
    output_embs_prefix = str(sys.argv[3])
    dimensions = int(sys.argv[4])

    # load final graph first to prepare a mapping of all vertices
    X2 = load_graph(input_large_graph_file)
    # initial(smaller) graph is loaded afterwards but padded with zero rows and columns to enable calculation of diff with final graph
 #   X1 = load_graph_with_mapping(input_small_graph_file)
    X1,src_dict = load_graph_with_mapping_in_nghbrs(input_small_graph_file)
    global num_vertices
    print("num_vertices :",num_vertices)
    print("X2 :" ,X2.shape)
    print("X1 :",X1.shape)

    X_diff, edge_added, vert_added = calc_diff(X1,X2)

    print("edge_added :",len(edge_added))
    print("vert_added :", len(vert_added))

    edge_added.extend(vert_added)
    save_mappings(output_embs_prefix)

    # first update edge additions.
    # NOTE : currently, vertex additions also behave like edge additions, a row which was entirely zero in initial graph is populated with ones
    X_hat = update_common_nghbrs(X1, X_diff,src_dict, edge_added, output_embs_prefix, dimensions)
 #   X_hat = update_common_nghbrs(X_inmdt, X_diff, vert_added, output_embs_prefix, dimensions)
   
if __name__ == "__main__":
    #incr_common_nghbrs()
    static_embs()
