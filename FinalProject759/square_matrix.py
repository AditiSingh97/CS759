import math
import argparse
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

A = np.array([[0,0,0,0,0],[0,0,1,0,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])
B = np.matmul(A,A)
print(B)

A = np.array([[0,0,0,0,0],[0,0,1,0,1],[1,0,0,1,1],[1,0,0,0,0],[1,0,0,0,0]])
C = np.matmul(A,A)
print(C)

D = C-B
print(D)
E = np.eye(5)
E[3][3]=0
E[4][4]=0

print(np.matmul(E,D))