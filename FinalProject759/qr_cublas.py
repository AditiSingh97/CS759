import math
import numpy as np
import logging
import struct
import sys
import time
import numba.cuda as cuda
import cupy as cp
from cupyx.profiler import benchmark
import os
import gc

gc.collect()

if __name__ == "__main__":
    i = int(sys.argv[1])
    n = pow(2, i)
    y = cp.asarray(np.random.rand(n,n))
    print(n)
    print(benchmark(cp.linalg.qr, (y, ), n_repeat=1000))
    del y
    gc.collect()

