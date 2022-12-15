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
    n = int(sys.argv[1])
    y = cp.asarray(np.random.rand(n,n))
    print(n)
    s=time.time()
    q, r = cp.linalg.qr(y)
    print(time.time()-s)
    del y
    gc.collect()

