import sys
import os
import numpy as np
import os
import random

def main():

    filename  = str(sys.argv[1])

    n = int(sys.argv[2])
    A = np.zeros((n,1), dtype=np.uint32)
    B = np.zeros((n,1), dtype=np.uint32)

    A[n-1][0] = int(1)
    print(np.shape(A))

    for i in range(0,n):
        x = random.uniform(0,1)
        if(x>0.5):
            B[i][0]=int(1)

    print(np.shape(B))
    C = np.append(A,B)
    print(np.shape(C))

    with open(filename, "wb") as f:
        f.write(bytes(C))


if __name__ == "__main__":
    main()
