import argparse
from collections import OrderedDict
import csv
import re
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import os
from queue import Queue
from random import randint
import struct
import sys

def txt_to_bin(input_file, output_file):
    edges = np.fromfile(input_file, dtype=np.int32,count=-1,sep='\n').reshape(-1,2);
    data = np.zeros((len(edges),3),dtype=np.int32)
    cur = 0
    for edge in edges:
        data[cur][0] = edge[0]
        data[cur][1] = 0
        data[cur][2] = edge[1]
        cur = cur+1
    
    data.tofile(output_file,sep='')

def main():

    input_graph_filename  = str(sys.argv[1])

    output_file = str(sys.argv[2])

    txt_to_bin(input_graph_filename, output_file)

if __name__ == "__main__":
    main()
