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

def split_test_train(input_file, output_dir, split_ratio):
    edges = np.fromfile(input_file, dtype=np.int32).reshape(-1,3);
    num_edges = len(edges)
    print(num_edges)
    num_test = int(split_ratio * num_edges)

    test_edges = edges[0:num_test,:]
    print(test_edges.shape)
    test_edges.tofile(output_dir+"/test_edges.bin",sep='')
    train_edges = edges[num_test:,:]
    print(train_edges.shape)
    train_edges.tofile(output_dir+"/train_edges.bin",sep='')

def split_slice(input_file, output_file, split_ratio):
    edges = np.fromfile(input_file, dtype=np.int32).reshape(-1,3);
    num_edges = len(edges)
    print(num_edges)
    num_slice = int(split_ratio * num_edges)

    slice_edges = edges[0:num_slice,:]
    print(slice_edges.shape)
    slice_edges.tofile(output_file,sep='')

def split_and_save_all():

    input_graph_filename  = str(sys.argv[1])
    output_dir = str(sys.argv[2])
    split_ratio = float(sys.argv[3])

    split_test_train(input_graph_filename, output_dir, split_ratio)

def split_and_save_slice():

    input_graph_filename  = str(sys.argv[1])
    output_file = str(sys.argv[2])
    split_ratio = float(sys.argv[3])

    split_slice(input_graph_filename, output_file, split_ratio)

if __name__ == "__main__":
    split_and_save_slice()
