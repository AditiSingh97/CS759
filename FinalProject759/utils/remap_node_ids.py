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

def remap_ids(input_edges_file, output_edges_file, output_map_file):
    edges = np.fromfile(input_edges_file, dtype=np.int32).reshape(-1,3);
    id_map = {}
    curr_count = 0
    unique_nodes = set()
    for edge in edges:
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[2])
    for node in unique_nodes:
        id_map[node] = curr_count
        curr_count += 1
    for edge in edges:
        edge[0] = id_map[edge[0]]
        edge[2] = id_map[edge[2]]

    edges.tofile(output_edges_file,sep='')
    mapping = np.asarray(list(id_map.items())).reshape(-1,2)
    mapping.tofile(output_map_file,sep='')

def main():

    input_edges_file  = str(sys.argv[1])
    output_edges_file = str(sys.argv[2])
    output_map_file = str(sys.argv[3])
    remap_ids(input_edges_file, output_edges_file, output_map_file)

if __name__ == "__main__":
    main()
