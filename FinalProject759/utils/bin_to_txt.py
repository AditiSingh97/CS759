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

def bin_to_txt(input_file, output_file):
    edges = np.fromfile(input_file, dtype=np.int32).reshape(-1,3);
    for edge in edges:
        data = ""
        data = data + str(edge[0]) + ' ' + str(edge[2]) + '\n'
        with open(output_file, mode='a') as file:
            file.write(str(data))

def main():

    input_graph_filename  = str(sys.argv[1])

    output_file = str(sys.argv[2])

    bin_to_txt(input_graph_filename, output_file)

if __name__ == "__main__":
    main()
