import argparse
from collections import OrderedDict
import csv
import re
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import os
import random
from queue import Queue
from random import randint
import struct
import sys

main_graph  = {}
derived_graph = {}
num_vertices = 0
num_edges = 0
num_vertices_derived = 0
num_edges_derived = 0
graph_frac = float(0.0)

max_degree_vertex_main = 0 #id of vertex in main graph with max degree, to be used as entry point for bfs
vertex_to_degree_dec = {}
visited = set()
queue = Queue(maxsize = -1)


def construct_input_graph(filename):
    edges = []
    all_vertices = set()
    with open(filename, mode='rb') as file: # b is important -> binary
            data = file.read()

            unpacked = struct.unpack("III" * ((len(data)) // 12), data)
            max_edges = int(len(unpacked) / 3)
            for i in range(0, max_edges):
                src = unpacked[i * 3]
                dest = unpacked[i * 3 + 2]
                new_edge = [src, 0, dest]
                all_vertices.add(src)
                all_vertices.add(dest)
                edges.append(new_edge)

    global num_vertices
    global num_edges
    global main_graph
    global vertex_to_degree_dec

    num_edges = len(edges)

    for edge in edges:
        src = edge[0]
        dest = edge[2]

        if(src not in main_graph.keys()):
            main_graph[src] = []
            main_graph[src].append(dest)
        else:
            if dest not in main_graph[src]:
                main_graph[src].append(dest)
    
    global max_degree_vertex_main
    max_degree = 0;
    for node in main_graph.keys():
        vertex_to_degree_dec[node] = len(main_graph[node])
        if(len(main_graph[node]) > max_degree):
            max_degree = len(main_graph[node])
            max_degree_vertex_main = node

    vertex_to_degree_dec = {k: v for k, v in sorted(vertex_to_degree_dec.items(), key=lambda item: item[1], reverse=True)}
    num_vertices = len(all_vertices)
    print("Input graph has " + str(num_vertices) + " ; " + str(num_edges))
    print("Vertex : ", max_degree_vertex_main, " has maximum degree of : " , max_degree , " in main graph")

def load_graph(filename):
    raw_data = np.fromfile(filename, dtype=np.uint64, count = -1)
    total_len = len(raw_data)
    graph_data = {}
    i = int(0);
    while i < total_len:
        begin = int(i+2)
        added = int(raw_data[i+1])
        end = int(i) + int(2) + added

        graph_data[np.uint64(raw_data[i])] = set(raw_data[begin:end].flatten())
        i = int(end)
        if(i >= total_len):
            break;

    return graph_data  

def addEdges(curr_vertex):
    nghbr_set = set()
    if curr_vertex in main_graph.keys():
        total_nghbrs = len(main_graph[curr_vertex])
        curr_degree = int(total_nghbrs * graph_frac)
        if(curr_degree < 1):
            curr_degree = 1
        #randomly sample neighbors of curr_degree

        for nghbr in main_graph[curr_vertex]:
            prob = random.uniform(0.0, 1.0)
            if(prob < graph_frac):
                nghbr_set.add(nghbr)
                if(len(nghbr_set) >= curr_degree):
                    break

    return nghbr_set


def bfs(vertex):
    # Mark the source node as
    # visited and enqueue it

    global visited
    global queue
    global num_vertices_derived
    global derived_graph
    print("num_vertices_derived = " + str(num_vertices_derived))
    
    queue.put(vertex)
    visited.add(vertex)
    vertex_count = 1;

    while not queue.empty():
        """
        Dequeue a vertex from
        queue and print it
        """
        vertex = queue.get()
 
        """
        Get all adjacent vertices of the
        dequeued vertex. If a adjacent vertex
        has not been visited, then mark it
        visited and enqueue it
        """

        if vertex in main_graph.keys():
            new_edges = addEdges(vertex)
            if (len(new_edges) > 0) and (vertex_count < num_vertices_derived):
                derived_graph[vertex] = set()
                for edge in new_edges:
                    if(vertex_count >= num_vertices_derived):
                        break
                    else:
                        derived_graph[vertex].add(edge)
                        if edge not in visited:
                            queue.put(edge)
                            visited.add(edge)
                            vertex_count += 1

#function takes a graph and derives a smaller representative graph from it
#graph_size is a float between 0 and 1.
def derive_subgraph():
    global num_vertices_derived
    print("subgraph fraction = " + str(graph_frac))
    num_vertices_derived = int(graph_frac * float(num_vertices))
    print("New subgraph will have " + str(num_vertices_derived) + " vertices\n")

    global derived_graph
    global visited
    global vertex_to_degree_dec

    curr_idx  = 0
    all_main_nodes = list(vertex_to_degree_dec.keys())
    while(len(visited) < num_vertices_derived):
        curr_vertex = all_main_nodes[curr_idx]
        print("Entry point for graph derivation BFS : ", curr_vertex)
        bfs(curr_vertex)
        print("BFS completed successfully. New graph has " + str(len(derived_graph.keys())) 
            + " sources, out of " + str(len(visited)) + " total vertices\n")
        curr_idx += 1
        if(curr_idx >= len(all_main_nodes)):
            break

    total_edges = 0
    max_degree = 0
    min_degree = len(derived_graph.keys())
    for vertex in derived_graph:
        curr_len = len(derived_graph[vertex])
        total_edges = total_edges + curr_len
        if(max_degree < curr_len):
            max_degree = curr_len
        if(min_degree > curr_len):
            min_degree = curr_len

    count = 0
    for vertex in derived_graph:
        count += 1
        if(count > 9):
            break

    print("Total edges : " + str(total_edges) + ". Max degree : " + str(max_degree)
            + ". Min degree : " + str(min_degree))

def write_graph(graph, output_graph_prefix, save_edges=False):
    graph_as_list = []
    edge_list = []
    for vertex in graph:
        graph_as_list.append(vertex)
        graph_as_list.append(len(graph[vertex]))
        for nghbr in graph[vertex]:
            edge_list.append(vertex)
            edge_list.append(0)
            edge_list.append(nghbr)
            graph_as_list.append(nghbr)

    edge_as_np = np.asarray(edge_list, dtype=np.int32).reshape(-1,3)
    graph_as_np = np.asarray(graph_as_list, dtype=np.uint64)
    
    with open(output_graph_prefix+".bin", "wb") as f:
        f.write(bytes(graph_as_np))
    
    if(save_edges):
        with open(output_graph_prefix+"_edges.bin", "wb") as f:
            f.write(bytes(edge_as_np))



#file will take as input a graph (as output by preprocess.py), number of nodes or edges to be added/or deleted, their ids and other details
#will output another graph in the same format as preprocess.py with these changes made

#####Types of Changes#####

#1. Function to load graph tensor dumps

#2. Function to initialize a subgraph from these tensors  - initially 50%

#3 Function to add a node

#Function to add a random edge

def addNewEdge():
    #pick a random vertex that has not saturated its degree from the parent graph
    #add new neighbor to it
    global main_graph
    global derived_graph
    global num_vertices
    global num_vertices_derived

    vertex_to_add_edges = 0
    for vertex in derived_graph.keys():
        if(len(derived_graph[vertex]) < len(main_graph[vertex])):
            vertex_to_add_edges = vertex
            break
    
    for nghbr in main_graph[vertex_to_add_edges]:
        if nghbr not in derived_graph[vertex_to_add_edges]:
            derived_graph[vertex_to_add_edges].add(nghbr)
            break
    """
    entry_point = 0;
    entry_point_idx = 0;
    all_keys = list(derived_subgraph.keys())
    count = 0
    while(True):
        entry_point_idx = randint(0,num_vertices_derived-1)
        count += 1
        if(len(derived_subgraph[all_keys[entry_point_idx]]) < len(main_graph[all_keys[entry_point_idx]])):
            entry_point = all_keys[entry_point_idx]
            break
        if(count == num_vertices_derived):
            print("Could not find a vertex with unsaturated degree....exiting function")
            exit(-1)

    print("New edge will be added to : " + str(entry_point))
    for nghbr in main_graph[entry_point]:
        if nghbr not in derived_subgraph[entry_point]:
            derived_subgraph[entry_point].add[nghbr]
            break

    print(str(derived_subgraph[entry_point]))
    """

def removeEdge(derived_subgraph):
    global main_graph
    global num_vertices_derived

    entry_point = 0;
    entry_point_idx = 0;
    all_keys = list(derived_subgraph.keys())
    count = 0
    while(True):
        entry_point_idx = randint(0,num_vertices_derived-1)
        count += 1
        if(len(derived_subgraph[all_keys[entry_point_idx]]) > 0):
            entry_point = all_keys[entry_point_idx]
            break
        if(count == num_vertices_derived):
            print("Could not find a vertex with non-zero degree....exiting function")
            exit(-1)

    print("New edge will be deleted from : " + str(entry_point))
    for nghbr in derived_subgraph[entry_point]:
        derived_subgraph[entry_point].remove(nghbr)
        break

def addNewVertex(derived_subgraph):
    global main_graph
    global num_vertices

    new_vertex = 0
    new_vertex_idx = 0

    all_vertices = list(main_graph.keys())
    while(True):
        new_vertex_idx = randint(0, num_vertices-1)
        if(all_vertices[new_vertex_idx] not in derived_subgraph):
            new_vertex = all_vertices[new_vertex_idx]
            break
        else:
            continue

    derived_subgraph[new_vertex] = set()
    for nghbr in main_graph[new_vertex]:
        derived_subgraph[new_vertex].add(nghbr)

def mutate_graph(derived_subgraph, frac_add_edges = 0.0, frac_remove_edges = 0.0):
    num_add_edges = int(frac_add_edges * num_edges)
    num_remove_edges = int(frac_remove_edges * num_edges)

    for i in range(0, num_add_edges):
        addNewEdge(derived_subgraph)
    
    total_edges = 0
    max_degree = 0
    min_degree = len(derived_graph.keys())
    for vertex in derived_graph:
        curr_len = len(derived_graph[vertex])
        total_edges = total_edges + curr_len
        if(max_degree < curr_len):
            max_degree = curr_len
        if(min_degree > curr_len):
            min_degree = curr_len
    
    print("Total edges : " + str(total_edges) + ". Max degree : " + str(max_degree)
            + ". Min degree : " + str(min_degree))

def gen_graph():
    input_edges_filename  = str(sys.argv[1])
    output_subgraph_prefix = str(sys.argv[2])
    construct_input_graph(input_edges_filename)
    write_graph(main_graph, output_subgraph_prefix)

def gen_subgraph():
    input_edges_filename  = str(sys.argv[1])
    subgraph_fraction = float(sys.argv[2])
    output_subgraph_prefix = str(sys.argv[3])
    construct_input_graph(input_edges_filename)

    global graph_frac
    graph_frac = subgraph_fraction
    derive_subgraph()
    write_graph(derived_graph, output_subgraph_prefix, True)

def gen_mutated_graph():
    input_main_graph_filename  = str(sys.argv[1])
    input_derived_graph_filename  = str(sys.argv[2])
    output_subgraph_prefix = str(sys.argv[3])

    global main_graph
    global derived_graph
    global num_vertices
    global num_vertices_derived
    main_graph = load_graph(input_main_graph_filename)
    derived_graph = load_graph(input_derived_graph_filename)
    num_vertices = len(main_graph.keys())
    num_vertices_derived = len(derived_graph.keys())
    print(num_vertices_derived)

    addNewEdge()
    print(len(derived_graph.keys()))
    write_graph(derived_graph, output_subgraph_prefix, True)

if __name__ == "__main__":
    gen_graph()
