import numpy as np
import time
import sys
import struct

from argparse import ArgumentParser

from collections import defaultdict
import bisect
import random
import traceback
NUM_NEG = 100

def sample_negative_edges(edges, num_neg, num_nodes, src=None, dst=None):
    neg_edges = []
    if src != None:
        while len(neg_edges) < num_neg:
            j = random.randint(0, num_nodes - 1)
            edge = (src, j)
            if edge not in edges:
                neg_edges.append((src, j))
    elif dst != None:
        while len(neg_edges) < num_neg:
            j = random.randint(0, num_nodes - 1)
            edge = (j, dst)
            if edge not in edges:
                neg_edges.append((j, dst))
    else:
        while len(neg_edges) < num_neg:
            i = random.randint(0, num_nodes - 1)
            j = random.randint(0, num_nodes - 1)
            edge = (i, j)
            if edge not in edges:
                neg_edges.append((i, j))

    return neg_edges

# Returns a list of (src, dst)
def parse_cora_edges_bin(edge_f):
    edge_list = np.fromfile(edge_f, dtype=np.int32).reshape(-1,3)
    edges = set()
    vertices = set()
    for edge in edge_list:
        edges.add((np.uint64(edge[0]), np.uint64(edge[2])))
        vertices.add(np.uint64(edge[0]))
        vertices.add(np.uint64(edge[2]))
    return edges, vertices

def eval_embed(edge_dir, embed_dir, d):

    t0 = time.time()
    train_edges, train_vertices = parse_cora_edges_bin(edge_dir+"/train_edges.bin")
    test_edges, test_vertices = parse_cora_edges_bin(edge_dir+"/test_edges.bin")
    all_edges = train_edges.union(test_edges)
    all_vertices = train_vertices.union(test_vertices)
    num_edges = len(all_edges)
    num_nodes = np.uint64((max(all_vertices)) + 1)
    num_train_edges = len(train_edges)
    num_test_edges = len(test_edges)
    num_train_vert = len(train_vertices)
    num_test_vert = len(test_vertices)
    print("Parsed graph with ", num_edges, " edges and", num_nodes, " nodes")

    train_src_embs = np.fromfile(embed_dir + "_source.bin", dtype=np.float64).reshape(num_train_vert, d)
    train_dst_embs = np.fromfile(embed_dir + "_target.bin", dtype=np.float64).reshape(num_train_vert, d)
    mapping = np.fromfile(embed_dir + "_mapping.bin", dtype=np.uint64).reshape(num_train_vert,1)
    
    node_src_embs = np.zeros((num_nodes,d), dtype=np.float64)
    node_dst_embs = np.zeros((num_nodes,d), dtype=np.float64)
    
    count = 0;
    for vertex in mapping:
        node_src_embs[vertex] = train_src_embs[count]
        node_dst_embs[vertex] = train_dst_embs[count]
        count += 1 

    t1 = time.time()
    print("Load Data: {:.3f}".format(t1-t0))
    hits = defaultdict(int)
    scores = {}

        # Get the rank of every true edge in the graph
    unseen = 0
    for edge in test_edges:
        try:
            src = edge[0]
            dst = edge[1]
            if(src not in train_vertices or dst not in train_vertices):
                unseen += 1
            key = ""
            key = str(src) + ":" + str(dst)
            pos_score = 0.0;
            if(key not in scores.keys()):
                src_embs = node_src_embs[src]
                dst_embs = node_dst_embs[dst]
                pos_score = np.dot(src_embs,dst_embs)
                scores[key] = pos_score
            else:
                pos_score = scores[key]

            neg_edges = sample_negative_edges(all_edges, NUM_NEG, num_nodes, dst=dst)
            # Scores are sorted in ascending order
            neg_scores = []
            for e in neg_edges:
                src = e[0]
                dst = e[1]
                key = ""
                key = str(src) + ":" + str(dst)
                neg_score = 0.0;
                if(key not in scores.keys()):
                    src_embs = node_src_embs[src]
                    dst_embs = node_dst_embs[dst]
                    neg_score = np.dot(src_embs,dst_embs)
                    scores[key] = neg_score
                else:
                    neg_score = scores[key]
                neg_scores.append(neg_score)
            
            neg_scores_sorted = sorted(neg_scores)
            # Find the place where pos_edge would be inserted in sorted list
            rank = bisect.bisect_right(neg_scores_sorted, pos_score)
            # since list was in ascending order, the rank we want is reversed
            rank = NUM_NEG - rank
            if rank == 0:
               hits[0] += 1
            if rank < 5:
                hits[4] += 1
            if rank < 10:
                hits[9] += 1
            if rank < 20:
                hits[19] += 1
        except Exception as e:
            traceback.print_exc()
            sys.exit(-1)

    unseen = float(unseen)
    print("#test edges with at least one unseen vertex : ", unseen)
    for k, v in hits.items():
        print("Hits at ", k , " is ", float(v)/num_test_edges)

def main():
    edge_dir = str(sys.argv[1])
    embed_prefix = str(sys.argv[2])
    dim = int(sys.argv[3])
    eval_embed(edge_dir, embed_prefix,dim)

if __name__ == '__main__':
    main()
