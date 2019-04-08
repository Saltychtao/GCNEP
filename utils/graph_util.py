
import numpy as np
import torch
import dgl
import math

#######################################################################
#
# Utility function for building training graphs
#
#######################################################################


def build_graph_from_adj_matrix(adj_matrix,device):

    num_nodes = len(adj_matrix)
    print('Total number of relations :{}'.format(num_nodes))
    # adj_matrix = (adj_matrix > 0).astype("int")
    # G = nx.from_numpy_matrix(adj_matrix,create_using=nx.MultiDiGraph(),parallel_edges=True)
    g = dgl.DGLGraph(multigraph=True)
    src,dst = [],[]
    cnt = 0
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if cnt % 1000 ==0 :
                print('\r{}'.format(cnt),end='')
            cnt += 1
            num_of_edges = int(adj_matrix[i][j])
            n = 0
            if num_of_edges  == 0:
                n = 0
            else:
                n = max(1+round(math.log2(num_of_edges)),10)
            for k in range(n):
                src.append(i)
                dst.append(j)
    # g.from_networkx(G)
    g.add_nodes(num_nodes)
    g.add_edges(src,dst)
    print('Total Edges: {}'.format(g.number_of_edges()))
    norm = comp_deg_norm(g)
    node_id = torch.arange(0,num_nodes,dtype=torch.long).view(-1,1).to(device)
    norm = torch.from_numpy(norm).view(-1,1).to(device)
    g.ndata.update({'id':node_id,'norm':norm})
    # g.edata['weight'] = g.edata['weight'].to(device)
    return g


def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def get_seen_density(adj_matrix,seen_idxs,unseen_idxs,order):
    order_adj_matrix = [adj_matrix]
    for i in range(1,order):
        order_adj_matrix.append(order_adj_matrix[i-1]@adj_matrix)
    cumsum_adj_matrix = np.stack(order_adj_matrix,axis=0).sum(axis=0)
    num_of_nodes = adj_matrix.shape[0]
    seen_mask_vec = np.zeros((num_of_nodes))
    for i in range(num_of_nodes):
        if i in seen_idxs:
            seen_mask_vec[i] = 1
    seen_density = dict()
    for i in range(num_of_nodes):
        seen_density[i] = (sum(cumsum_adj_matrix[i] * seen_mask_vec))
    return seen_density
