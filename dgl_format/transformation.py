import torch
import copy
import networkx as nx
import random
import numpy as np
from torch_geometric.transforms import VirtualNode, AddLaplacianEigenvectorPE
from torch_geometric.utils import from_networkx, to_networkx, to_undirected
from torch_geometric.data import InMemoryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# VN

transform = VirtualNode()
def apply_vn(dgl_graphs):
  vn_EXP_dgl = []
  for graph in dgl_graphs:
    graph_pyg = from_dgl(graph)
    graph_pyg_copy = copy.deepcopy(graph_pyg)
    graph_vn = transform(graph_pyg_copy)
    graph_vn_dgl = to_dgl(graph_vn)
    vn_EXP_dgl.append(graph_vn_dgl)

  return vn_EXP_dgl

# Centrality
def add_centrality_to_node_features(dgl_graph, centrality_measure='degree'):
    # Convert DGL data to NetworkX graph
    G = dgl_graph.to_networkx()
    G = nx.Graph(G)

    # Compute the centrality measure
    if centrality_measure == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_measure == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_measure == 'eigenvector':
        if not nx.is_connected(G):
        # Handle connected components separately
            centrality = {}
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                sub_centrality = nx.eigenvector_centrality(subgraph, max_iter=500, tol=1e-4)
                centrality.update(sub_centrality)
        else:
            centrality = nx.eigenvector_centrality(G, max_iter=500, tol=1e-4)
    else:
        raise ValueError(f'Unknown centrality measure: {centrality_measure}')

    # Convert centrality to tensor and add as node feature
    centrality_values = np.array([centrality[node] for node in range(dgl_graph.number_of_nodes())], dtype=np.float32).reshape(-1, 1)
    centrality_values = torch.round(torch.tensor(centrality_values) * 10000) / 10000
    # Concatenate the centrality with existing node features
    if 'x' in dgl_graph.ndata:
        dgl_graph.ndata['x'] = torch.cat([dgl_graph.ndata['x'], centrality_values], dim=1)
    else:
        dgl_graph.ndata['x'] = centrality_values
    return dgl_graph

# Degree
def degree_dataset(dataset):
    # Compute centrality and add it as an additional feature
    Graph_data_degree = []
    for data in dataset:
        data_copy = copy.deepcopy(data)  # Create a deep copy of the graph
        data_copy = add_centrality_to_node_features(data_copy, centrality_measure='degree')
        Graph_data_degree.append(data_copy)
    return Graph_data_degree

# Closeness
def closeness_dataset(dataset):
    # Compute centrality and add it as an additional feature
    Graph_data_clo = []
    for data in dataset:
        data_copy = copy.deepcopy(data)
        data_copy = add_centrality_to_node_features(data_copy, centrality_measure='closeness')
        Graph_data_clo.append(data_copy)
    return Graph_data_clo

#Betweenness
def betweenness_dataset(dataset):
    # Compute centrality and add it as an additional feature
    Graph_data_bet = []
    for data in dataset:
        data_copy = copy.deepcopy(data)
        data_copy = add_centrality_to_node_features(data_copy, centrality_measure='betweenness')
        Graph_data_bet.append(data_copy)
    return Graph_data_bet

# Eigenvector
def eigenvector_dataset(dataset):
    # Compute centrality and add it as an additional feature
    Graph_data_eig = []
    for data in dataset:
        data_copy = copy.deepcopy(data)
        data_copy = add_centrality_to_node_features(data_copy, centrality_measure='eigenvector')
        Graph_data_eig.append(data_copy)
    return Graph_data_eig

# DE
def add_distance_encoding(dgl_graph):
    # Compute the shortest distance matrix using dgl.shortest_dist
    dist = dgl.shortest_dist(dgl_graph).float()  # Convert to float to handle inf

    # Replace -1 with inf (to handle unreachable nodes similar to NetworkX's behavior)
    dist[dist == -1] = float('inf')

    # Calculate the average shortest distance for each node
    finite_distances = torch.where(dist == float('inf'), torch.tensor(float('nan')), dist)
    average_distance = torch.nanmean(finite_distances, dim=1).view(-1, 1)  # Use nanmean to ignore infinities

    # Add the average distance to the existing node features in the DGL graph
    if 'x' in dgl_graph.ndata:
        dgl_graph.ndata['x'] = torch.cat([dgl_graph.ndata['x'], average_distance], dim=1)
    else:
        dgl_graph.ndata['x'] = average_distance

    return dgl_graph

def distance_encoding(dataset):
    Graph_data_DE = []
    for data in dataset:
        data_copy = copy.deepcopy(data)
        data_copy = add_distance_encoding(data_copy)
        Graph_data_DE.append(data_copy)
    return Graph_data_DE

# GE
from torch_geometric.transforms import AddLaplacianEigenvectorPE

def Graph_encoding(dgl_graphs):
    GE_EXP_dgl = []
    transform = AddLaplacianEigenvectorPE(k=5, attr_name = None)
    for graph in dgl_graphs:
        graph_pyg = from_dgl(graph)
        graph_pyg_copy = copy.deepcopy(graph_pyg)
        graph_GE = transform(graph_pyg_copy)
        graph_GE_dgl = to_dgl(graph_GE)
        GE_EXP_dgl.append(graph_GE_dgl)
    return GE_EXP_dgl

# Sub
def extract_local_subgraph_features(dgl_graph, radius=2):
    # Convert PyG data to NetworkX graph
    G = dgl_graph.to_networkx()
    G = nx.Graph(G)

    # Initialize a list to store subgraph features for each node
    subgraph_sizes = []
    subgraph_degrees = []

    for node in G.nodes():
        # Extract the ego graph (subgraph) around the node
        subgraph = nx.ego_graph(G, node, radius=radius)

        # Example feature 1: Size of the subgraph (number of nodes)
        subgraph_size = subgraph.number_of_nodes()
        subgraph_sizes.append(subgraph_size)

        # Example feature 2: Average degree of the subgraph
        subgraph_degree = np.mean([d for n, d in subgraph.degree()])
        subgraph_degrees.append(subgraph_degree)

    # Convert the features to tensors and add them as node features
    subgraph_sizes_tensor = torch.tensor(subgraph_sizes, dtype=torch.float).view(-1, 1)
    subgraph_degrees_tensor = torch.tensor(subgraph_degrees, dtype=torch.float).view(-1, 1)

    if 'x' in dgl_graph.ndata:
        dgl_graph.ndata['x'] = torch.cat([dgl_graph.ndata['x'], subgraph_sizes_tensor, subgraph_degrees_tensor], dim=1)
    else:
        dgl_graph.ndata['x'] = torch.cat([subgraph_sizes_tensor, subgraph_degrees_tensor], dim=1)

    return dgl_graph

def subgraph_dataset(dataset):
    # Compute centrality and add it as an additional feature
    Graph_data_sub = []
    for data in dataset:
        data_copy = copy.deepcopy(data)
        data_copy = extract_local_subgraph_features(data_copy, radius=2)
        Graph_data_sub.append(data_copy)
    return Graph_data_sub

# ExN
def add_extra_node_on_each_edge(dgl_graph):
    # Collect new edges (source, destination) and the new node features
    new_edges_src = []
    new_edges_dst = []
    new_node_features = []

    # Original number of nodes
    num_original_nodes = dgl_graph.num_nodes()

    # Use a set to track edges we have already processed (to avoid duplicates)
    processed_edges = set()

    # Iterate over all edges
    for i in range(dgl_graph.num_edges()):
        u, v = dgl_graph.edges()[0][i].item(), dgl_graph.edges()[1][i].item()

        # Avoid processing reverse edges (v, u) if (u, v) is already processed
        if (u, v) in processed_edges or (v, u) in processed_edges:
            continue

        # Mark the edge as processed
        processed_edges.add((u, v))
        processed_edges.add((v, u))  # In case there is a reverse edge

        # Add a new node
        new_node_id = num_original_nodes + len(new_node_features)
        mean_feature = (dgl_graph.ndata['x'][u] + dgl_graph.ndata['x'][v]) / 2
        new_node_features.append(mean_feature)

        # Add new edges connecting the new node to the original nodes
        new_edges_src.append(u)
        new_edges_dst.append(new_node_id)

        new_edges_src.append(new_node_id)
        new_edges_dst.append(v)

    # Add new nodes to the DGL graph
    dgl_graph.add_nodes(len(new_node_features), {'x': torch.stack(new_node_features)})

    # Remove the original edges
    dgl_graph.remove_edges(torch.arange(dgl_graph.num_edges()))

    # Add new edges to the DGL graph
    dgl_graph.add_edges(new_edges_src, new_edges_dst)

    return dgl_graph

def extra_node_dataset(dataset):
    Graph_data_exN = []
    for data in dataset:
        data_copy = copy.deepcopy(data)
        dgl_graph = add_extra_node_on_each_edge(data_copy)
        Graph_data_exN.append(dgl_graph)
    return Graph_data_exN
