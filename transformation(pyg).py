import torch
import copy
import networkx as nx
import random
import numpy as np
from torch_geometric.transforms import VirtualNode, AddLaplacianEigenvectorPE
from torch_geometric.utils import from_networkx, to_networkx, to_undirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CustomQM9Dataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CustomQM9Dataset, self).__init__()
        self.data, self.slices = self.collate(data_list)
        


def apply_vn(pyg_dataset):
    vn_dataset = copy.deepcopy(pyg_dataset)
    transform = VirtualNode()
    vn_dataset.transform = transform
    return pyg_dataset

def add_centrality_to_node_features(data, centrality_measure='degree'):
    G = to_networkx(data, node_attrs=['x'], to_undirected=True)

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
    centrality_tensor = torch.tensor(centrality_values, dtype=torch.float).view(-1, 1)
    data.x = torch.cat([data.x, centrality_tensor], dim=-1)

    return data

def centrality(dataset, centrality_measure='degree'):
    original_dataset = copy.deepcopy(dataset)
    addCentrality_list = []
    for data in dataset:
        if centrality_measure == 'degree':
            data = add_centrality_to_node_features(data, centrality_measure='degree')
            addCentrality_list.append(data)
        elif centrality_measure == 'closeness':
            data = add_centrality_to_node_features(data, centrality_measure='closeness')
            addCentrality_list.append(data)
        elif centrality_measure == 'betweenness':
            data = add_centrality_to_node_features(data, centrality_measure='betweenness')
            addCentrality_list.append(data)
        elif centrality_measure == 'eigenvector':
            data = add_centrality_to_node_features(data, centrality_measure='igenvector')
            addCentrality_list.append(data)
        else:
            raise ValueError(f'Unknown centrality measure: {centrality_measure}')
        
    addCentrality_dataset = CustomQM9Dataset(addCentrality_list)

    return addCentrality_dataset

def distance_encoding_node_augmentation(data):
    G = to_networkx(data, node_attr=['x'], to_undirected = True)
    num_nodes = data.num_nodes

    # Initialize the distance matrix with infinity
    distance_matrix = [[float('inf')] * num_nodes for _ in range(num_nodes)]
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Populate the distance matrix with actual shortest path lengths
    for i in range(num_nodes):
        distance_matrix[i][i] = 0  # Distance to self is 0
        if i in shortest_paths:
            for j, d in shortest_paths[i].items():
                distance_matrix[i][j] = d

    # Convert the distance matrix to a tensor
    distance_tensor = torch.tensor(distance_matrix, dtype=torch.float)
    
    # Example: Add average distance to node features
    finite_distances = torch.where(distance_tensor == float('inf'), torch.tensor(float('nan')), distance_tensor)
    average_distance = torch.nanmean(finite_distances, dim=1).view(-1, 1)  # Use nanmean to ignore infinities
    data.x = torch.cat([data.x, average_distance], dim=1)
    
    return data

    


