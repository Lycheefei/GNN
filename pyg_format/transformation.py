import torch
import copy
import networkx as nx
import random
import numpy as np
from torch_geometric.transforms import VirtualNode, AddLaplacianEigenvectorPE
from torch_geometric.utils import from_networkx, to_networkx, to_undirected
from torch_geometric.data import InMemoryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CustomQM9Dataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CustomQM9Dataset, self).__init__()
        self.data, self.slices = self.collate(data_list)
        

###Visual Node
def apply_vn(pyg_dataset):
    vn_dataset = copy.deepcopy(pyg_dataset)
    transform = VirtualNode()
    vn_dataset.transform = transform
    return pyg_dataset

###Centrality
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
    centrality_values = list(centrality.values())
    centrality_tensor = torch.tensor(centrality_values, dtype=torch.float).view(-1, 1)
    centrality_tensor = (centrality_tensor - centrality_tensor.mean()) / (centrality_tensor.std() + 1e-8)
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

###Distance Encoding
def distance_encoding_node_augmentation(data):
    G = to_networkx(data, node_attrs=['x'], to_undirected = True)
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

def distance_encoding_edge_rewiring(data):

    G = to_networkx(data, node_attrs=['x'], edge_attrs = ['edge_attr'], to_undirected=True)

    # Create a copy of the graph to avoid modifying the original
    G_transformed = G.copy()

    # Compute shortest path distances for all pairs of nodes
    connected_components = list(nx.connected_components(G))
    shortest_paths = {}

    # Compute shortest paths for each connected component
    for component in connected_components:
        subgraph = G.subgraph(component)
        component_paths = dict(nx.all_pairs_shortest_path_length(subgraph))
        shortest_paths.update(component_paths)

    # Get the list of all nodes
    nodes = list(G.nodes)

    # Add edges between all pairs of nodes
    for i in nodes:
        for j in nodes:
            if i != j:  # Avoid self-loops
                if G.has_edge(i, j):
                    # If the edge exists in the input graph, assign distance 1
                    G_transformed[i][j]["distance"] = 1
                else:
                    if j in shortest_paths[i]:
                        # Nodes are in the same connected component
                        distance = shortest_paths[i][j]
                    else:
                        # Nodes are in different connected components
                        distance = float('inf')  # Or assign a fixed large value

                    # Add the edge with the computed distance attribute
                    G_transformed.add_edge(i, j, distance=distance)

    data = from_networkx(G_transformed, group_node_attrs=['x'], group_edge_attrs=['edge_attr', 'distance'])
    return data

def distance_encoding(dataset, method = 'node_augmentation'):
    original_dataset = copy.deepcopy(dataset)
    distance_encoding_list = []
    for data in dataset:
        if method == 'node_augmentation':
            data = distance_encoding_node_augmentation(data)
            distance_encoding_list.append(data)
        elif method == 'edge_rewiring':
            data = distance_encoding_edge_rewiring(data)
            distance_encoding_list.append(data)
        else:
            raise ValueError(f'Unknown distance encoding method: {method}')
    distance_encoding_dataset = CustomQM9Dataset(distance_encoding_list)
    return distance_encoding_dataset

###Subgraph Extraction
def extract_local_subgraph_features(data, radius=2):
    # Convert PyG data to NetworkX graph
    G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)

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
    
    # Concatenate the new features to the existing node features
    data.x = torch.cat([data.x, subgraph_sizes_tensor, subgraph_degrees_tensor], dim=-1)
    
    return data

###Graph Encoding
def graph_encoding(dataset, k=2):
    GE_dataset = copy.deepcopy(dataset)
    transform = AddLaplacianEigenvectorPE(k=2, attr_name = None)
    GE_dataset.transform = transform
    return GE_dataset


###Add Extra Node on Each Edge
def add_extra_node_on_each_edge(data):
    # Convert PyG data to a NetworkX graph for easier manipulation
    G = to_networkx(data, node_attrs=['x'], edge_attrs = ['edge_attr'])
    
    # Original number of nodes
    num_original_nodes = G.number_of_nodes()
    
    # Prepare lists for new features
    edges = list(G.edges(data=True))
    new_node_features = []
    new_edges_src = []
    new_edges_dst = []
    new_edge_features = []

    for u, v, edge_data in edges:
        # Remove the original edge
        G.remove_edge(u, v)

        # Create new node as the mean of connected node features
        new_node_id = num_original_nodes + len(new_node_features)
        new_node_feature = (data.x[u] + data.x[v]) / 2
        new_node_features.append(new_node_feature)
        
        # Add new node with feature
        G.add_node(new_node_id, x=new_node_feature)

        # Add edges from new node to each original node
        G.add_edge(u, new_node_id)
        G.add_edge(new_node_id, v)

        # Use original edge feature for each new edge
        edge_feature = edge_data['edge_attr']
        edge_feature_tensor = (
            edge_feature if isinstance(edge_feature, torch.Tensor) else torch.tensor(edge_feature)
        )
        new_edge_features.append(edge_feature_tensor)  # for edge (u, new_node_id)
        new_edge_features.append(edge_feature_tensor)  # for edge (new_node_id, v)
    
    # Convert back to PyG Data object
    modified_data = from_networkx(G, group_node_attrs=['x'], group_edge_attrs=['edge_attr'])

    # Update node features
    modified_data.x = torch.cat([data.x, torch.stack(new_node_features)], dim=0)

    # Update edge features to include only the new edges
    modified_data.edge_attr = torch.stack(new_edge_features)  # Only include new edge features
    
    return modified_data

def extra_node(dataset):
    original_dataset = copy.deepcopy(dataset)
    extra_node_list = []
    for data in dataset:
        data = add_extra_node_on_each_edge(data)
        extra_node_list.append(data)
    extra_node_dataset = CustomQM9Dataset(extra_node_list)
    return extra_node_dataset

