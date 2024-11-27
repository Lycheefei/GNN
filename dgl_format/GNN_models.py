import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv, SumPooling, PNAConv
import networkx as nx
import random
import pickle
import numpy as np
import glob

# Define a GIN model
class GIN(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim, num_layers):
        super(GIN, self).__init__()
        self.in_feats = in_feats
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.layers.append(GINConv(
            nn.Sequential(
                nn.Linear(in_feats, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ), 'sum'))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ), 'sum'))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.layers.append(GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            ), 'sum'))
        self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Pooling layer
        self.pool = SumPooling()

    def forward(self, g, h):
        h = torch.round(h * 100) / 100
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            h = layer(g, h)
            h = batch_norm(h)
            h = F.relu(h)
        g_embedding = self.pool(g, h)
        return g_embedding

# Define a PNA model
class PNA(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim, num_layers, aggregators, scalers, deg):
        super(PNA, self).__init__()
        self.in_feats = in_feats
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(PNAConv(in_feats, hidden_dim, aggregators, scalers, deg))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(PNAConv(hidden_dim, hidden_dim, aggregators, scalers, deg))

        # Output layer
        self.layers.append(PNAConv(hidden_dim, out_dim, aggregators, scalers, deg))

        # Pooling layer
        self.pool = SumPooling()

    def forward(self, g, h):
        h = torch.round(h * 100) / 100
        for layer in self.layers:
            h = layer(g, h)
            h = F.relu(h)
        g_embedding = self.pool(g, h)
        return g_embedding

# Define a DeepSet model
class DeepSet(nn.Module):
    def __init__(self, input_dim):
        super(DeepSet, self).__init__()
        self.input_dim = input_dim  # Store input dimension
        # First neural network to map node features to node embeddings
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        # Second neural network to map summed node embeddings to final graph embedding
        self.rho = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output a single value
        )

    def forward(self, g, h):
        # Apply the first neural network to each node feature
        h = self.phi(h)
        h = torch.round(h * 100) / 100  # Round to reduce precision
        # Sum the node embeddings to create a graph embedding
        g_embedding = h.sum(dim=0, keepdim=True)
        # Apply the second neural network to the summed embedding
        output = self.rho(g_embedding)
        return output
