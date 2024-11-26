import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, GINEConv, PNAConv, global_add_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, num_node_features, dim_h, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h
    


class GINENet(torch.nn.Module):
    def __init__(self, num_node_features, dim_h, edge_attr):
        super(GINENet, self).__init__()
        
        # Define GINE layers with the specified edge_dim
        self.conv1 = GINEConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_attr)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_attr)
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_attr)
        
        # Define linear layers for classification or regression
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Pass node features and edge attributes through GINE layers
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        # Apply global pooling for graph-level output
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate pooled features and pass through final linear layers
        h = torch.cat((h1, h2, h3), dim=1)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h



class PNANet(torch.nn.Module):
    def __init__(self, num_node_features, dim_h, edge_attr, aggregators, scalers, deg):
        super(PNANet, self).__init__()
        
        # Define PNA layers with specified aggregators, scalers, and degree tensor
        self.conv1 = PNAConv(
            in_channels=num_node_features,
            out_channels=dim_h,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_attr
        )
        self.conv2 = PNAConv(
            in_channels=dim_h,
            out_channels=dim_h,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_attr
        )
        self.conv3 = PNAConv(
            in_channels=dim_h,
            out_channels=dim_h,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_attr
        )
        
        # Define linear layers for final graph-level output
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Pass node features and edge attributes through PNA layers
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        # Apply global pooling for graph-level output
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate pooled features and pass through final linear layers
        h = torch.cat((h1, h2, h3), dim=1)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h
    

