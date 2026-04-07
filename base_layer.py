from torch_geometric.nn import HeteroConv, GraphConv, GATConv
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn


def xavier_init(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class LinearLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.clf(x)
        x = F.relu(x)
        return x


class GateSelect(torch.nn.Module):
    def __init__(self, in_dim, dropout):
        super().__init__()
        # 可能relu会破坏？
        self.att = torch.nn.Linear(in_dim, in_dim)
        self.dropout = dropout

    def forward(self, x):
        att_score = torch.sigmoid(self.att(x))
        feat_emb = torch.mul(att_score, x)
        # 先去掉看看效果
        # feat_emb = F.relu(feat_emb)
        feat_emb = F.dropout(feat_emb, self.dropout, training=self.training)

        return att_score, feat_emb


class HeteroGNN(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super(HeteroGNN, self).__init__()
        self.dropout = dropout

        # Define edge types for the heterogeneous graph
        self.edge_types = [
            ('x1', 'in', 'x1'),  # Intra-modal connections
            ('x2', 'in', 'x2'),
            ('x3', 'in', 'x3'),
            ('x1', 'between', 'x2'),  # Cross-modal connections
            ('x1', 'between', 'x3'),
            ('x2', 'between', 'x3'),
        ]
        # project
        self.x1_proj = nn.Linear(in_dim[0], in_dim[0])
        self.x2_proj = nn.Linear(in_dim[0], in_dim[0])
        self.x3_proj = nn.Linear(in_dim[0], in_dim[0])
        # Initialize graph convolutional layers
        self.conv1 = HeteroConv(self._create_conv_dict(-1, in_dim[0]))
        self.conv2 = HeteroConv(self._create_conv_dict(in_dim[0], in_dim[1]))

    def _create_conv_dict(self, in_dim, out_dim):
        """Create dictionary of graph convolution layers for each edge type."""
        return {edge_type: GraphConv(in_dim, out_dim) for edge_type in self.edge_types}

    def _process_features(self, x_dict):
        """Apply activation and dropout to node features."""
        return {
            key: F.dropout(
                F.leaky_relu(x),
                p=self.dropout,
                training=self.training
            )
            for key, x in x_dict.items()
        }

    def _get_edge_indices(self, data):
        """Extract edge indices from data object."""
        return {edge_type: data[edge_type].edge_index for edge_type in self.edge_types}

    def forward(self, data):
        # Extract initial node features
        # x_dict = {
        #     'x1': data['x1'].x,
        #     'x2': data['x2'].x,
        #     'x3': data['x3'].x
        # }
        x_dict = {
            'x1': self.x1_proj(data['x1'].x),
            'x2': self.x2_proj(data['x2'].x),
            'x3': self.x3_proj(data['x3'].x)
        }
        # Get edge indices
        edge_index_dict = self._get_edge_indices(data)

        # First convolutional layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self._process_features(x_dict)

        # Second convolutional layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = self._process_features(x_dict)

        # Extract features for fusion
        x1, x2, x3 = x_dict['x1'], x_dict['x2'], x_dict['x3']
        return x1, x2, x3


class EncoderGAT(torch.nn.Module):
    def __init__(self, feature_dim, in_dim, dropout):
        super(EncoderGAT, self).__init__()
        self.gat_layer1 = GATConv(feature_dim, in_dim[0], edge_dim=1)
        self.gat_layer2 = GATConv(in_dim[0], in_dim[1], edge_dim=1)
        self.gat_layer3 = GATConv(in_dim[1], in_dim[2], edge_dim=1)
        self.dropout = dropout

    def forward(self, x, adj, edge_weight=None):
        if edge_weight is not None:
            x = self.gat_layer1(x, adj, edge_attr=edge_weight)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.gat_layer2(x, adj, edge_attr=edge_weight)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.gat_layer3(x, adj, edge_attr=edge_weight)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = self.gat_layer1(x, adj)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.gat_layer2(x, adj)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.gat_layer3(x, adj)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x