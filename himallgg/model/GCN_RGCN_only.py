import torch.nn as nn
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, TransformerConv, GCN2Conv, GAT


class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2  # 这里是RGCN的8个关系
        # self.num_relations = 1  # 改成1，作为消融实验：没有了8个关系，只有一个关系的消融
        # self.num_relations = 4
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=10)
        self.conv2 = GraphConv(h1_dim, h2_dim)
        # self.conv2 = GATConv(h1_dim, h2_dim, concat=False, head=4)#新改的，跑这个，作为创新点

    def forward(self, node_features, edge_index, edge_type):
        # x = self.conv1(node_features, edge_index, edge_type, edge_norm=edge_norm)
        x = self.conv1(node_features, edge_index, edge_type)

        return x


class SGCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(SGCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = TransformerConv(g_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_type):
        # x = self.conv1(node_features, edge_index, edge_type, edge_norm=edge_norm)
        x = self.conv1(node_features, edge_index)
        return x