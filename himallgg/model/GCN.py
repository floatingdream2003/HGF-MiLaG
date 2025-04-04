import torch.nn as nn
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, TransformerConv, GCN2Conv, GAT
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
import numpy as np

class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2 #这里是RGCN的8个关系
        # self.num_relations = 1  # 改成1，作为消融实验：没有了8个关系，只有一个关系的消融
        # self.num_relations = 4
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=10)
        self.conv2 = GraphConv(h1_dim, h2_dim)
        # self.conv2 = GATConv(h1_dim, h2_dim, concat=False, head=4)#新改的，跑这个，作为创新点

    def forward(self, node_features, edge_index, edge_type):
        # x = self.conv1(node_features, edge_index, edge_type, edge_norm=edge_norm)
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)

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

class GAT(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, heads, args):
        super(GAT, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=20)
        self.conv2 = GATConv(h1_dim, h2_dim, heads, dropout=0.6)
    def visualize_tsneTAV(self, features,
                       x_lim=(-102, 100), y_lim=(-102, 100), font_size=55,
                       legend_font_size=40, save_path='RGCNout for T+A+V Modals'):
        # 分割特征以初始化T, A, V模态的特征容器
        T_features, A_features, V_features = [], [], []
        for i in range(0, 800, 100):
            T_features.append(features[:, i:i + 33].cpu().numpy())
            A_features.append(features[:, i + 33:i + 66].cpu().numpy())
            V_features.append(features[:, i + 66:i + 99].cpu().numpy())
        T_features = np.concatenate(T_features, axis=1)
        A_features = np.concatenate(A_features, axis=1)
        V_features = np.concatenate(V_features, axis=1)

        color_T = '#FFDAB9'  # Peach Puff, 浅橙色
        color_A = '#90ee90'  # Light Green, 浅绿色
        color_V = '#ffcccc'  # Light Red, 浅红色

        tsne = TSNE(n_components=2, random_state=0)
        T_features_2d = tsne.fit_transform(T_features)
        A_features_2d = tsne.fit_transform(A_features)
        V_features_2d = tsne.fit_transform(V_features)
        total_points = T_features_2d.shape[0] + A_features_2d.shape[0] + V_features_2d.shape[0]

        plt.figure(figsize=(21, 18))
        plt.rc('font', size=font_size)

        # 绘制凸包并填充颜色的函数，与之前相同
        def plot_convex_hull(points, color):
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
            plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)

        plot_convex_hull(T_features_2d, color_T)
        plot_convex_hull(A_features_2d, color_A)
        plot_convex_hull(V_features_2d, color_V)

        plt.scatter(T_features_2d[:, 0], T_features_2d[:, 1], color='orange', label='T', marker='o')
        plt.scatter(A_features_2d[:, 0], A_features_2d[:, 1], color='green', label='A', marker='o')
        plt.scatter(V_features_2d[:, 0], V_features_2d[:, 1], color='red', label='V', marker='o')

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xticks([-100, -50, 0, 50, 100])
        plt.yticks([-100, -50, 0, 50, 100])

        plt.legend(loc='best', fontsize=legend_font_size)
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig("./graphout_vis/" + save_path, format='pdf', dpi=300)
        plt.show()
    def forward(self, node_features, edge_index, edge_type, return_attention_weights=False):
        # x = self.conv1(node_features, edge_index, edge_type)
        # # self.visualize_tsneTAV(x)
        # x = self.conv2(x, edge_index)
        # return x
        x = self.conv1(node_features, edge_index, edge_type)
        x, attn_weights = self.conv2(x, edge_index, return_attention_weights=True)
        if return_attention_weights:
            return x, attn_weights
        return x