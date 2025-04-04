# from .ContextualEncoder342 import ContextualEncoder342
# from .ContextualEncoder1024 import ContextualEncoder1024
# from .ContextualEncoder1582 import ContextualEncoder1582
from .ContextualEncoder import ContextualEncoder
from .EdgeAtt import EdgeAtt
from .GCN import GCN, SGCN, GAT
from .Classifier import Classifier
from .SoftHGRLoss import SoftHGRLoss
from .SampleWeightedFocalContrastiveLoss import SampleWeightedFocalContrastiveLoss
from .functions import batch_graphify, batch_graphify1
import himallgg
import torch.nn.functional as F
from .Fusion import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
import numpy as np
import os


class CrossModalAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y, mask=None):
        '''
        x: [batch_size, seq_len_x, d_model]
        y: [batch_size, seq_len_y, d_model]
        mask: [batch_size, seq_len_x, seq_len_y]
        '''
        q = self.q_layer(x)  # [batch_size, seq_len_x, d_model]
        k = self.k_layer(y)  # [batch_size, seq_len_y, d_model]
        v = self.v_layer(y)  # [batch_size, seq_len_y, d_model]

        # calculate attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, seq_len_x, seq_len_y]

        if mask is not None:
            attn_scores.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len_x, seq_len_y]

        # apply attention weights to values
        attn_output = torch.bmm(attn_weights, v)  # [batch_size, seq_len_x, d_model]

        # apply dropout and residual connection
        attn_output = self.dropout(attn_output)
        output = attn_output + x  # [batch_size, seq_len_x, d_model]

        return output

log = himallgg.utils.get_logger()

def visualize_tsneclass(self, graph_out, label_tensor, font_size=20,
                        legend_font_size=20, save_path='4.pdf'):
    # 获取特征和情绪标签
    emotion_labels = label_tensor.cpu().numpy()

    # 获取特征
    features = graph_out.cpu().numpy()

    # 将数据分割为六种情绪类别
    num_emotions = 6

    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    plt.rc('font', size=font_size)

    color_T = '#add8e6'  # 浅蓝色
    color_A = '#90ee90'  # 浅绿色
    color_V = '#ffcccc'  # 浅红色

    for emotion in range(num_emotions):
        ax = axes[emotion // 3, emotion % 3]

        # 提取当前情绪类别的特征
        emotion_indices = np.where(emotion_labels == emotion)[0]
        emotion_features = features[emotion_indices]

        # 分割特征以初始化T, A, V模态的特征容器
        T_features, A_features, V_features = [], [], []
        for i in range(0, 800, 100):
            T_features.append(emotion_features[:, i:i + 33])
            A_features.append(emotion_features[:, i + 33:i + 66])
            V_features.append(emotion_features[:, i + 66:i + 100])
        T_features = np.concatenate(T_features, axis=1)
        A_features = np.concatenate(A_features, axis=1)
        V_features = np.concatenate(V_features, axis=1)

        num_points = T_features.shape[0]
        print(f"Emotion {emotion + 1}: {num_points} points")

        tsne = TSNE(n_components=2, random_state=0)
        T_features_2d = tsne.fit_transform(T_features)
        A_features_2d = tsne.fit_transform(A_features)
        V_features_2d = tsne.fit_transform(V_features)

        # 绘制凸包并填充颜色的函数
        def plot_convex_hull(points, color, ax):
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
            ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)

        plot_convex_hull(T_features_2d, color_T, ax)
        plot_convex_hull(A_features_2d, color_A, ax)
        plot_convex_hull(V_features_2d, color_V, ax)

        ax.scatter(T_features_2d[:, 0], T_features_2d[:, 1], color='blue', label='T', marker='o')
        ax.scatter(A_features_2d[:, 0], A_features_2d[:, 1], color='green', label='A', marker='o')
        ax.scatter(V_features_2d[:, 0], V_features_2d[:, 1], color='red', label='V', marker='o')
        ax.set_title(f'Emotion {emotion + 1}', fontsize=font_size)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=legend_font_size, ncol=3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not save_path.endswith(".pdf"):
        save_path += ".pdf"
    plt.savefig("./graphout_vis/" + save_path, format='pdf', dpi=300)
    plt.show()

def tsne_visualizationgender(tensor, ax, color_labels=None, n_components=2, perplexity=30, n_iter=1000):
    tensor_np = tensor.cpu().numpy()
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=0)
    tsne_results = tsne.fit_transform(tensor_np)

    if color_labels is not None:
        cmap = ListedColormap(['red', 'blue'])
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=color_labels, s=5, cmap=cmap)

        # 绘制轮廓
        for cluster_value in np.unique(color_labels):
            cluster_points = tsne_results[color_labels == cluster_value]
            if cluster_points.shape[0] > 3:  # 确保有数据点存在
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', alpha=0.2,
                            color=cmap(cluster_value))
                # 使用浅色填充轮廓
                ax.fill(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1], color=cmap(cluster_value),
                        alpha=0.1)

        legend_labels = ['Male', 'Female']  # 替换为实际标签
        handles, _ = scatter.legend_elements()
        legend1 = ax.legend(handles=handles, labels=legend_labels, title="Gender")
        ax.add_artist(legend1)
    else:
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5)

    ax.set_xlabel('t-SNE component 1')
    ax.set_ylabel('t-SNE component 2')

class LGGCN(nn.Module):
    def __init__(self, args):
        super(LGGCN, self).__init__()
        uT_dim = 1024
        uA_dim = 1582
        uV_dim = 342
        g_dim = 160
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 6
        n_head = 8

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device
        self.rnn = ContextualEncoder(uT_dim, g_dim, args)
        self.rnn_A = ContextualEncoder(uA_dim, g_dim, args)
        self.rnn_V = ContextualEncoder(uV_dim, g_dim, args)
        self.cross_attention = CrossModalAttention(g_dim)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gat = GAT(g_dim*3, h1_dim, h2_dim, n_head, args)
        self.gcn = GCN(g_dim*3, h1_dim, h2_dim, args)
        self.edge_att_all = EdgeAtt(g_dim * 3, args)

        self.clf = Classifier(h2_dim+g_dim*3, hc_dim, tag_size, args)

        self.clf_T = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_A = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_V = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_FM = Classifier(g_dim, g_dim, 2, args)

        self.clf_gender = Classifier(h2_dim + g_dim * 3, hc_dim, 2, args)  # 2 表示性别分类

        self.clf_gT = Classifier(g_dim, hc_dim, 2, args)
        self.clf_gA = Classifier(g_dim, hc_dim, 2, args)
        self.clf_gV = Classifier(g_dim, hc_dim, 2, args)

        self.clf_gT0 = Classifier(480, hc_dim, 2, args)
        self.clf_gA0 = Classifier(480, hc_dim, 2, args)
        self.clf_gV0 = Classifier(480, hc_dim, 2, args)


        # self.clf_FM = Classifier(h2_dim*n_head+g_dim*3,h2_dim*n_head+g_dim*3, 2, args)
        self.label_to_idx = {'Hap': 0, 'Sad': 1, 'Neu': 2, 'Ang': 3, 'Exc': 4, 'Fru': 5}

        self.gcn1 = SGCN(g_dim , h1_dim, g_dim, args)
        self.gcn2 = SGCN(g_dim , h1_dim, g_dim, args)
        self.gcn3 = SGCN(g_dim , h1_dim, g_dim, args)
        self.att = MultiHeadedAttention(10, g_dim)
        self.args = args

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        self.edge_type_to_idx1 = {'00': 0, '01': 1, '10': 2, '11': 3}
        edge_type_to_idxs = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idxs[str(j) + str(k)] = len(edge_type_to_idxs)
        self.edge_type_to_idxs = edge_type_to_idxs
        self.edge_type_to_idxt = {'0': 0, '1': 1}
        self.edge_type_to_idx_ = {'0': 0}

    # def visualize_tsne_TAV(self, features_T, features_A, features_V,
    #                        x_lim=(-81, 80), y_lim=(-81, 80), font_size=55,
    #                        save_path='T+A+V Modal Feature'):
    #     tsne = TSNE(n_components=2, random_state=0)
    #     all_features = np.vstack((features_T, features_A, features_V))
    #     all_features_2d = tsne.fit_transform(all_features)
    #     num_T = features_T.shape[0]
    #     num_A = features_A.shape[0]
    #     features_2d_T = all_features_2d[:num_T, :]
    #     features_2d_A = all_features_2d[num_T:num_T + num_A, :]
    #     features_2d_V = all_features_2d[num_T + num_A:, :]
    #     color_T = '#F1AEA7'  # Peach Puff, 浅橙色
    #     color_A = '#9D9ECD'  # Light Green, 浅绿色
    #     color_V = '#E68D3D'  # Light Red, 浅红色
    #     plt.rc('font', size=font_size)
    #     plt.figure(figsize=(21, 18))
    #
    #     # 绘制凸包并填充颜色
    #     def plot_convex_hull(points, color):
    #         hull = ConvexHull(points)
    #         for simplex in hull.simplices:
    #             plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    #         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)
    #
    #     # 绘制T, A, V模态的点和凸包
    #     plot_convex_hull(features_2d_T, color_T)
    #     plot_convex_hull(features_2d_A, color_A)
    #     plot_convex_hull(features_2d_V, color_V)
    #
    #     plt.scatter(features_2d_T[:, 0], features_2d_T[:, 1], color='#5AB682', marker='o')
    #     plt.scatter(features_2d_A[:, 0], features_2d_A[:, 1], color='#E58579', marker='o')
    #     plt.scatter(features_2d_V[:, 0], features_2d_V[:, 1], color='#6270B7', marker='o')
    #
    #     if x_lim is not None:
    #         plt.xlim(x_lim)
    #     if y_lim is not None:
    #         plt.ylim(y_lim)
    #     plt.xticks([-80, -40, 0, 40, 80])
    #     plt.yticks([-80, -40, 0, 40, 80])
    #
    #     # 确保保存路径的文件夹存在
    #     output_dir = "./feature_vis/"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     # 保存为 JPG 和 PDF 格式
    #     plt.savefig(os.path.join(output_dir, save_path + ".jpg"), format='jpg', dpi=300)
    #     plt.savefig(os.path.join(output_dir, save_path + ".pdf"), format='pdf', dpi=300)
    #     plt.show()
    #
    # def visualize_tsneTAV(self, features,
    #                       x_lim=(-81, 80), y_lim=(-81, 80), font_size=45,
    #                       save_path='T+A+V'):
    #     # 分割特征以初始化T, A, V模态的特征容器
    #     T_features, A_features, V_features = [], [], []
    #     for i in range(0, 800, 100):
    #         # features = graph_out.detach().cpu().numpy()
    #         T_features.append(features[:, i:i + 33].detach().cpu().numpy())
    #         A_features.append(features[:, i + 33:i + 66].detach().cpu().numpy())
    #         V_features.append(features[:, i + 66:i + 99].detach().cpu().numpy())
    #     T_features = np.concatenate(T_features, axis=1)
    #     A_features = np.concatenate(A_features, axis=1)
    #     V_features = np.concatenate(V_features, axis=1)
    #
    #     color_T = '#F1AEA7'
    #     color_A = '#9D9ECD'
    #     color_V = '#E68D3D'
    #
    #     tsne = TSNE(n_components=2, random_state=0)
    #     T_features_2d = tsne.fit_transform(T_features)
    #     A_features_2d = tsne.fit_transform(A_features)
    #     V_features_2d = tsne.fit_transform(V_features)
    #
    #     plt.figure(figsize=(21, 18))
    #     plt.rc('font', size=font_size)
    #
    #     # 绘制凸包并填充颜色的函数，与之前相同
    #     def plot_convex_hull(points, color):
    #         hull = ConvexHull(points)
    #         for simplex in hull.simplices:
    #             plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    #         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)
    #
    #     plot_convex_hull(T_features_2d, color_T)
    #     plot_convex_hull(A_features_2d, color_A)
    #     plot_convex_hull(V_features_2d, color_V)
    #
    #     plt.scatter(T_features_2d[:, 0], T_features_2d[:, 1], color='#5AB682', marker='o')
    #     plt.scatter(A_features_2d[:, 0], A_features_2d[:, 1], color='#E58579', marker='o')
    #     plt.scatter(V_features_2d[:, 0], V_features_2d[:, 1], color='#6270B7', marker='o')
    #     plt.xlim(x_lim)
    #     plt.ylim(y_lim)
    #     plt.xticks([-80, -40, 0, 40, 80])
    #     plt.yticks([-80, -40, 0, 40, 80])
    #     import os
    #
    #     # 确保保存路径的文件夹存在
    #     output_dir = "./graphout_vis/"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     # 保存为 JPG 和 PDF 格式
    #     # plt.savefig(os.path.join(output_dir, save_path + ".jpg"), format='jpg', dpi=300)
    #     plt.savefig(os.path.join(output_dir, save_path + ".pdf"), format='pdf', dpi=300)
    #     # plt.show()
    # #
    # # def visualize_tsneTA(self, features, title='Graph network output for T+A Modal',
    # #                      x_lim=(-81, 80), y_lim=(-81, 80), font_size=55,
    # #                      save_path='Graph network output for T+A Modal'):
    # #     # Initialize T, A modal feature containers
    # #     T_features, A_features = [], []
    # #
    # #     # Divide by the second dimension into 4 parts, each part 100 dimensions
    # #     for i in range(0, 800, 100):
    # #         # Extract T, A modal features according to new requirements
    # #         T_features.append(features[:, i:i + 50].cpu().numpy())  # T is the first 50 dimensions
    # #         A_features.append(features[:, i + 50:i + 100].cpu().numpy())  # A is 50-100 dimensions
    # #
    # #     # Merge data along the feature dimension
    # #     T_features = np.concatenate(T_features, axis=1)
    # #     A_features = np.concatenate(A_features, axis=1)
    # #
    # #     # Initialize t-SNE
    # #     tsne = TSNE(n_components=2, random_state=0)
    # #
    # #     # Perform t-SNE dimensionality reduction on each modality's features
    # #     T_features_2d = tsne.fit_transform(T_features)
    # #     A_features_2d = tsne.fit_transform(A_features)
    # #
    # #     plt.figure(figsize=(21, 18))
    # #     plt.rc('font', size=font_size)
    # #
    # #     # Plotting function for convex hull and fill color, same as before
    # #     def plot_convex_hull(points, color):
    # #         hull = ConvexHull(points)
    # #         for simplex in hull.simplices:
    # #             plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    # #         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)
    # #
    # #     color_T = '#F1AEA7'
    # #     color_A = '#9D9ECD'
    # #
    # #     plot_convex_hull(T_features_2d, color_T)
    # #     plot_convex_hull(A_features_2d, color_A)
    # #
    # #     plt.scatter(T_features_2d[:, 0], T_features_2d[:, 1], color='#5AB682', marker='o')
    # #     plt.scatter(A_features_2d[:, 0], A_features_2d[:, 1], color='#E58579', marker='o')
    # #
    # #     plt.xlim(x_lim)
    # #     plt.ylim(y_lim)
    # #     plt.xticks([-80, -40, 0, 40, 80])
    # #     plt.yticks([-80, -40, 0, 40, 80])
    # #
    # #     # 确保保存路径的文件夹存在
    # #     output_dir = "./graphout_vis/"
    # #     if not os.path.exists(output_dir):
    # #         os.makedirs(output_dir)
    # #
    # #     # 保存为 JPG 和 PDF 格式
    # #     plt.savefig(os.path.join(output_dir, save_path + ".jpg"), format='jpg', dpi=300)
    # #     plt.savefig(os.path.join(output_dir, save_path + ".pdf"), format='pdf', dpi=300)
    # #     plt.show()
    # #
    # # def visualize_tsneTV(self, features, title='Graph network output for T+V Modal',
    # #                      x_lim=(-81, 80), y_lim=(-81, 80), font_size=55,
    # #                      save_path='Graph network output for T+V Modal'):
    # #     # Initialize T, V modal feature containers
    # #     T_features, V_features = [], []
    # #
    # #     for i in range(0, 800, 100):
    # #         # Extract T, V modal features
    # #         T_features.append(features[:, i:i + 50].cpu().numpy())
    # #         V_features.append(features[:, i + 50:i + 100].cpu().numpy())
    # #
    # #     T_features = np.concatenate(T_features, axis=1)
    # #     V_features = np.concatenate(V_features, axis=1)
    # #
    # #     tsne = TSNE(n_components=2, random_state=0)
    # #     T_features_2d = tsne.fit_transform(T_features)
    # #     V_features_2d = tsne.fit_transform(V_features)
    # #
    # #     plt.figure(figsize=(21, 18))
    # #     plt.rc('font', size=font_size)
    # #
    # #     color_T = '#F1AEA7'
    # #     color_V = '#E68D3D'
    # #
    # #     def plot_convex_hull(points, color):
    # #         hull = ConvexHull(points)
    # #         for simplex in hull.simplices:
    # #             plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    # #         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)
    # #
    # #     plot_convex_hull(T_features_2d, color_T)
    # #     plot_convex_hull(V_features_2d, color_V)
    # #
    # #     plt.scatter(T_features_2d[:, 0], T_features_2d[:, 1], color='#5AB682', marker='o')
    # #     plt.scatter(V_features_2d[:, 0], V_features_2d[:, 1], color='#6270B7', marker='o')
    # #
    # #     plt.xlim(x_lim)
    # #     plt.ylim(y_lim)
    # #     plt.xticks([-80, -40, 0, 40, 80])
    # #     plt.yticks([-80, -40, 0, 40, 80])
    # #
    # #     # 确保保存路径的文件夹存在
    # #     output_dir = "./graphout_vis/"
    # #     if not os.path.exists(output_dir):
    # #         os.makedirs(output_dir)
    # #
    # #     # 保存为 JPG 和 PDF 格式
    # #     plt.savefig(os.path.join(output_dir, save_path + ".jpg"), format='jpg', dpi=300)
    # #     plt.savefig(os.path.join(output_dir, save_path + ".pdf"), format='pdf', dpi=300)
    # #     plt.show()
    # #
    # # def visualize_tsneAV(self, features, title='Graph network output for A+V Modal',
    # #                      x_lim=(-81, 80), y_lim=(-81, 80), font_size=55,
    # #                      save_path='Graph network output for A+V Modal'):
    # #     # Initialize A, V modal feature containers
    # #     A_features, V_features = [], []
    # #
    # #     for i in range(0, 800, 100):
    # #         # Extract A, V modal features
    # #         A_features.append(features[:, i:i + 50].cpu().numpy())
    # #         V_features.append(features[:, i + 50:i + 100].cpu().numpy())
    # #
    # #     A_features = np.concatenate(A_features, axis=1)
    # #     V_features = np.concatenate(V_features, axis=1)
    # #
    # #     tsne = TSNE(n_components=2, random_state=0)
    # #     A_features_2d = tsne.fit_transform(A_features)
    # #     V_features_2d = tsne.fit_transform(V_features)
    # #
    # #     plt.figure(figsize=(21, 18))
    # #     plt.rc('font', size=font_size)
    # #
    # #     color_A = '#9D9ECD'
    # #     color_V = '#E68D3D'
    # #
    # #     def plot_convex_hull(points, color):
    # #         hull = ConvexHull(points)
    # #         for simplex in hull.simplices:
    # #             plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    # #         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)
    # #
    # #     plot_convex_hull(A_features_2d, color_A)
    # #     plot_convex_hull(V_features_2d, color_V)
    # #
    # #     plt.scatter(A_features_2d[:, 0], A_features_2d[:, 1], color='#E58579', marker='o')
    # #     plt.scatter(V_features_2d[:, 0], V_features_2d[:, 1], color='#6270B7', marker='o')
    # #
    # #     plt.xlim(x_lim)
    # #     plt.ylim(y_lim)
    # #     plt.xticks([-80, -40, 0, 40, 80])
    # #     plt.yticks([-80, -40, 0, 40, 80])
    # #
    # #     # 确保保存路径的文件夹存在
    # #     output_dir = "./graphout_vis/"
    # #     if not os.path.exists(output_dir):
    # #         os.makedirs(output_dir)
    # #
    # #     # 保存为 JPG 和 PDF 格式
    # #     plt.savefig(os.path.join(output_dir, save_path + ".jpg"), format='jpg', dpi=300)
    # #     plt.savefig(os.path.join(output_dir, save_path + ".pdf"), format='pdf', dpi=300)
    # #     plt.show()
    # #
    # # def visualize_tsne_TA(self, features_T, features_A, title='T+A Modal Feature',
    # #                       x_lim=(-81, 80), y_lim=(-81, 80), font_size=55,
    # #                       save_path='T+A Modal Feature'):
    # #     tsne = TSNE(n_components=2, random_state=0)
    # #     all_features = np.vstack((features_T, features_A))
    # #     all_features_2d = tsne.fit_transform(all_features)
    # #     num_T = features_T.shape[0]
    # #     features_2d_T = all_features_2d[:num_T, :]
    # #     features_2d_A = all_features_2d[num_T:, :]
    # #
    # #     color_T = '#F1AEA7'
    # #     color_A = '#9D9ECD'
    # #
    # #     def plot_convex_hull(points, color):
    # #         hull = ConvexHull(points)
    # #         for simplex in hull.simplices:
    # #             plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    # #         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)
    # #
    # #     plt.rc('font', size=font_size)
    # #     plt.figure(figsize=(21, 18))
    # #     plot_convex_hull(features_2d_T, color_T)
    # #     plot_convex_hull(features_2d_A, color_A)
    # #     plt.scatter(features_2d_T[:, 0], features_2d_T[:, 1], color='#5AB682', marker='o', alpha=0.5)
    # #     plt.scatter(features_2d_A[:, 0], features_2d_A[:, 1], color='#E58579', marker='o', alpha=0.5)
    # #
    # #     plt.xlim(x_lim)
    # #     plt.ylim(y_lim)
    # #     plt.xticks([-80, -40, 0, 40, 80])
    # #     plt.yticks([-80, -40, 0, 40, 80])
    # #
    # #     # 确保保存路径的文件夹存在
    # #     output_dir = "./feature_vis/"
    # #     if not os.path.exists(output_dir):
    # #         os.makedirs(output_dir)
    # #
    # #     # 保存为 JPG 和 PDF 格式
    # #     plt.savefig(os.path.join(output_dir, save_path + ".jpg"), format='jpg', dpi=300)
    # #     plt.savefig(os.path.join(output_dir, save_path + ".pdf"), format='pdf', dpi=300)
    # #     plt.show()
    # #
    # # def visualize_tsne_TV(self, features_T, features_V, title='T+V Modal Feature',
    # #                       x_lim=(-81, 80), y_lim=(-81, 80), font_size=55,
    # #                       save_path='T+V Modal Feature'):
    # #     tsne = TSNE(n_components=2, random_state=0)
    # #     all_features = np.vstack((features_T, features_V))
    # #     all_features_2d = tsne.fit_transform(all_features)
    # #     num_T = features_T.shape[0]
    # #     features_2d_T = all_features_2d[:num_T, :]
    # #     features_2d_V = all_features_2d[num_T:, :]
    # #
    # #     color_T = '#F1AEA7'
    # #     color_V = '#E68D3D'
    # #
    # #     def plot_convex_hull(points, color):
    # #         hull = ConvexHull(points)
    # #         for simplex in hull.simplices:
    # #             plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    # #         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)
    # #
    # #     plt.rc('font', size=font_size)
    # #     plt.figure(figsize=(21, 18))
    # #     plot_convex_hull(features_2d_T, color_T)
    # #     plot_convex_hull(features_2d_V, color_V)
    # #     plt.scatter(features_2d_T[:, 0], features_2d_T[:, 1], color='#5AB682', marker='o', alpha=0.5)
    # #     plt.scatter(features_2d_V[:, 0], features_2d_V[:, 1], color='#6270B7', marker='o', alpha=0.5)
    # #
    # #     plt.xlim(x_lim)
    # #     plt.ylim(y_lim)
    # #     plt.xticks([-80, -40, 0, 40, 80])
    # #     plt.yticks([-80, -40, 0, 40, 80])
    # #
    # #     # 确保保存路径的文件夹存在
    # #     output_dir = "./feature_vis/"
    # #     if not os.path.exists(output_dir):
    # #         os.makedirs(output_dir)
    # #
    # #     # 保存为 JPG 和 PDF 格式
    # #     plt.savefig(os.path.join(output_dir, save_path + ".jpg"), format='jpg', dpi=300)
    # #     plt.savefig(os.path.join(output_dir, save_path + ".pdf"), format='pdf', dpi=300)
    # #     plt.show()
    # #
    # # def visualize_tsne_AV(self, features_A, features_V, title='A+V Modal Feature',
    # #                       x_lim=(-81, 80), y_lim=(-81, 80), font_size=55,
    # #                       save_path='A+V Modal Feature'):
    # #     tsne = TSNE(n_components=2, random_state=0)
    # #     all_features = np.vstack((features_A, features_V))
    # #     all_features_2d = tsne.fit_transform(all_features)
    # #     num_A = features_A.shape[0]
    # #     features_2d_A = all_features_2d[:num_A, :]
    # #     features_2d_V = all_features_2d[num_A:, :]
    # #
    # #     color_A = '#9D9ECD'
    # #     color_V = '#E68D3D'
    # #
    # #     def plot_convex_hull(points, color):
    # #         hull = ConvexHull(points)
    # #         for simplex in hull.simplices:
    # #             plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    # #         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)
    # #
    # #     plt.rc('font', size=font_size)
    # #     plt.figure(figsize=(21, 18))
    # #     plot_convex_hull(features_2d_A, color_A)
    # #     plot_convex_hull(features_2d_V, color_V)
    # #     plt.scatter(features_2d_A[:, 0], features_2d_A[:, 1], color='#E58579', marker='o', alpha=0.5)
    # #     plt.scatter(features_2d_V[:, 0], features_2d_V[:, 1], color='#6270B7', marker='o', alpha=0.5)
    # #
    # #     plt.xlim(x_lim)
    # #     plt.ylim(y_lim)
    # #     plt.xticks([-80, -40, 0, 40, 80])
    # #     plt.yticks([-80, -40, 0, 40, 80])
    # #
    # #     # 确保保存路径的文件夹存在
    # #     output_dir = "./feature_vis/"
    # #     if not os.path.exists(output_dir):
    # #         os.makedirs(output_dir)
    # #
    # #     # 保存为 JPG 和 PDF 格式
    # #     plt.savefig(os.path.join(output_dir, save_path + ".jpg"), format='jpg', dpi=300)
    # #     plt.savefig(os.path.join(output_dir, save_path + ".pdf"), format='pdf', dpi=300)
    # #     plt.show()
    # #
    # # def visualize_tsnegenderall(self, features_T, features_A, features_V, labels, filename="tsne_visualization.jpg", dpi=300):
    # #     def plot_tsne(features, labels, ax, title, color_dict):
    # #         tsne = TSNE(n_components=2, random_state=42)
    # #         tsne_results = tsne.fit_transform(features)
    # #         cmap = ListedColormap([color_dict[0], color_dict[1]])
    # #         scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap, alpha=0.7)
    # #         handles = [
    # #             plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[0], markersize=10, label='Male'),
    # #             plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[1], markersize=10, label='Female')]
    # #         ax.legend(handles=handles, title="Gender")
    # #         ax.set_title(title)
    # #         ax.set_xticks([])  # 去除x轴刻度
    # #         ax.set_yticks([])  # 去除y轴刻度
    # #     #     for gender in np.unique(labels):
    # #     #         gender_mask = labels == gender
    # #     #         points = tsne_results[gender_mask]
    # #     #         color = color_dict[gender]
    # #     #         plot_convex_hull(points, color, ax)
    # #     # def plot_convex_hull(points, color, ax):
    # #     #     if len(points) < 3:
    # #     #         return  # Convex hull cannot be computed with fewer than 3 points
    # #     #     hull = ConvexHull(points)
    # #     #     for simplex in hull.simplices:
    # #     #         ax.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
    # #     #     ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.3)
    # #
    # #     color_dict = {0: '#9D9ECD', 1: '#E68D3D'}  # Define colors for genders
    # #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # #     plot_tsne(features_T.detach().cpu().numpy(), labels.detach().cpu().numpy(), axs[0], 'T modal', color_dict)
    # #     plot_tsne(features_A.detach().cpu().numpy(), labels.detach().cpu().numpy(), axs[1], 'A modal', color_dict)
    # #     plot_tsne(features_V.detach().cpu().numpy(), labels.detach().cpu().numpy(), axs[2], 'V modal', color_dict)
    # #     plt.tight_layout()
    # #     plt.savefig(filename, dpi=dpi)  # 以 JPG 格式保存图像，并设置 DPI
    # #     plt.show()
    # #
    # # def visualize_tsnegender(self, features_T, features_A, features_V, labels, folder="nogenderticks",
    # #                          filename="tsne_visualization", dpi=300):
    # #     def plot_tsne(features, labels, ax, color_dict):
    # #         tsne = TSNE(n_components=2, random_state=42)
    # #         tsne_results = tsne.fit_transform(features)
    # #         cmap = ListedColormap([color_dict[0], color_dict[1]])
    # #         scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap, alpha=0.7)
    # #         # ax.set_xticks([])  # 去除x轴刻度
    # #         # ax.set_yticks([])  # 去除y轴刻度
    # #         ax.set_xticks(range(-60, 61, 20))  # 设置x轴刻度
    # #         ax.set_yticks(range(-50, 51, 20))  # 设置y轴刻度
    # #         ax.set_xlim(-60, 60)  # 设置x轴范围
    # #         ax.set_ylim(-50, 50)  # 设置y轴范围
    # #         ax.tick_params(axis='both', which='major', labelsize=23)  # 设置刻度大小
    # #
    # #     import os
    # #     if not os.path.exists(folder):
    # #         os.makedirs(folder)
    # #
    # #     color_dict = {0: '#9D9ECD', 1: '#E68D3D'}  # Define colors for genders
    # #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # #     plot_tsne(features_T.detach().cpu().numpy(), labels.detach().cpu().numpy(), axs[0], color_dict)
    # #     plot_tsne(features_A.detach().cpu().numpy(), labels.detach().cpu().numpy(), axs[1], color_dict)
    # #     plot_tsne(features_V.detach().cpu().numpy(), labels.detach().cpu().numpy(), axs[2], color_dict)
    # #     plt.tight_layout()
    # #
    # #     # 保存为 JPG 和 PDF 格式
    # #     plt.savefig(os.path.join(folder, f"{filename}.jpg"), dpi=dpi)
    # #     plt.savefig(os.path.join(folder, f"{filename}.pdf"), dpi=dpi)
    # #     plt.show()

    def visualize_tsneclass1(self, graph_out, label_tensor, font_size=20,
                            legend_font_size=20, save_path='2.pdf'):
        # 获取特征和情绪标签
        emotion_labels = label_tensor.cpu().numpy()

        # 获取特征
        # features = graph_out.cpu().numpy()
        features = graph_out.detach().cpu().numpy()


        plt.figure(figsize=(12, 8))
        plt.rc('font', size=font_size)

        # 定义每种情绪的较深颜色
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']  # 深蓝、深绿、深红、深紫、深棕、深粉


        for label, emotion in self.label_to_idx.items():
            # 提取当前情绪类别的特征
            emotion_indices = np.where(emotion_labels == emotion)[0]
            emotion_features0 = features[emotion_indices]
            emotion_features = features[emotion_indices]

            num_points = emotion_features.shape[0]
            # print(f"{label}: {num_points} points")


            tsne = TSNE(n_components=2, random_state=0)

            emotion_features_2d = tsne.fit_transform(emotion_features)

            # 绘制当前情绪类别的散点图
            plt.scatter(emotion_features_2d[:, 0], emotion_features_2d[:, 1], color=colors[emotion], label=label,
                        marker='o')
        #
        # plt.xlim(-40, 40)
        # plt.ylim(-40, 40)

        # 隐藏坐标刻度线、坐标数字，但保留边框（黑色矩形框）
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
        ax.set_xticklabels([])  # 隐藏x轴坐标数字
        ax.set_yticklabels([])  # 隐藏y轴坐标数字

        # 将图例放置在图框外面
        # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5), fontsize=legend_font_size)
        # plt.title('t-SNE Visualization of Emotions', fontsize=

        # plt.text(0.5, -3, '(a) Early', fontsize=font_size, ha='center', va='center')
        # plt.title('(a) Early', fontsize=font_size, loc='center',pad=10)
        # plt.title('(b) Middle', fontsize=font_size, loc='lower center',pad=10)
        # plt.title('(c) Late', fontsize=font_size, loc='lower center',pad=10)
        # plt.title('(d) Mid-Late', fontsize=font_size, loc='lower center',pad=10)

        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        # plt.savefig("./graphout_vis/" + save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.show()

#有凸包
    def visualize_tsneclass0(self, graph_out, label_tensor, font_size=20,
                             legend_font_size=20, save_path='4.pdf'):
        # 获取特征和情绪标签
        emotion_labels = label_tensor.cpu().numpy()

        # 获取特征
        # features = graph_out.cpu().numpy()
        features = graph_out.detach().cpu().numpy()

        plt.figure(figsize=(12, 8))
        plt.rc('font', size=font_size)

        # 定义每种情绪的较深颜色
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']  # 深蓝、深绿、深红、深紫、深棕、深粉

        for label, emotion in self.label_to_idx.items():
            # 提取当前情绪类别的特征
            emotion_indices = np.where(emotion_labels == emotion)[0]
            emotion_features0 = features[emotion_indices]
            emotion_features = features[emotion_indices]

            num_points = emotion_features.shape[0]
            # print(f"{label}: {num_points} points")

            tsne = TSNE(n_components=2, random_state=0)

            emotion_features_2d = tsne.fit_transform(emotion_features)

            # 绘制当前情绪类别的凸包并填充颜色
            def plot_convex_hull(points, color):
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
                # plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)

            plot_convex_hull(emotion_features_2d, colors[emotion])

            # 绘制当前情绪类别的散点图
            plt.scatter(emotion_features_2d[:, 0], emotion_features_2d[:, 1], color=colors[emotion], label=label,
                        marker='o')

        # plt.xlim(-40, 40)
        # plt.ylim(-40, 40)

        # 隐藏坐标刻度线、坐标数字，但保留边框（黑色矩形框）
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
        ax.set_xticklabels([])  # 隐藏x轴坐标数字
        ax.set_yticklabels([])  # 隐藏y轴坐标数字

        # 将图例放置在图框外面
        # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5), fontsize=legend_font_size)

        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig("./graphout_vis/" + save_path, format='pdf', dpi=300, bbox_inches='tight')
        # plt.show()

#集聚
    def visualize_tsneclass_gather(self, graph_out, label_tensor, font_size=20,
                             legend_font_size=20, save_path='4.pdf'):
        # 获取特征和情绪标签
        emotion_labels = label_tensor.cpu().numpy()

        # 获取特征
        features = graph_out.detach().cpu().numpy()

        plt.figure(figsize=(12, 8))
        plt.rc('font', size=font_size)

        # 定义每种情绪的较深颜色
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']  # 深蓝、深绿、深红、深紫、深棕、深粉

        offset_factor = 0.1  # 控制偏移程度的比例因子，可根据需要调整

        for label, emotion in self.label_to_idx.items():
            # 提取当前情绪类别的特征
            emotion_indices = np.where(emotion_labels == emotion)[0]
            emotion_features = features[emotion_indices]

            num_points = emotion_features.shape[0]

            tsne = TSNE(n_components=2, random_state=0)
            emotion_features_2d = tsne.fit_transform(emotion_features)

            # 计算当前情绪类别点的中心坐标（均值坐标）
            center_x = np.mean(emotion_features_2d[:, 0])
            center_y = np.mean(emotion_features_2d[:, 1])
            center = np.array([center_x, center_y])

            # 让各点朝着中心坐标偏移一定距离
            offset_emotion_features_2d = []
            for point in emotion_features_2d:
                offset_point = point + offset_factor * (center - point)
                offset_emotion_features_2d.append(offset_point)
            offset_emotion_features_2d = np.array(offset_emotion_features_2d)

            # 绘制当前情绪类别的凸包并填充颜色
            def plot_convex_hull(points, color):
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
                plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.5)

            plot_convex_hull(offset_emotion_features_2d, colors[emotion])

            # 绘制当前情绪类别的散点图
            plt.scatter(offset_emotion_features_2d[:, 0], offset_emotion_features_2d[:, 1], color=colors[emotion],
                        label=label,
                        marker='o')

        # 隐藏坐标刻度线、坐标数字，但保留边框（黑色矩形框）
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
        ax.set_xticklabels([])  # 隐藏x轴坐标数字
        ax.set_yticklabels([])  # 隐藏y轴坐标数字

        # 将图例放置在图框外面
        # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5), fontsize=legend_font_size)

        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig("./graphout_vis/" + save_path, format='pdf', dpi=300, bbox_inches='tight')
        # plt.show()

    def get_rep(self, data):
        node_features_T = self.rnn(data["text_len_tensor"], data["text_tensor"])  # [batch_size, mx_len, D_g]
        node_features_A = self.rnn_A(data["text_len_tensor"], data["audio_tensor"])  # [batch_size, mx_len, D_g]
        node_features_V = self.rnn_V(data["text_len_tensor"], data["visual_tensor"])  # [batch_size, mx_len, D_g]

        node_features = torch.cat((node_features_T, node_features_A, node_features_V), 2)

        features_T, edge_index_T, edge_type_T, edge_index_lengths_T = batch_graphify1(
            node_features_T, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx1, self.edge_att, self.device)

        features_A, edge_index_A, edge_type_A, edge_index_lengths_A = batch_graphify1(
            node_features_A, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx1, self.edge_att, self.device)

        features_V, edge_index_V, edge_type_V, edge_index_lengths = batch_graphify1(
            node_features_V, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx1, self.edge_att, self.device)

        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att_all, self.device)

        ##################################################################################################################
        # 局部图
        graph_out_T = self.gcn1(features_T, edge_index_T, edge_type_T)
        graph_out_A = self.gcn2(features_A, edge_index_A, edge_type_A)
        graph_out_V = self.gcn3(features_V, edge_index_V, edge_type_V)

        fea1 = self.att(graph_out_T, graph_out_A, graph_out_A)
        fea2 = self.att(graph_out_T, graph_out_V, graph_out_V)

        # 全局图
        features_graph = torch.cat([graph_out_T, fea1.squeeze(1), fea2.squeeze(1)], dim=-1)
        graph_out = self.gcn(features_graph, edge_index, edge_type)
        ##################################################################################################################

        # self.visualize_tsneclass1(features, data["label_tensor"])
        #
        # import matplotlib.pyplot as plt
        # import networkx as nx
        #
        # def visualize_graph_with_edge_weights(features, edge_index, edge_type, attn_weights,
        #                                       node_size=300, font_size=9, width=2, title_font_size=12):
        #     # 创建2x4的子图布局
        #     fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        #     axes = axes.flatten()
        #     num_nodes_to_visualize = 20
        #     # 将数据转换为CPU格式，以便处理
        #     edge_index = edge_index.cpu().numpy() if edge_index.is_cuda else edge_index.numpy()
        #     edge_type = edge_type.cpu().numpy() if edge_type.is_cuda else edge_type.numpy()
        #     attn_weights = attn_weights[1]  # 提取第二个元素，即注意力权重
        #     attn_weights = attn_weights.mean(dim=1)  # 计算每条边的平均注意力权重
        #     # attn_weights = attn_weights.cpu().numpy() if attn_weights.is_cuda else attn_weights.numpy()#旧
        #
        #     attn_weights = attn_weights.detach().cpu().numpy() if attn_weights.is_cuda else attn_weights.detach().numpy()#改
        #
        #
        #     # 定义每个子图的标题，使用 LaTeX 语法
        #     titles = [
        #         r"$e(\mathrm{s1}, \mathrm{s1})_{\mathrm{forward}}$",
        #         r"$e(\mathrm{s1}, \mathrm{s1})_{\mathrm{backward}}$",
        #         r"$e(\mathrm{s1}, \mathrm{s2})_{\mathrm{forward}}$",
        #         r"$e(\mathrm{s1}, \mathrm{s2})_{\mathrm{backward}}$",
        #         r"$e(\mathrm{s2}, \mathrm{s1})_{\mathrm{forward}}$",
        #         r"$e(\mathrm{s2}, \mathrm{s1})_{\mathrm{backward}}$",
        #         r"$e(\mathrm{s2}, \mathrm{s2})_{\mathrm{forward}}$",
        #         r"$e(\mathrm{s2}, \mathrm{s2})_{\mathrm{backward}}$"
        #     ]
        #
        #     # 定义统一的颜色（例如蓝色）
        #     base_color = (0, 0, 1)  # RGB for blue
        #
        #     # 为每种边类型绘制一个子图
        #     for idx, etype in enumerate(range(8)):  # 处理8种边类型
        #         ax = axes[idx]
        #         G = nx.DiGraph()  # 使用有向图以区分方向
        #
        #         # 筛选并添加对应类型和方向的边
        #         relevant_nodes = set()
        #         for i, (src, dst) in enumerate(edge_index.T):
        #             if src < num_nodes_to_visualize and dst < num_nodes_to_visualize and edge_type[i] == etype:
        #                 eweight = attn_weights[i]  # 获取边的权重
        #                 G.add_edge(int(src), int(dst), weight=eweight)
        #                 relevant_nodes.add(src)
        #                 relevant_nodes.add(dst)
        #
        #         # 只添加实际用到的节点
        #         for node in relevant_nodes:
        #             G.add_node(node)
        #
        #         # 获取边权重并映射到颜色深浅（透明度），同时归一化权重
        #         edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        #         edge_colors = []
        #         normalized_weights = []
        #         if edge_weights:  # 防止空图时报错
        #             min_weight, max_weight = min(edge_weights), max(edge_weights)
        #             for w in edge_weights:
        #                 normalized_weight = (w - min_weight) / (
        #                         max_weight - min_weight) if max_weight > min_weight else 1
        #                 alpha = normalized_weight  # 归一化后的权重用于调整透明度
        #                 rgba_color = base_color + (alpha,)  # 将颜色和alpha组合为RGBA
        #                 edge_colors.append(rgba_color)
        #                 normalized_weights.append(normalized_weight)
        #
        #             # 使用更适合避免重叠的布局
        #             pos = nx.spring_layout(G, k=0.15, iterations=20)  # 调整布局参数以减少重叠
        #             nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color=edge_colors,
        #                     node_size=node_size, font_size=font_size, width=width, arrows=True, arrowsize=30)
        #
        #             # 调整自回归边标签的位置
        #             edge_labels = {}
        #             for (u, v), nw in zip(G.edges(), normalized_weights):
        #                 if u == v:  # 自回归边
        #                     label_pos = (pos[u][0], pos[u][1] + 0.1)  # 手动调整标签位置
        #                     ax.text(label_pos[0], label_pos[1], f'{nw:.2f}', fontsize=font_size, color='red',
        #                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
        #                             horizontalalignment='center')
        #                 else:
        #                     edge_labels[(u, v)] = f'{nw:.2f}'
        #
        #             # 绘制非自回归边的标签
        #             nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax,
        #                                          font_size=font_size)
        #
        #         ax.set_title(titles[idx], fontsize=title_font_size)  # 在子图上标注边类型
        #
        #     # 调整子图间的间隔
        #     plt.subplots_adjust(hspace=0, wspace=0.01)
        #     # 调整布局以减少边距
        #     plt.tight_layout()
        #     plt.savefig("graph_visualization.pdf", format='pdf', bbox_inches='tight')
        #     plt.show()

        graph_out0, attn_weights = self.gat(features_graph, edge_index, edge_type, return_attention_weights=True)
        # print(edge_type)
        # count_c0 = sum(1 for edge in edge_type if edge == 2)
        # count_c1 = sum(1 for edge in edge_type if edge == 3)
        # print(f'c=0: {count_c0}, c=1: {count_c1}')
        # print("graph_out", graph_out.shape)

        # visualize_graph_with_edge_weights(features_graph, edge_index, edge_type, attn_weights, node_size=600,
        #                                   font_size=12, width=6, title_font_size=25)

        # self.visualize_tsneAV(graph_out)



        return graph_out, features, graph_out_T, graph_out_A, graph_out_V, features_T, features_A, features_V, fea1, fea2

    def forward(self, data):
        graph_out, features, graph_out_T, graph_out_A, graph_out_V, features_T, features_A, features_V, fea1, fea2 = self.get_rep(
            data)

        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])

        score = self.clf.get_prob1(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score_T = self.clf_T.get_prob1(graph_out_T, data["text_len_tensor"])
        score_A = self.clf_A.get_prob1(graph_out_A, data["text_len_tensor"])
        score_V = self.clf_V.get_prob1(graph_out_V, data["text_len_tensor"])
        scores = score + 0.7 * score_T + 0.2 * score_V + 0.1 * score_A

        # 性别分类
        score_gender = self.clf_gender.get_prob1(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])

        score_gT0 = self.clf_gT0.get_prob1_gender(torch.cat([fea1.squeeze(1), fea2.squeeze(1), graph_out_T], dim=-1),
                                                  data["text_len_tensor"])
        score_gA0 = self.clf_gA0.get_prob1_gender(torch.cat([fea1.squeeze(1), fea2.squeeze(1), graph_out_A], dim=-1),
                                                  data["text_len_tensor"])
        score_gV0 = self.clf_gV0.get_prob1_gender(torch.cat([fea1.squeeze(1), fea2.squeeze(1), graph_out_V], dim=-1),
                                                  data["text_len_tensor"])

        score_gT2 = self.clf_gT.get_prob1_gender(graph_out_T, data["text_len_tensor"])
        score_gA2 = self.clf_gA.get_prob1_gender(graph_out_A, data["text_len_tensor"])
        score_gV2 = self.clf_gV.get_prob1_gender(graph_out_V, data["text_len_tensor"])

        score_gT1 = self.clf_gT.get_prob1_gender(features_T, data["text_len_tensor"])
        score_gA1 = self.clf_gA.get_prob1_gender(features_A, data["text_len_tensor"])
        score_gV1 = self.clf_gV.get_prob1_gender(features_V, data["text_len_tensor"])


        score_g1 = 0.7 * score_gT1 + 0.2 * score_gA1 + 0.1 * score_gV1
        score_g2 = 0.7 * score_gT2 + 0.2 * score_gA2 + 0.1 * score_gV2
        score_g3 = score_gender

        score_g4 = score_gender + 0.7 * score_gT2 + 0.2 * score_gA2 + 0.1 * score_gV2

        # 三个位置注入性别。
        # scores_g = score_g1
        # scores_g = score_g2
        # scores_g = score_g3
        scores_g = score_g4


        log_gender = F.log_softmax(scores_g, dim=-1)
        y_hat_gender = torch.argmax(log_gender, dim=-1)

        log_prob = F.log_softmax(scores, dim=-1)
        y_hat = torch.argmax(log_prob, dim=-1)

        return y_hat, y_hat_gender

    def update_class_counts(self, class_counts, data, num_classes, device):
        """
        Update the class counts based on the label tensor in the data.

        Args:
            class_counts (torch.Tensor): Tensor storing the current counts of each class.
            data (dict): A dictionary containing the data, including 'label_tensor'.
            num_classes (int): The number of classes.
            device (str): The device to run the operation on (e.g., 'cuda:0').

        Returns:
            torch.Tensor: Updated class counts tensor.
        """
        labels = data['label_tensor'].to(device)
        labels = labels.reshape(-1)  # Flatten the label tensor if necessary
        class_counts += torch.bincount(labels, minlength=num_classes).float().to(device)
        return class_counts

    def get_loss(self, data,epoch):

        graph_out, features, graph_out_T, graph_out_A, graph_out_V, features_T, features_A, features_V, fea1, fea2 = self.get_rep(
            data)

        # self.visualize_tsneclass1(graph_out, data["label_tensor"])
        if (epoch==80):
            self.visualize_tsneclass0(graph_out, data["label_tensor"])


        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])
        loss_T = self.clf_T.get_loss(graph_out_T, data["label_tensor"], data["text_len_tensor"])
        loss_A = self.clf_A.get_loss(graph_out_A, data["label_tensor"], data["text_len_tensor"])
        loss_V = self.clf_V.get_loss(graph_out_V, data["label_tensor"], data["text_len_tensor"])

        loss_gT1 = self.clf_gT.get_loss(features_T, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gA1 = self.clf_gA.get_loss(features_A, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gV1 = self.clf_gV.get_loss(features_V, data["xingbie_tensor"], data["text_len_tensor"])  # 局部图前

        loss_gT2 = self.clf_gT.get_loss(graph_out_T, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gA2 = self.clf_gA.get_loss(graph_out_A, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gV2 = self.clf_gV.get_loss(graph_out_V, data["xingbie_tensor"], data["text_len_tensor"])  # 局部图后全局图前

        loss_gT0 = self.clf_gT0.get_loss(torch.cat([fea1.squeeze(1), fea2.squeeze(1), graph_out_T], dim=-1),
                                         data["xingbie_tensor"], data["text_len_tensor"])
        loss_gA0 = self.clf_gA0.get_loss(torch.cat([fea1.squeeze(1), fea2.squeeze(1), graph_out_A], dim=-1),
                                         data["xingbie_tensor"], data["text_len_tensor"])
        loss_gV0 = self.clf_gV0.get_loss(torch.cat([fea1.squeeze(1), fea2.squeeze(1), graph_out_V], dim=-1),
                                         data["xingbie_tensor"], data["text_len_tensor"])  # try

        loss_g = self.clf_gender.get_loss(torch.cat([features, graph_out], dim=-1),
                                          data["xingbie_tensor"], data["text_len_tensor"])  # 全局图后

        loss_emotion = loss + 0.7 * loss_T + 0.2 * loss_V + 0.1 * loss_A

        loss_gender1 = 0.7 * loss_gT1 + 0.2 * loss_gA1 + 0.1 * loss_gV1
        loss_gender2 = 0.7 * loss_gT2 + 0.2 * loss_gA2 + 0.1 * loss_gV2  # fact
        loss_gender3 = loss_g
        loss_gender4 = loss_g + 0.7 * loss_gT2 + 0.2 * loss_gA2 + 0.1 * loss_gV2

        # 三种位置注入性别
        # loss_gender = loss_gender1
        # loss_gender = loss_gender2
        # loss_gender = loss_gender3
        loss_gender = loss_gender4

        loss_total = 0.7 * loss_emotion + 0.3 * loss_gender
        # loss_total = 0.9 * loss_emotion + 0.1 * loss_gender

        return loss_total, loss_emotion, loss_gender
