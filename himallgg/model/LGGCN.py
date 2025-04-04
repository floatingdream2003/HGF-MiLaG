import torch
import torch.nn as nn

from .ContextualEncoder import ContextualEncoder
from .EdgeAtt import EdgeAtt

from .GCN import GCN, SGCN


# from .GCN_RGCN_only import GCN, SGCN#消融实验
# from .GCN_GraphConv_only import GCN, SGCN#消融实验
# from .GCN_RGCN_1head import GCN,SGCN#消融实验

from .Classifier import Classifier
from .functions import batch_graphify, batch_graphify1
import himallgg

from .SoftHGRLoss import SoftHGRLoss
from .SampleWeightedFocalContrastiveLoss import SampleWeightedFocalContrastiveLoss

import torch
from torch.nn import TransformerEncoderLayer
import torch.nn as nn
import torch.nn.functional as F
from .Fusion import *



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


class LGGCN(nn.Module):

    def __init__(self, args):
        super(LGGCN, self).__init__()
        u_dim = 1024
        uA_dim = 1582
        uV_dim = 342
        g_dim = 160
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 6

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        #编码器
        self.rnn = ContextualEncoder(u_dim, g_dim, args)
        self.rnn_A = ContextualEncoder(uA_dim, g_dim, args)
        self.rnn_V = ContextualEncoder(uV_dim, g_dim, args)

        self.cross_attention = CrossModalAttention(g_dim)
        self.edge_att = EdgeAtt(g_dim, args)

        self.gcn = GCN(g_dim * 3, h1_dim, h2_dim, args)
        self.edge_att_all = EdgeAtt(g_dim * 3, args)

        self.clf = Classifier(h2_dim + g_dim * 3, hc_dim, tag_size, args)
        self.clf_T = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_A = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_V = Classifier(g_dim, hc_dim, tag_size, args)

        self.clf_gender = Classifier(h2_dim + g_dim * 3, hc_dim, 2, args)  # 2 表示性别分类

        self.clf_gT = Classifier(g_dim, hc_dim, 2, args)
        self.clf_gA = Classifier(g_dim, hc_dim, 2, args)
        self.clf_gV = Classifier(g_dim, hc_dim, 2, args)

        self.clf_gT0 = Classifier(480, hc_dim, 2, args)
        self.clf_gA0 = Classifier(480, hc_dim, 2, args)
        self.clf_gV0 = Classifier(480, hc_dim, 2, args)

        # 针对不同模态的分类器

        ##################################################################################################################
        #新加代码
        self.HGR_loss = SoftHGRLoss()
        ##################################################################################################################
        # 用于图处理的GCN层
        self.gcn1 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn2 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn3 = SGCN(g_dim, h1_dim, g_dim, args)
        self.att = MultiHeadedAttention(10, g_dim)
        self.args = args  # 超参数
        ##################################################################################################################

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        self.edge_type_to_idx1 = {'00': 0, '01': 1, '10': 2, '11': 3}

        log.debug(self.edge_type_to_idx)

    # 特征提取
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
        #局部图
        graph_out_T = self.gcn1(features_T, edge_index_T, edge_type_T)
        graph_out_A = self.gcn2(features_A, edge_index_A, edge_type_A)
        graph_out_V = self.gcn3(features_V, edge_index_V, edge_type_V)

        fea1 = self.att(graph_out_T, graph_out_A, graph_out_A)
        fea2 = self.att(graph_out_T, graph_out_V, graph_out_V)


        #全局图
        features_graph = torch.cat([graph_out_T, fea1.squeeze(1), fea2.squeeze(1)], dim=-1)
        graph_out = self.gcn(features_graph, edge_index, edge_type)
        ##################################################################################################################
        return graph_out, features, graph_out_T, graph_out_A, graph_out_V,features_T,features_A,features_V,fea1,fea2

    # 前向传播
    def forward(self, data):
        graph_out, features, graph_out_T, graph_out_A, graph_out_V,features_T,features_A,features_V,fea1,fea2 = self.get_rep(data)

        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])

        score = self.clf.get_prob1(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score_T = self.clf_T.get_prob1(graph_out_T, data["text_len_tensor"])
        score_A = self.clf_A.get_prob1(graph_out_A, data["text_len_tensor"])
        score_V = self.clf_V.get_prob1(graph_out_V, data["text_len_tensor"])
        scores = score + 0.7 * score_T + 0.2 * score_V + 0.1 * score_A

        # 性别分类
        score_gender = self.clf_gender.get_prob1(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])



        score_gT0 = self.clf_gT0.get_prob1_gender(torch.cat([fea1.squeeze(1),fea2.squeeze(1),graph_out_T],dim=-1),
                                                data["text_len_tensor"])
        score_gA0 = self.clf_gA0.get_prob1_gender(torch.cat([fea1.squeeze(1),fea2.squeeze(1),graph_out_A],dim=-1),
                                                 data["text_len_tensor"])
        score_gV0 = self.clf_gV0.get_prob1_gender(torch.cat([fea1.squeeze(1),fea2.squeeze(1),graph_out_V],dim=-1),
                                                 data["text_len_tensor"])

        score_gT2 = self.clf_gT.get_prob1_gender(graph_out_T, data["text_len_tensor"])
        score_gA2 = self.clf_gA.get_prob1_gender(graph_out_A, data["text_len_tensor"])
        score_gV2 = self.clf_gV.get_prob1_gender(graph_out_V, data["text_len_tensor"])



        score_gT1 = self.clf_gT.get_prob1_gender(features_T, data["text_len_tensor"])
        score_gA1 = self.clf_gA.get_prob1_gender(features_A, data["text_len_tensor"])
        score_gV1 = self.clf_gV.get_prob1_gender(features_V, data["text_len_tensor"])

        score_g0=0.7*score_gT0+0.2*score_gA0+0.1*score_gV0
        score_g1 = 0.7*score_gT1 + 0.2*score_gA1 + 0.1*score_gV1
        score_g2 = 0.7*score_gT2+0.2*score_gA2+0.1*score_gV2
        score_g3 = score_gender

        scores_g=score_gender+0.7*score_gT2+0.2*score_gA2+0.1*score_gV2

        #三个位置注入性别。
        # scores_g = score_g1
        # scores_g = score_g2
        # scores_g = score_g3
        # scores_g=score_g0

        log_gender = F.log_softmax(scores_g, dim=-1)
        y_hat_gender = torch.argmax(log_gender, dim=-1)

        log_prob = F.log_softmax(scores, dim=-1)
        y_hat = torch.argmax(log_prob, dim=-1)

        return y_hat, y_hat_gender

    # 计算损失
    def get_loss(self, data):

        graph_out, features, graph_out_T, graph_out_A, graph_out_V,features_T,features_A,features_V,fea1,fea2 = self.get_rep(data)


        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])
        loss_T = self.clf_T.get_loss(graph_out_T, data["label_tensor"], data["text_len_tensor"])
        loss_A = self.clf_A.get_loss(graph_out_A, data["label_tensor"], data["text_len_tensor"])
        loss_V = self.clf_V.get_loss(graph_out_V, data["label_tensor"], data["text_len_tensor"])



        loss_gT1 = self.clf_gT.get_loss(features_T, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gA1 = self.clf_gA.get_loss(features_A, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gV1 = self.clf_gV.get_loss(features_V, data["xingbie_tensor"], data["text_len_tensor"])#局部图前

        loss_gT2 = self.clf_gT.get_loss(graph_out_T, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gA2 = self.clf_gA.get_loss(graph_out_A, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gV2 = self.clf_gV.get_loss(graph_out_V, data["xingbie_tensor"], data["text_len_tensor"])#局部图后全局图前


        loss_gT0 = self.clf_gT0.get_loss(torch.cat([fea1.squeeze(1),fea2.squeeze(1),graph_out_T],dim=-1),
                                        data["xingbie_tensor"], data["text_len_tensor"])
        loss_gA0 = self.clf_gA0.get_loss(torch.cat([fea1.squeeze(1),fea2.squeeze(1),graph_out_A],dim=-1),
                                        data["xingbie_tensor"], data["text_len_tensor"])
        loss_gV0 = self.clf_gV0.get_loss(torch.cat([fea1.squeeze(1),fea2.squeeze(1),graph_out_V],dim=-1),
                                        data["xingbie_tensor"], data["text_len_tensor"])  # try


        loss_g = self.clf_gender.get_loss(torch.cat([features, graph_out], dim=-1),
                                               data["xingbie_tensor"], data["text_len_tensor"])#全局图后


        loss_emotion = loss + 0.7 * loss_T + 0.2 * loss_V + 0.1 * loss_A


        loss_gender1 = 0.7*loss_gT1 + 0.2* loss_gA1 +0.1*loss_gV1
        loss_gender2 = 0.7*loss_gT2 + 0.2*loss_gA2 + 0.1*loss_gV2#fact
        loss_gender3 = loss_g
        loss_gender0 = 0.7*loss_gT0 + 0.2*loss_gA0 + 0.1*loss_gV0

        loss_gender=loss_g+0.7*loss_gT2+0.2*loss_gA2+0.1*loss_gV2

        #三种位置注入性别
        # loss_gender = loss_gender1
        # loss_gender = loss_gender2
        # loss_gender = loss_gender3
        # loss_gender = loss_gender0

        loss_total = 0.7*loss_emotion + 0.3*loss_gender
        # loss_total = 0.9 * loss_emotion + 0.1 * loss_gender

        return loss_total,loss_emotion,loss_gender
