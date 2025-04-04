# 写下改的LGGCN类，加入了权重计算


import torch
import torch.nn as nn

from .ContextualEncoder import ContextualEncoder
from .EdgeAtt import EdgeAtt
from .GCN import GCN, SGCN
from .Classifier import Classifier
from .functions import batch_graphify, batch_graphify1
import himallgg

from .SoftHGRLoss import SoftHGRLoss
from .SampleWeightedFocalContrastiveLoss import SampleWeightedFocalContrastiveLoss

from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from .Fusion import *
from transformers import BertTokenizerFast, BertModel
from .SoftHGRLoss import SoftHGRLoss
from .SampleWeightedFocalContrastiveLoss import SampleWeightedFocalContrastiveLoss


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

        # 初始化各模态特征提取器
        self.rnn = ContextualEncoder(u_dim, g_dim, args)#g_dim特征提取器的输出维度
        self.rnn_A = ContextualEncoder(uA_dim, g_dim, args)
        self.rnn_V = ContextualEncoder(uV_dim, g_dim, args)

        self.cross_attention = CrossModalAttention(g_dim)
        self.edge_att = EdgeAtt(g_dim, args)

        ########################################################################################
        # 初始化 ConfidNet 网络，用于计算各模态置信度
        self.ConfidNet_T = nn.Sequential(
            nn.Linear(g_dim, g_dim * 2),
            nn.ReLU(),
            nn.Linear(g_dim * 2, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, 1),
            nn.Sigmoid()  # 输出在 [0,1] 之间
        )
        self.ConfidNet_A = nn.Sequential(
            nn.Linear(g_dim, g_dim * 2),
            nn.ReLU(),
            nn.Linear(g_dim * 2, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, 1),
            nn.Sigmoid()
        )
        self.ConfidNet_V = nn.Sequential(
            nn.Linear(g_dim, g_dim * 2),
            nn.ReLU(),
            nn.Linear(g_dim * 2, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, 1),
            nn.Sigmoid()
        )
########################################################################################
        #新加代码
        self.HGR_loss = SoftHGRLoss()
        # self.temp_param = args.temp_param
        # self.focus_param = args.focus_param
        # self.sample_weight_param = args.sample_weight_param

        ########################################################################################
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
        # 针对不同模态的分类器


        # 其他层定义，保持和原来一样
        self.gcn1 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn2 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn3 = SGCN(g_dim, h1_dim, g_dim, args)
        self.att = MultiHeadedAttention(10, g_dim)
        self.args = args#超参数

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        self.edge_type_to_idx1 = {'00': 0, '01': 1, '10': 2, '11': 3}

        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        node_features_T = self.rnn(data["text_len_tensor"], data["text_tensor"])  # Text 模态特征
        node_features_A = self.rnn_A(data["text_len_tensor"], data["audio_tensor"])  # Audio 模态特征
        node_features_V = self.rnn_V(data["text_len_tensor"], data["visual_tensor"])  # Visual 模态特征

#########################
        # 计算每个模态的置信度权重
        #写死训练
        txt_conf = self.ConfidNet_T(node_features_T.clone().detach())
        aud_conf = self.ConfidNet_A(node_features_A.clone().detach())
        vis_conf = self.ConfidNet_V(node_features_V.clone().detach())

        # 计算 holo 值
        txt_holo = torch.log(aud_conf) / (torch.log(txt_conf * aud_conf) + 1e-8)
        aud_holo = torch.log(vis_conf) / (torch.log(aud_conf * vis_conf) + 1e-8)
        vis_holo = torch.log(txt_conf) / (torch.log(txt_conf * vis_conf) + 1e-8)

        # 最终权重计算
        cb_txt = txt_conf.detach() + txt_holo.detach()
        cb_aud = aud_conf.detach() + aud_holo.detach()
        cb_vis = vis_conf.detach() + vis_holo.detach()

        # 计算最终的融合权重
        w_all = torch.stack((cb_txt, cb_aud, cb_vis), dim=1)
        softmax = nn.Softmax(dim=1)
        w_all = softmax(w_all)
        w_txt_train = w_all[:, 0]
        w_aud_train = w_all[:, 1]
        w_vis_train = w_all[:, 2]

########################################################################################################################
        #以下是测试的动态权重
        #动态测试
        txt_conf2 = self.ConfidNet_T(node_features_T)
        aud_conf2 = self.ConfidNet_A(node_features_A)
        vis_conf2 = self.ConfidNet_V(node_features_V)

        txt_holo2 = torch.log(aud_conf2) / (torch.log(txt_conf2 * aud_conf2) + 1e-8)
        aud_holo2 = torch.log(vis_conf2) / (torch.log(aud_conf2 * vis_conf2) + 1e-8)
        vis_holo2 = torch.log(txt_conf2) / (torch.log(txt_conf2 * vis_conf2) + 1e-8)

        cb_txt2 = txt_conf2 + txt_holo2
        cb_aud2 = aud_conf2 + aud_holo2
        cb_vis2 = vis_conf2 + vis_holo2

        txt_pred = torch.nn.functional.softmax(node_features_T, dim=1)
        aud_pred = torch.nn.functional.softmax(node_features_V, dim=1)
        vis_pred = torch.nn.functional.softmax(node_features_A, dim=1)

        txt_du = torch.mean(torch.abs(txt_pred - 1 / txt_pred.shape[1]), dim=1, keepdim=True)
        aud_du = torch.mean(torch.abs(aud_pred - 1 / aud_pred.shape[1]), dim=1, keepdim=True)
        vis_du = torch.mean(torch.abs(vis_pred - 1 / vis_pred.shape[1]), dim=1, keepdim=True)

        condition_txt_aud = txt_du > aud_du
        condition_txt_vis = txt_du > vis_du
        condition_aud_vis = vis_du > aud_du

        # 计算 rc
        rc_txt_aud = torch.where(condition_txt_aud, torch.ones_like(txt_du), txt_du / aud_du)
        rc_txt_vis = torch.where(condition_txt_vis, torch.ones_like(txt_du), txt_du / vis_du)
        rc_vis_aud = torch.where(condition_aud_vis, torch.ones_like(vis_du), vis_du / aud_du)

        # 加权组合
        ccb_txt = cb_txt2 * rc_txt_aud * rc_txt_vis
        ccb_aud = cb_aud2 * rc_txt_aud * rc_vis_aud
        ccb_vis = cb_vis2 * rc_txt_vis * rc_vis_aud

        # 计算最终的融合权重
        w_all2 = torch.stack((0.7*ccb_txt, 0.1*ccb_aud, 0.2*ccb_vis), dim=1)
        softmax = nn.Softmax(dim=1)
        w_all2 = softmax(w_all2)
        w_txt_test = w_all2[:, 0]
        w_aud_test = w_all2[:, 1]
        w_vis_test = w_all2[:, 2]

        # features_graph = w_txt.detach() * node_features_T + w_aud.detach() * node_features_A + w_vis.detach() * node_features_V

##############################################

        node_features = torch.cat((node_features_T, node_features_A, node_features_V), 2)
        # 图结构处理和 GCN 操作（保持原有流程）
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

        graph_out_T = self.gcn1(features_T, edge_index_T, edge_type_T)
        graph_out_A = self.gcn2(features_A, edge_index_A, edge_type_A)
        graph_out_V = self.gcn3(features_V, edge_index_V, edge_type_V)

        fea1 = self.att(graph_out_A, graph_out_T, graph_out_A)
        fea2 = self.att(graph_out_V, graph_out_T, graph_out_V)
        features_graph = torch.cat([graph_out_T, fea1.squeeze(1), fea2.squeeze(1)], dim=-1)

        graph_out = self.gcn(features_graph, edge_index, edge_type)

        return graph_out, features, graph_out_T, graph_out_A, graph_out_V,w_txt_train,w_vis_train,w_aud_train,w_txt_test,w_vis_test,w_aud_test


    def forward(self, data):
        # 获取加权后的模态特征
        graph_out, features, graph_out_T, graph_out_A, graph_out_V,w_T,w_V,w_A,w_T2,w_V2,w_A2 = self.get_rep(data)

        # 将多维张量转换为标量
        w_T0=torch.mean(w_T2.detach()).item()
        w_V0=torch.mean(w_V2.detach()).item()
        w_A0=torch.mean(w_A2.detach()).item()
        w_T1=round(w_T0,4)
        w_V1=round(w_V0,4)
        w_A1=round(w_A0,4)
        # print("w_T1", w_T1)
        # print("w_V1", w_V1)
        # print("w_A1", w_A1)

        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])

        score = self.clf.get_prob1(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score_T = self.clf_T.get_prob1(graph_out_T, data["text_len_tensor"])
        score_A = self.clf_A.get_prob1(graph_out_A, data["text_len_tensor"])
        score_V = self.clf_V.get_prob1(graph_out_V, data["text_len_tensor"])
        # 最终融合结果，按一定比例进行加权求和
        scores = score + w_T1 * score_T + w_V1 * score_V + w_A1 * score_A

        # 性别分类
        score_gender = self.clf_gender.get_prob1(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score_gT = self.clf_gender.get_prob1_gender(graph_out_T, data["text_len_tensor"])
        score_gA = self.clf_gender.get_prob1_gender(graph_out_A, data["text_len_tensor"])
        score_gV = self.clf_gender.get_prob1_gender(graph_out_V, data["text_len_tensor"])
        # scores_g = score_gender + w_T1 * score_gT + w_V1 * score_gV + w_A1 * score_gA
        scores_g = score_gender

        log_gender = F.log_softmax(scores_g, dim=-1)
        y_hat_gender = torch.argmax(log_gender, dim=-1)

        log_prob = F.log_softmax(scores, dim=-1)
        y_hat = torch.argmax(log_prob, dim=-1)

        return y_hat, y_hat_gender

    ##################################################################################################################

    # 学长新加的代码

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

    def get_loss(self, data):
        # class_counts = 0
        # class_counts = self.update_class_counts(class_counts, data, 6, self.args.device)
        # self.SWFC_loss = SampleWeightedFocalContrastiveLoss(self.temp_param, self.focus_param, self.sample_weight_param,
        #                                                     class_counts, self.device)
        graph_out, features, graph_out_T, graph_out_A, graph_out_V,w_T,w_V,w_A,w_T2,w_V2,w_A2 = self.get_rep(data)
        # SWFC_loss = self.SWFC_loss(graph_out, data["label_tensor"])

        #将多维张量转换为标量
        w_T0=torch.mean(w_T2.detach()).item()
        w_V0=torch.mean(w_V2.detach()).item()
        w_A0=torch.mean(w_A2.detach()).item()
        w_T1=round(w_T0,4)
        w_V1=round(w_V0,4)
        w_A1=round(w_A0,4)

        # print("w_T=: ", w_T1)
        # print("w_V=: ", w_V1)
        # print("w_A=: ", w_A1)
        # print("sum=",w_T1+w_V1+w_A1)

        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])

        loss_T = self.clf_T.get_loss(graph_out_T, data["label_tensor"], data["text_len_tensor"])
        loss_A = self.clf_A.get_loss(graph_out_A, data["label_tensor"], data["text_len_tensor"])
        loss_V = self.clf_V.get_loss(graph_out_V, data["label_tensor"], data["text_len_tensor"])

        loss_gender = self.clf_gender.get_loss(torch.cat([features, graph_out], dim=-1),
                                               data["xingbie_tensor"], data["text_len_tensor"])

        soft_HGR_loss = self.HGR_loss(graph_out_T, graph_out_A, graph_out_V)

        loss_emotion = loss + w_T1 * loss_T + w_V1 * loss_V + w_A1 * loss_A
        loss_gender = loss_gender

# TODO:把下面的情感和性别的权重进行调整，现在精度低

        #loss_total = 0.8*loss_emotion + 0.1*loss_gender + 0.1*soft_HGR_loss
        loss_total = 0.7 * loss_emotion + 0.3 * loss_gender

        return loss_total