import torch
from torch import nn
from torch.nn import functional as F


class EntityModalAlign(nn.Module):
    def __init__(self, tau=0.05, alpha=0.5, n_view=2):
        self.tau = tau
        self.alpha = alpha
        self.n_view = n_view

    def softXEnt(self, labels, logits, w_min=None):
        """
        :param labels: [batch_size, batch_size * 2]
        :param logits: [batch_size, batch_size * 2]
        :param w_min: [batch_size,]
        :return:
        """
        log_probs = F.log_softmax(logits, dim=1)
        if w_min is not None:
            w_min_t = w_min.unsqueeze(1)
            loss = -(labels * log_probs * w_min_t).sum() / logits.shape[0]
        else:
            loss = -(labels * log_probs).sum() / logits.shape[0]
        return loss

    def forward(self, emb, train_links=None, weight_norm=None, norm=True):
        """
        :param emb: [num_ent, emb_dim]
        :param train_links: [batch_size, 2]
        :param neg_l: None
        :param neg_r: None
        :param weight_norm: [batch_size, 2]
            weight_norm=True: 计算不同知识图谱同一实体同一模态下的损失
            weight_norm=False: 计算不同知识图谱同一实体融合模态下的损失
        :param norm: True
        :return:
        """
        if norm:
            emb = F.normalize(emb, dim=1)
        src = emb[train_links[:, 0]]
        tar = emb[train_links[:, 1]]
        if weight_norm is not None:
            src_w = weight_norm(train_links[:, 0])
            tar_w = weight_norm(train_links[:, 1])
            score_w = torch.stack([src_w, tar_w], dim=1)
            score_w_min = torch.min(score_w, 1)[0]
        else:
            score_w_min = None

        alpha = self.alpha  # 可调参数
        tau = self.tan  # 可调参数
        n_view = self.n_view  # 知识图谱数量 2
        MASK_NUM = 1e9  # 掩码值

        # 对比学习
        hidden1, hidden2 = src, tar
        batch_size = hidden1.shape[0]
        hidden_1_large, hidden_2_large = hidden1, hidden2
        num_classes = batch_size * n_view
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64),
                           num_classes=num_classes).float()
        labels = labels.cuda()
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.cuda().float()
        logits_aa = torch.matmul(hidden1, torch.transpose(src, 0, 1)) / tau
        logits_aa = logits_aa - masks * MASK_NUM  # [batch_size, batch_size] 源知识图谱得分计算
        logits_bb = torch.matmul(hidden2, torch.transpose(tar, 0, 1)) / tau
        logits_bb = logits_bb - masks * MASK_NUM  # [batch_size, batch_size] 目标知识图谱得分计算
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden_2_large, 0, 1)) / tau
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden_1_large, 0, 1)) / tau
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
        loss_a = self.softXEnt(labels, logits_a, w_min=score_w_min)
        loss_b = self.softXEnt(labels, logits_b, w_min=score_w_min)
        return alpha * loss_a + (1 - alpha) * loss_b
