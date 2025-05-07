import torch
from torch import nn
from torch.nn import functional as F


class ModalAlign(nn.Module):
    def __init__(self):
        pass

    def softXEnt(self, target, logits, w_min=None):
        pass

    def forward(self, emb, train_links=None, neg_l=None, neg_r=None, weight_norm=None, norm=True):
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
        n_view = self.n_view  # 知识图谱数量
        mask_num = 1e9  # 掩码值

        hidden1, hidden2 = src, tar
        batch_size = hidden1.shape[0]
        hidden_1_large, hidden_2_large = hidden1, hidden2

        num_classes_1 = None
        num_classes_2 = None
        if neg_l is None:
            num_classes_1 = batch_size * n_view
        else:
            num_classes_1 = batch_size * n_view + neg_l.shape[0]
            num_classes_2 = batch_size * n_view + neg_r.shape[0]
        labels_1 = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64),
                             num_classes=num_classes_1).float()
        labels_1 = labels_1.cuda()
        if neg_l is not None:
            labels_2 = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64),
                                 num_classes=num_classes_2).float()
            labels_2 = labels_2.cuda()
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.cuda().float()

        logits_aa = torch.matmul(hidden1, torch.transpose(src, 0, 1)) / tau


class EntityAlign(nn.Module):
    def __init__(self):
        pass
