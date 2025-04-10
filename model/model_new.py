import torch
from torch import nn


class Tucker(nn.Module):
    def __init__(self, e_dim, r_dim):
        super(Tucker, self).__init__()
        # self.e_dim = e_dim
        # self.r_dim = r_dim
        self.W = nn.Parameter(torch.rand(r_dim, e_dim, e_dim))  # [r_dim, e_dim, e_dim]
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(e_dim)
        self.bn1 = nn.BatchNorm1d(e_dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.output_drop = nn.Dropout(0.5)

    """
        计算的是: W * e * r, 计算的不是得分
    """

    def forward(self, ent_emb, rel_emb):
        """
        :param ent_emb: [batch_size, e_dim]
        :param rel_emb: [batch_size, r_dim]
        :return: [batch_size, e_dim]
        """
        x = self.bn0(ent_emb)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))  # [batch_size, 1, e_dim]

        r = torch.mm(rel_emb, self.W.view(rel_emb.size(1), -1))  # [batch_size, e_dim * e_dim]
        r = r.view(-1, x.size(2), x.size(2))  # [batch_size, e_dim, e_dim]
        r = self.hidden_drop(r)

        x = torch.bmm(x, r)  # [batch_size, 1, e_dim]
        x = x.view(-1, x.size(2))  # [batch_size, e_dim]
        x = self.bn1(x)
        x = self.output_drop(x)
        return x


"""
    计算嵌入的相似度
"""


class Similarity(nn.Module):
    def __init__(self, temp):
        super(Similarity, self).__init__()
        self.temp = temp  # 温度参数
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super(ContrastiveLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.similarity_fn = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        """:
        :param emb1: [batch_size, dim]->[batch_size, 1, dim]
        :param emb2: [batch_size, dim]->[1, batch_size, dim]
        """
        batch_sim = self.similarity_fn(emb1.unsqueeze(1), emb2.unsqueeze(0))  # [batch_size, batch_size]
        """
            计算第一次、第二次过实体编码器Transformer得到的嵌入的损失
        """
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels.cuda())


if __name__ == '__main__':
    x = torch.randn((3, 6, 5))
    print(x.size())
    print(x.size(0))
    print(x.size(1))
    print(x.size(2))
