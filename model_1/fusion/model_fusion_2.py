import torch
from torch import nn
from fusion.swin_transformer import SwinTransformer
from model_1.model_new import Tucker


class TwoDimensionFusion(nn.Module):
    def __init__(self, num_ent, num_rel, str_dim, num_head, dim_hid, dropout, num_layer_dec, in_channels=1,
                 patch_size=4, window_size=7, embed_dim=16, depths=(2, 2, 4, 2), num_heads=(2, 4, 8, 2),
                 num_classes=-1, in_feat=128, out_feat=256, score_function='tucker'):
        super(TwoDimensionFusion, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.in_channels = in_channels
        self.swin_encoder = SwinTransformer(in_channels=in_channels,
                                            patch_size=patch_size,
                                            window_size=window_size,
                                            embed_dim=embed_dim,
                                            depths=depths,
                                            num_heads=num_heads,
                                            num_classes=num_classes)
        self.swin_linear = nn.Linear(in_features=in_feat, out_features=out_feat)
        decoder_layer = nn.TransformerEncoderLayer(d_model=str_dim, nhead=num_head, dim_feedforward=dim_hid,
                                                   dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layer_dec)
        self.pos_head = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.lp_token = nn.Parameter(torch.Tensor(1, str_dim))
        nn.init.xavier_uniform_(self.lp_token)
        # self.contrastive = ContrastiveLoss()
        self.score_function = score_function
        if score_function == 'tucker':
            self.tucker_decoder = Tucker(str_dim, str_dim)
        else:
            pass
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

    def forward(self, ent_embs, rel_embs):
        ent_embs = ent_embs.view(-1, self.in_channels, 16, 16)
        ent_embs = self.swin_encoder(ent_embs)
        ent_embs = self.swin_linear(ent_embs)
        return torch.cat([ent_embs, self.lp_token], dim=0), rel_embs

    def score(self, triples, emb_ent, emb_rel):
        """
        :param triples: [batch_size, 3]
        :param emb_ent: [num_ent, str_dim]
        :param emb_rel: [num_rel, str_dim]
        :return: [batch_size, num_entity]
        """
        h_seq = emb_ent[triples[:, 0] - self.num_rel].unsqueeze(1) + self.pos_head  # [batch_size, 1, str_dim]
        r_seq = emb_rel[triples[:, 1] - self.num_ent].unsqueeze(1) + self.pos_rel  # [batch_size, 1, str_dim]
        t_seq = emb_ent[triples[:, 2] - self.num_rel].unsqueeze(1) + self.pos_tail  # [batch_size, 1, str_dim]
        triple_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)  # [batch_size, 3, str_dim]
        triple_out = self.decoder(triple_seq)  # [batch_size, 3, str_dim]
        rel_out = triple_out[:, 1, :]  # [batch_size, 1, str_dim] -> [batch_size, str_dim] 降维
        ctx_out = triple_out[
            triples == self.num_ent + self.num_rel]  # [batch_size, 1, str_dim] -> [batch_size, str_dim] 降维
        if self.score_function == 'tucker':
            tucker_emb = self.tucker_decoder(ctx_out, rel_out)
            score = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))  # [batch_size, num_entity]
        else:
            score = torch.inner(ctx_out, emb_ent[:-1])  # [batch_size, num_entity, 1] -> [batch_size, num_entity] 降维
        return score
