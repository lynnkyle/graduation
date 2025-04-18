import torch
from torch import nn
from model_new import Tucker, ContrastiveLoss
from fusion.swin_transformer import SwinTransformer
import numpy as np


class MyGo(nn.Module):
    def __init__(self, num_ent, num_rel, str_dim,
                 visual_tokenizer, textual_tokenizer,
                 visual_token_index, textual_token_index,
                 visual_ent_mask, textual_ent_mask,
                 num_head, dim_hid, num_layer_enc_ent,
                 num_layer_enc_rel, num_layer_dec,
                 dropout=0.1, str_dropout=0.6,
                 visual_dropout=0.1, textual_dropout=0.1,
                 score_function=None):
        super(MyGo, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.str_dim = str_dim
        if visual_tokenizer == 'beit':
            visual_tokens = torch.load("tokens/visual.pth")
        elif visual_tokenizer == 'vggan':
            visual_tokens = torch.load("tokens/visual_vqgan.pth")
        else:
            raise NotImplementedError
        if textual_tokenizer == 'bert':
            textual_tokens = torch.load("tokens/textual.pth")
        elif textual_tokenizer == 'roberta':
            textual_tokens = torch.load("tokens/textual_roberta.pth")
        elif textual_tokenizer == 'llama':
            textual_tokens = torch.load("tokens/textual_roberta.pth")
        else:
            raise NotImplementedError

        self.visual_token_index = visual_token_index  # [12842, 4]
        self.visual_token_embed = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.textual_token_index = textual_token_index  # [12842, 4]
        self.textual_token_embed = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.visual_token_embed.requires_grad_(False)
        self.textual_token_embed.requires_grad_(False)
        false_ent = torch.full((self.num_ent, 1), False).cuda()
        self.ent_mask = torch.cat([false_ent, false_ent, visual_ent_mask, textual_ent_mask], dim=1)
        false_rel = torch.full((self.num_rel, 1), False).cuda()
        self.rel_mask = torch.cat([false_rel, false_rel], dim=1)
        self.score_function = score_function
        self.visual_dim = visual_tokens.shape[1]
        self.textual_dim = textual_tokens.shape[1]

        """
            初始化满足Transformer的输入大小: [batch_size, seq_len, emb_dim]
        """
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, str_dim))  # [1, 1, str_dim] -> [batch_size, 1, str_dim]
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, str_dim))  # [1, 1, str_dim] -> [batch_size, 1, str_dim]
        self.ent_emb = nn.Parameter(torch.Tensor(num_ent, 1, str_dim))
        self.lp_token = nn.Parameter(torch.Tensor(1, str_dim))
        self.rel_emb = nn.Parameter(torch.Tensor(num_rel, 1, str_dim))

        self.str_ln = nn.LayerNorm(str_dim)
        self.str_rel_ln = nn.LayerNorm(str_dim)
        self.visual_ln = nn.LayerNorm(str_dim)
        self.textual_ln = nn.LayerNorm(str_dim)

        self.str_drop = nn.Dropout(str_dropout)
        self.visual_drop = nn.Dropout(visual_dropout)
        self.textual_drop = nn.Dropout(textual_dropout)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_visual_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_textual_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_visual_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_textual_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        # self.pos_head = nn.Parameter(torch.Tensor(1, 1, str_dim))
        # self.pos_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        # self.pos_tail = nn.Parameter(torch.Tensor(1, 1, str_dim))

        self.proj_ent_visual = nn.Linear(self.visual_dim, self.str_dim)
        self.proj_ent_textual = nn.Linear(self.textual_dim, self.str_dim)

        # ent_encoder_layer = nn.TransformerEncoderLayer(d_model=str_dim, nhead=num_head, dim_feedforward=dim_hid,
        #                                                dropout=dropout, batch_first=True)
        # self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layers=num_layer_enc_ent)
        # rel_encoder_layer = nn.TransformerEncoderLayer(d_model=str_dim, nhead=num_head, dim_feedforward=dim_hid,
        #                                                dropout=dropout, batch_first=True)
        # self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layers=num_layer_enc_rel)
        # decoder_layer = nn.TransformerEncoderLayer(d_model=str_dim, nhead=num_head, dim_feedforward=dim_hid,
        #                                            dropout=dropout, batch_first=True)
        # self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layer_dec)

        self.contrastive = ContrastiveLoss()
        self.num_visual_token = visual_ent_mask.shape[1]
        # if self.score_function == 'tucker':
        #     self.tucker_decoder = Tucker(str_dim, str_dim)
        # else:
        #     pass

        # [!!!important] 2D fusion
        # str_dim=256
        # visual_dim=32
        # textual_dim=768
        self.proj_ent_str_2d_1 = nn.Linear(str_dim, 1024)
        self.proj_ent_str_2d_2 = nn.Linear(str_dim, 1024)
        self.proj_rel_str_2d = nn.Linear(str_dim, 768)
        self.str_ln_2d_1 = nn.LayerNorm(1024)
        self.str_drop_2d_1 = nn.Dropout(0.3)
        self.str_ln_2d_2 = nn.LayerNorm(1024)
        self.str_drop_2d_2 = nn.Dropout(0.3)
        self.str_rel_ln_2d = nn.LayerNorm(768)
        self.str_rel_drop_2d = nn.Dropout(0.3)
        self.proj_ent_visual_2d = nn.Linear(self.visual_dim, 256)
        self.visual_ln_2d = nn.LayerNorm(256)
        self.visual_drop_2d = nn.Dropout(0.3)
        self.proj_ent_textual_2d = nn.Linear(self.textual_dim, 256)
        self.textual_ln_2d = nn.LayerNorm(256)
        self.textual_drop_2d = nn.Dropout(0.1)
        self.swim_transformer = SwinTransformer(patch_size=4, in_channels=4, num_classes=-1, embed_dim=96,
                                                window_size=4)
        self.lp_token_2d = nn.Parameter(torch.Tensor(1, 768))
        self.pos_head = nn.Parameter(torch.Tensor(1, 1, 768))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, 768))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, 768))
        decoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
        if self.score_function == 'tucker':
            self.tucker_decoder = Tucker(768, 768)
        else:
            pass
        # init_weight
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_emb)
        nn.init.xavier_uniform_(self.rel_emb)
        nn.init.xavier_uniform_(self.proj_ent_visual.weight)
        nn.init.xavier_uniform_(self.proj_ent_textual.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_visual_ent)
        nn.init.xavier_uniform_(self.pos_textual_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_visual_rel)
        nn.init.xavier_uniform_(self.pos_textual_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)
        # self.proj_ent_visual.bias.data.zero_()
        # self.proj_ent_textual.bias.data.zero_()

    def encoder1DFusion(self):
        ent_token = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.str_drop(self.str_ln(self.ent_emb)) + self.pos_str_ent
        ent_visual_token = self.visual_token_embed(self.visual_token_index)
        rep_ent_visual = self.visual_drop(self.visual_ln(self.proj_ent_visual(ent_visual_token))) + self.pos_visual_ent
        ent_textual_token = self.textual_token_embed(self.textual_token_index)
        rep_ent_textual = self.textual_drop(
            self.textual_ln(self.proj_ent_textual(ent_textual_token))) + self.pos_textual_ent
        ent_seq = torch.cat([ent_token, rep_ent_str, rep_ent_visual, rep_ent_textual], dim=1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)[:, 0]
        rel_embs = self.str_rel_drop_2d(self.str_rel_ln_2d(self.proj_rel_str_2d())).squeeze(1)
        return torch.cat([ent_embs, self.lp_token], dim=0), rel_embs

    def encoder2DFusion(self, emb_ent, emb_rel, ent_visual_token, ent_textual_token):
        ent_emb_matrix_1 = self.str_drop_2d_1(self.str_ln_2d_1(self.proj_ent_str_2d_1(emb_ent)))
        ent_emb_matrix_1 = ent_emb_matrix_1.view(-1, 4, 16, 16).contiguous()  # [12842, 4, 16, 16]
        # print(ent_emb_matrix_1.shape)
        ent_emb_matrix_2 = self.str_drop_2d_2(self.str_ln_2d_2(self.proj_ent_str_2d_2(emb_ent)))
        ent_emb_matrix_2 = ent_emb_matrix_2.view(-1, 4, 16, 16).contiguous()  # [12842, 4, 16, 16]
        # print(ent_emb_matrix_2.shape)
        visual_emb_matrix = self.visual_drop_2d(self.visual_ln_2d(self.proj_ent_visual_2d(ent_visual_token)))
        visual_emb_matrix = visual_emb_matrix.view(-1, 4, 16, 16).contiguous()  # [12842, 4, 16, 16]
        # print(visual_emb_matrix.shape)
        textual_emb_matrix = self.textual_drop_2d(self.textual_ln_2d(self.proj_ent_textual_2d(ent_textual_token)))
        textual_emb_matrix = textual_emb_matrix.view(-1, 4, 16, 16).contiguous()  # [12842, 4, 16, 16]
        # print(textual_emb_matrix.shape)
        ent_emb_matrix = torch.cat((ent_emb_matrix_1, ent_emb_matrix_2), dim=-2)
        ent_modal_matrix = torch.cat((visual_emb_matrix, textual_emb_matrix), dim=-2)
        matrix = torch.cat((ent_emb_matrix, ent_modal_matrix), dim=-1)  # [12842, 4, 32, 32]
        # print(matrix.shape)
        ent_embs = self.swim_transformer(matrix)
        # print(ent_embs.shape)
        rel_embs = self.str_rel_drop_2d(self.str_rel_ln_2d(self.proj_rel_str_2d(emb_rel))).squeeze(1)
        return torch.cat([ent_embs, self.lp_token_2d], dim=0), rel_embs

    def forward(self, triples):
        h_seq_idx = triples[:, 0] - self.num_rel
        t_seq_idx = triples[:, 2] - self.num_rel
        ent_idx = torch.where(h_seq_idx == self.num_ent, t_seq_idx, h_seq_idx)
        emb_ent = self.ent_emb[ent_idx]
        emb_rel = self.rel_emb[triples[:, 1] - self.num_ent]
        visual_token_idx = self.visual_token_index[ent_idx]
        ent_visual_token = self.visual_token_embed(visual_token_idx)
        textual_token_idx = self.textual_token_index[ent_idx]
        ent_textual_token = self.textual_token_embed(textual_token_idx)
        return self.encoder2DFusion(emb_ent, emb_rel, ent_visual_token, ent_textual_token)

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

    """
        MYGO 模型利用 Transformer 编码器中的 dropout 机制人为制造多模态嵌入的细微变化，相当于一种轻量级数据增强。
        然后，它在批量（batch）中使用对比学习，让模型学会关注真正重要的信息，提高实体表示的区分性和鲁棒性。
    """

    def contrastive_loss_finegrained(self, emb_ent1):
        """
        :param emb_ent: [num_ent, str_dim]
        :return:
        """
        ent_token = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.str_drop(self.str_ln(self.ent_emb)) + self.pos_str_ent
        ent_visual_token = self.visual_token_embed(self.visual_token_index)
        rep_ent_visual_token = self.visual_drop(
            self.visual_ln(self.proj_ent_visual(ent_visual_token))) + self.pos_visual_ent
        ent_textual_token = self.textual_token_embed(self.textual_token_index)
        rep_ent_textual_token = self.textual_drop(
            self.textual_ln(self.proj_ent_textual(ent_textual_token))) + self.pos_textual_ent
        ent_seq = torch.cat([ent_token, rep_ent_str, rep_ent_visual_token, rep_ent_textual_token], dim=1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)  # [batch_size, 4, str_dim]
        emb_ent2 = torch.cat([ent_embs[:, 0], self.lp_token], dim=0)
        emb_ent3 = torch.cat([torch.mean(ent_embs, dim=1), self.lp_token], dim=0)
        emb_ent4 = torch.cat([torch.mean(ent_embs[:, 2:2 + self.num_visual_token], dim=1), self.lp_token],
                             dim=0)  # [batch_size, str_dim]
        emb_ent5 = torch.cat([torch.mean(ent_embs[:, 2 + self.num_visual_token:], dim=1), self.lp_token],
                             dim=0)  # [batch_size, str_dim]
        select_ent = torch.randperm(ent_embs.shape[0])[:2 * self.str_dim]
        contrastive_loss = 0
        for emb in [emb_ent2, emb_ent3, emb_ent4, emb_ent5]:
            contrastive_loss += self.contrastive(emb[select_ent], emb_ent1[select_ent])
        contrastive_loss /= 4
        return contrastive_loss
