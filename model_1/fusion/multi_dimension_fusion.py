import torch
from torch import nn

from fusion.swin_transformer import SwinTransformer
from model_1.fusion.model_fusion_1 import OneDimensionFusion
from model_1.fusion.model_fusion_2 import TwoDimensionFusion
from model_1.model_new import ContrastiveLoss, Tucker


class MultiDimensionFusion(nn.Module):
    def __init__(self, num_ent, num_rel, str_dim,
                 visual_tokenizer, textual_tokenizer,
                 visual_token_index, textual_token_index,
                 visual_ent_mask, textual_ent_mask,
                 num_head, dim_hid, num_layer_enc_ent, num_layer_dec,
                 dropout=0.1, str_dropout=0.6,
                 visual_dropout=0.1, textual_dropout=0.1,
                 score_function=None):
        super(MultiDimensionFusion, self).__init__()
        self.num_ent = num_ent  # [12842]
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

        self.visual_token_index = visual_token_index
        self.visual_token_embed = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.textual_token_index = textual_token_index
        self.textual_token_embed = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.visual_token_embed.requires_grad_(False)
        self.textual_token_embed.requires_grad_(False)
        false_ent = torch.full((self.num_ent, 1), False).cuda()
        self.ent_mask = torch.cat([false_ent, false_ent, visual_ent_mask, textual_ent_mask], dim=1)
        # false_rel = torch.full((self.num_rel, 1), False).cuda()
        # self.rel_mask = torch.cat([false_rel, false_rel], dim=1)
        self.score_function = score_function
        self.visual_dim = visual_tokens.shape[1]
        self.textual_dim = textual_tokens.shape[1]

        """
            初始化满足Transformer的输入大小: [batch_size, seq_len, emb_dim]
        """
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, str_dim))  # [1, 1, str_dim] -> [batch_size, 1, str_dim]
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, str_dim))  # [1, 1, str_dim] -> [batch_size, 1, str_dim]
        self.ent_emb = nn.Parameter(torch.Tensor(num_ent, 1, str_dim))
        self.rel_emb = nn.Parameter(torch.Tensor(num_rel, 1, str_dim))

        self.str_1d_ln = nn.LayerNorm(str_dim)
        self.str_2d_ln = nn.LayerNorm(str_dim)
        self.str_rel_ln = nn.LayerNorm(str_dim)
        self.visual_ln = nn.LayerNorm(str_dim)
        self.textual_ln = nn.LayerNorm(str_dim)

        self.str_1d_drop = nn.Dropout(str_dropout)
        self.str_2d_drop = nn.Dropout(str_dropout)
        self.str_rel_drop = nn.Dropout(str_dropout)
        self.visual_drop = nn.Dropout(visual_dropout)
        self.textual_drop = nn.Dropout(textual_dropout)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_visual_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_textual_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_visual_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_textual_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))

        self.proj_ent_visual = nn.Linear(self.visual_dim, self.str_dim)
        self.proj_ent_textual = nn.Linear(self.textual_dim, self.str_dim)

        decoder_layer = nn.TransformerEncoderLayer(d_model=str_dim, nhead=num_head, dim_feedforward=dim_hid,
                                                   dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layer_dec)

        self.contrastive = ContrastiveLoss()
        self.num_visual_token = visual_ent_mask.shape[1]

        self.one_dimension_fusion = OneDimensionFusion(num_ent=num_ent,
                                                       num_rel=num_rel,
                                                       str_dim=str_dim,
                                                       num_head=num_head,
                                                       dim_hid=dim_hid,
                                                       num_layer_enc_ent=num_layer_enc_ent,
                                                       num_layer_dec=num_layer_dec,
                                                       dropout=dropout,
                                                       score_function='tucker')
        self.two_dimension_fusion = TwoDimensionFusion(num_ent=num_ent,
                                                       num_rel=num_rel,
                                                       str_dim=str_dim,
                                                       num_head=num_head,
                                                       dim_hid=dim_hid,
                                                       dropout=dropout,
                                                       num_layer_dec=num_layer_dec,
                                                       score_function='tucker')

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_emb)
        nn.init.xavier_uniform_(self.rel_emb)
        nn.init.xavier_uniform_(self.proj_ent_visual.weight)
        nn.init.xavier_uniform_(self.proj_ent_textual.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        # nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_visual_ent)
        nn.init.xavier_uniform_(self.pos_textual_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_visual_rel)
        nn.init.xavier_uniform_(self.pos_textual_rel)
        # nn.init.xavier_uniform_(self.pos_head)
        # nn.init.xavier_uniform_(self.pos_rel)
        # nn.init.xavier_uniform_(self.pos_tail)
        # self.proj_ent_visual.bias.data.zero_()
        # self.proj_ent_textual.bias.data.zero_()

    def forward(self):
        ent_token = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_1d_str = self.str_1d_drop(self.str_1d_ln(self.ent_emb)) + self.pos_str_ent
        ent_visual_token = self.visual_token_embed(self.visual_token_index)
        rep_ent_visual = self.visual_drop(self.visual_ln(self.proj_ent_visual(ent_visual_token))) + self.pos_visual_ent
        ent_textual_token = self.textual_token_embed(self.textual_token_index)
        rep_ent_textual = self.textual_drop(
            self.textual_ln(self.proj_ent_textual(ent_textual_token))) + self.pos_textual_ent
        ent_seq = torch.cat([ent_token, rep_ent_1d_str, rep_ent_visual, rep_ent_textual], dim=1)
        rel_embs = self.str_rel_drop(self.str_rel_ln(self.rel_emb)).squeeze(1)
        d1_fusion = self.one_dimension_fusion(ent_seq, self.ent_mask, rel_embs)
        rep_ent_2d_str = self.str_2d_drop(self.str_2d_ln(self.ent_emb))
        d2_fusion = self.two_dimension_fusion(rep_ent_2d_str, rel_embs)
        return d1_fusion, d2_fusion

    def score(self, triple, d1_fusion, d2_fusion):
        d1_ent_emb, d1_rel_emb = d1_fusion
        d2_ent_emb, d2_rel_emb = d2_fusion
        d1_score = self.one_dimension_fusion.score(triple, d1_ent_emb, d1_rel_emb)
        d2_score = self.two_dimension_fusion.score(triple, d2_ent_emb, d2_rel_emb)
        return 0.9 * d1_score + 0.1 * d2_score
