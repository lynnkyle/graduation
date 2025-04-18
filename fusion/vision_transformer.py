"""
    2020 ICLR Paper
    PatchEmbed模块: Patch(16*16)对图片进行下采样
    Multi-Self-Attention(MSA)模块: Patch的多头注意力机制, [!!!important]会对每个Patch求Q, K, V
"""
from collections import OrderedDict
from functools import partial

import torch
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    :param x:   模型的输入
    :param drop_prob:   模型随机丢弃某个分支的概率
    :param training:    模型的状态
    :return:
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [batch_size, 1, ..., 1]
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
        [Dropout的平替]
        Drop paths (Stochastic Depth) per sample pub.
    """

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
        2D Image to Patch Embedding
        Patch_Num:  H/16 * W/16
        Patch_Size: 768 = 16 * 16 * 3
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.path_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        :param x:
        :return: [B, HW, C] [B, 14*14, 768]
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"img size {H} and {W} error"
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, C, HW] -> [B, HW, C]
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # [拼接多头进行映射]
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        """
        :param x: [batch_size, num_patches, embed_dim]
        :return: [batch_size, num_patches, embed_dim]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [3, batch_size, num_head, num_patches, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [batch_size, num_head, num_patches, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # [batch_size, num_head, num_patches, num_patches]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # [batch_size, num_head, num_patches, head_dim] -> [batch_size, num_patches, embed_dim]
        x = self.proj(x)
        # [batch_size, num_patches, embed_dim]
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
        MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        :param x: [batch_size, num_patches, embed_dim]
        :return: [batch_size, num_patches, embed_dim]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qkv_scale=None, mlp_ratio=4., attn_drop_ratio=0.,
                 proj_drop_ratio=0., drop_path_ratio=0., mlp_drop_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
            drop_path: 不会更新参数, 可以复用
            norm: 会更新参数，不可以复用
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qkv_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio)
        self.drop_path = DropPath(drop_prob=drop_path_ratio, training=False) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop_ratio)

    def forward(self, x):
        """
        :param x: [B, HW, C] <==> [batch_size, num_patches, embed_dim]
        :return: [batch_size, num_patches, embed_dim]
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _init_vit_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, qkv_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.,
                 drop_path_ratio=0., mlp_drop_ratio=0., drop_ratio=0, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, representation_size=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_tokens = 1
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # [1, 1, 768] 用于分类的token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # [1, 197, 768]
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(
            *[Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_scale=qkv_scale, mlp_ratio=mlp_ratio,
                    attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio, drop_path_ratio=dpr[i],
                    mlp_drop_ratio=mlp_drop_ratio, norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        if representation_size:
            self.has_logits = True
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(representation_size, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weight)

    def forward_features(self, x):
        # [B,C,H,W]->[B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)  # [B, 197, 768]
        x = self.blocks(x)  # [batch_size, num_patches, embed_dim]
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])  # [batch_size, 768]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


import random
import numpy as np

"""
    代码可复现
"""
torch.cuda.set_device(1)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# torch.set_num_threads(8)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    vit = VisionTransformer(img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12,
                            representation_size=1024, norm_layer=nn.LayerNorm, act_layer=nn.GELU, num_classes=15)
    x = torch.randn((6, 3, 224, 224))
    y = vit(x)
    print(y.shape)
    print(y)
