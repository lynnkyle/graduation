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

    def __init__(self, drop_prob=0., training=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
        2D Image to Patch Embedding
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
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"img size {H} and {W} error"
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, C, HW] -> [B, HW, C] ???
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
        :return:
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
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, mlp_ratio=4., attn_drop_ratio=0.,
                 proj_drop_ratio=0, drop_path_ratio=0., mlp_drop_ratio=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
            drop_path: 不会更新参数, 可以复用
            norm: 会更新参数，不可以复用
        """
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio)
        self.drop_path = DropPath(drop_prob=drop_path_ratio, training=False) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    pass
