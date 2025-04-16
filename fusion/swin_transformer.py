"""
    2021 ICCV Paper
    PatchMerging模块: 高、宽缩小为原来的一半, 深度翻倍
    W-MSA模块-(MSA模块): 目的是为了减少计算量, 缺点是窗口之间无法进行信息交互
    Shifted Window: 实现不同window之间的信息交互, 使用Masked MSA机制(l层是W-MSA模块, l+1层是Shifted Window模块)
    Relative position bias: Attention(Q,K,V) = SoftMax(QK^t / d**0.5 + B)V
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import SwinTransformer, VisionTransformer, ConvNeXt


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 掩码针对样本级别, 每个样本都会有一个对立的随机数
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 取值范围是[keep_dim, 1 + keep_dim)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def window_partition(x, window_size):
    """
    将img_mask划分成一个个没有重叠的window
    :param x:[B, H, W, C]
    :param window_size:
    :return:[B * H // window_size * W // window_size, window_size, window_size, C]
    """
    B, H, W, C = x.size()
    x = x.view(-1, H // window_size, window_size, W // window_size, window_size, C)
    # [B, H // window_size, window_size, W // window_size, window_size, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # [B, H // window_size, W // window_size, window_size, window_size, C] -> [B * H // window_size * W // window_size, window_size, window_size, C]
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将一个个没有重叠的window还原成img_mask
    :param x:[B * H // window_size * W // window_size, window_size, window_size, C]
    :param window_size:
    :return:[B, H, W, C]
    """

    B = int(windows.shape[0] / (H / window_size * W / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # [B, H // window_size, W // window_size, window_size, window_size, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # [B, H // window_size, window_size, W // window_size, window_size, C] -> [B, H, W, C]
    return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, training=False):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        x = drop_path(x, self.drop_prob, self.training)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        :param x: [B, C, H, W]
        :return: [B, H / patch_size * W / patch_size, C]
        """
        B, C, H, W = x.shape

        # padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # sample
        x = self.proj(x)
        B, C, H, W = x.shape
        # H = H / 4, W = W / 4
        x = x.flatten(2).permute(0, 2, 1)
        # [B, C, H, W] -> [B, H / patch_size * W / patch_size, C]
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # number of input channel
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        :param x: [B,L,C]
        :param H: L/W
        :param W: L/H
        :return:# [B, H/2 * W/2, 2C]
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)  # [B, H, W, C]

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]

        x = torch.cat((x0, x1, x2, x3), dim=-1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)  # [B, H/2 * W/2, 4C]
        x = self.norm(x)  # [B, H/2 * W/2, 4C]
        x = self.reduction(x)  # [B, H/2 * W/2, 2C]
        return x


class WindowAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # number of input channel
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        """
        :param x: [B, L, C] PatchMerging之后
        :param attn_mask:
        :return:
        """
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # [B, H, W, C]

        # 把feature_map给pad到window_size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows(Patch ==> Window)
        x_windows = window_partition(shifted_x, self.window_size)
        # [B * Hp // window_size * Wp // window_size, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # [B * Hp // window_size * Wp // window_size, window_size * window_size, C]

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, attn_mask)
        # [B * Hp // window_size * Wp // window_size, window_size * window_size, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # [B * Hp // window_size * Wp // window_size, window_size, window_size, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        # [B, Hp, Wp, C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # FFN
        x = x.view(B, H * W, C)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """
        A basic SwinTransformer layer for one stage = swim transformer block(W-MSA + SW-MSA) + patch merging layer
    """

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim  # number of input channel
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # swim transformer block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if i % 2 == 0 else self.shift_size, mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        """
        为 SW-MSA 计算注意力掩码
        :param x:
        :param H:
        :param W:
        :return:
        """
        # padding
        Hp = int(np.ceil(H / self.window_size) * self.window_size)
        Wp = int(np.ceil(W / self.window_size) * self.window_size)
        # mask
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        masked_windows = window_partition(img_mask, self.window_size)
        # [B * H // window_size * W // window_size, window_size, window_size, C] -> [1 * num_window, window_size, window_size, 1]
        masked_windows = masked_windows.view(-1, self.window_size * self.window_size)
        # [num_window, window_size * window_size]
        attn_mask = masked_windows.unsqueeze(1) - masked_windows.unsqueeze(2)
        # [num_window, 1, window_size * window_size] - [num_window, window_size * window_size, 1]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        """
        :param x:
        :param H:
        :param W:
        :return:
        """
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh * Mw, Mh * Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
            H, W = (H + 1) // 2, (W + 1) // 2  # 由于H,W是奇数时, 需要进行Padding H = H + 1 W = W + 1
        return x, H, W


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, num_classes=1000, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, patch_norm=True, norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 常用于卷积神经网络最后阶段,把每个通道的特征图变成 1×1
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def forward(self, x):
        """
        :param x: [B, C, H, W]
        :return: [B, num_classes]
        """
        x, H, W = self.patch_embed(x)  # [B, L=H / patch_size * W / patch_size, C]
        x = self.pos_drop(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # [B, L, C]
        x = self.avg_pool(x.transpose(1, 2))  # [B, C, 1] -> [B, C]
        x = torch.flatten(x, 1)
        x = self.head(x)  # [B, num_classes]
        return x


if __name__ == '__main__':
    x = torch.randn((2, 3, 224, 224))
    patch_emb = PatchEmbed(patch_size=4, in_channels=3, embed_dim=96, norm_layer=None)
    y = patch_emb(x)
    print(y[0].shape)
    patch_mergy = PatchMerging(dim=96, norm_layer=nn.LayerNorm)
    z = patch_mergy(*y)
    print(z.shape)
