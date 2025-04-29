"""
    2021 ICCV Paper
    PatchMerging模块: 高、宽缩小为原来的一半, 深度翻倍
    W-MSA模块-(MSA模块): 目的是为了减少计算量, 缺点是窗口之间无法进行信息交互
    Shifted Window: 实现不同window之间的信息交互, 使用Masked MSA机制(l层是W-MSA模块, l+1层是Shifted Window模块)
    Relative position bias: Attention(Q,K,V) = SoftMax(QK^t / d**0.5 + B)V
"""
import random
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
    # [B, H // window_size, W // window_size, window_size, window_size, C] => [B * H // window_size * W // window_size, window_size, window_size, C]
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
    # [B, H // window_size, window_size, W // window_size, window_size, C] => [B, H, W, C]
    return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

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
        # [B, C, H, W] => [B, H / patch_size * W / patch_size, C]
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


class WindowAttention(nn.Module):
    """
        WMSA MSA
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim  # number of input channel
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative_position_bias_table
        # relative_position_bias_table是从relative_position_index到relative_position_bias的映射
        # 行索引的范围 [-M+1, M-1] 列索引的范围 [-M+1, M-1]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # relative_position_bias_index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh * Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # [2, Mh * Mw, 1] - [2, 1, Mh * Mw] => [2, Mh * Mw, Mh * Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh * Mw, Mh * Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = torch.sum(relative_coords, dim=-1)  # [Mh * Mw, Mh * Mw]
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        :param x: [B * num_windows, window_size * window_size, C] => [B * Hp // window_size * Wp // window_size, window_size * window_size, C]
        :param mask: []
        :return:
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [3, B_, num_heads, window_size * window_size, C // num_heads]
        q, k, v = qkv.unbind(0)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # [B_, num_heads, window_size * window_size, C // num_heads]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # [B_, num_heads, window_size * window_size, window_size * window_size]
        relative_position_bias = (self.relative_position_bias_table[self.relative_position_index.view(-1)]
                                  .view(self.window_size[0] * self.window_size[1],
                                        self.window_size[0] * self.window_size[1],
                                        -1))
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # [1, window_size * window_size, window_size * window_size] => [num_head, window_size * window_size, window_size * window_size]
        attn = attn + relative_position_bias.unsqueeze(0)
        # [B_, num_heads, window_size * window_size, window_size * window_size]
        if mask is not None:
            # mask: [num_windows, window_size * window_size, window_size * window_size]
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # [B_, num_heads, window_size * window_size, C // num_heads] => [B_, window_size * window_size, C]
        x = self.proj(x)
        x = self.proj_drop(x)
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
        self.attn = WindowAttention(dim=dim, window_size=(window_size, window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
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

        # partition windows(Patch => Window)
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
        x = shortcut + self.drop_path(x)
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
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
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
        # [B * H // window_size * W // window_size, window_size, window_size, C] => [1 * num_window, window_size, window_size, 1]
        masked_windows = masked_windows.view(-1, self.window_size * self.window_size)
        # [num_window, window_size * window_size]
        attn_mask = masked_windows.unsqueeze(1) - masked_windows.unsqueeze(2)
        # [num_window, 1, window_size * window_size] - [num_window, window_size * window_size, 1]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # [num_window, window_size * window_size, window_size * window_size]
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
            x = self.downsample(x, H, W)
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
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 常用于卷积神经网络最后阶段,把每个通道的特征图变成 1×1
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
        x = self.avg_pool(x.transpose(1, 2))  # [B, C, 1] => [B, C]
        x = torch.flatten(x, 1)
        x = self.head(x)  # [B, num_classes]
        return x


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
    model = SwinTransformer(in_channels=1,
                            patch_size=4,
                            window_size=7,
                            embed_dim=16,
                            depths=(2, 2, 6, 2),
                            num_heads=(2, 4, 8, 2),
                            num_classes=5)
    # self.swin_encoder = SwinTransformer(in_channels=1,
    #                                     patch_size=4,
    #                                     window_size=7,
    #                                     embed_dim=64,
    #                                     depths=(2,),
    #                                     num_heads=(4,),
    #                                     num_classes=-1)
    x = torch.randn(12846, 1, 16, 16)
    y = model(x)
    print(y)
