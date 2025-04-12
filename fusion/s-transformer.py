import torch
from torch import nn
import torch.nn.functional as F


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
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        :param x:
        :param H:
        :param W:
        :return:
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat((x0, x1, x2, x3), dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, H, W):
        pass


class SwinTransformerBlock(nn.Module):
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

    def forward(self, x):
        pass


if __name__ == '__main__':
    x = torch.randn((2, 3, 224, 224))
    patch_emb = PatchEmbed(patch_size=4, in_channels=3, embed_dim=96, norm_layer=None)
    y = patch_emb(x)
    print(y[0].shape)
    patch_mergy = PatchMerging(dim=96, norm_layer=nn.LayerNorm)
    z = patch_mergy(*y)
    print(z.shape)
