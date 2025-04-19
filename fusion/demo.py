import torch
from torch import nn

# x = torch.randn(2, 32, 1024)
#
# # 1.TransformerEncoder实验
# decoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4, batch_first=True)
# decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
# y = decoder(x)
# print(y)
#
# # 2.BertLayer实验
# bertLayer = BertLayer(hidden_size=1024, num_attention_head=4, intermediate_size=2048, hidden_layers=1)
# y = bertLayer(x, output_attention=False)
# print(y)

# 3.Vision Transformer实验
# 1). Conv2D
x = torch.randn(2, 3, 15, 15)
conv = nn.Conv2d(3, 16, 3, stride=3)
y = conv(x)
print(y.shape)
y = y.flatten(2)
print(y.shape)
# 2). Droppath
x = torch.ones(2, 3, 2, 5)
shape = (x.shape[0],) + (1,) * (x.ndim - 1)
print(shape)
x = torch.rand(shape, dtype=x.dtype, device=x.device)
print(x)
print(0.1 + x)
# 3). AdaptiveAvgPool
# 1d
x = torch.tensor([[[1, 2], [2, 3]], [[3, 4], [4, 5]], [[5, 6], [6, 7]]], dtype=torch.float)
avg_pool = nn.AdaptiveAvgPool1d(1)
y = avg_pool(x)
print(y)
# 2d
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
avg_pool = nn.AdaptiveAvgPool2d(1)
y = avg_pool(x)
print(y.shape)
y = y.flatten(-1)
print(y.shape)

# 4.Swim Transformer实验
x = 72
y = 10
print(x / y)
print(x // y)


# drop_path
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 取值范围是[keep_dim, 1 + keep_dim)
    # random_tensor = random_tensor.floor() # Tensor非原地修改
    random_tensor.floor_()  # Tensor原地修改
    x = x.div(keep_prob) * random_tensor
    return x


x = torch.randn(4, 3, 5, 5)
print(drop_path(x, 0.2, True))

# torch.roll
x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
x = torch.roll(x, shifts=-2, dims=-1)
print(x)

# torch.meshgrid
x = torch.arange(4)
y = torch.arange(4)
coords = torch.stack(torch.meshgrid([x, y], indexing='ij'))  # [2, Mh, Mw]
print(coords)
coords_flatten = torch.flatten(coords, 1)
print(coords_flatten)

# torch.unbind
x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8])
y = x.reshape(3, -1)
a, b, c = y
print(a, b, c)
d, e, f = y.unbind(0)
print(d, e, f)

# torch.split
matrix = torch.randn(12842, 768)
chunk_size = 1024
chunks = matrix.split(chunk_size, dim=0)
print(len(chunks))
print(len(chunks[12]))

# torch.where
x = torch.tensor([0, 100, 1, 100, 2])
y = torch.tensor([100, 3, 100, 4, 100])
z = torch.where(x == 100, y, x)
print(z)
