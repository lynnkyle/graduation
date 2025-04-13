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
