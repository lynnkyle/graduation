import torch
from torch import nn
from transformers import BertModel

from fusion.fusion import BertLayer

x = torch.randn(2, 16, 1024)
ent_encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4, dim_feedforward=2048, batch_first=True)
encoder = nn.TransformerEncoder(ent_encoder_layer, num_layers=1)
y = encoder(x)
print(y)

encoder = BertLayer(hidden_size=1024, num_attention_head=4, intermediate_size=2048, hidden_layers=1)
y = encoder(x)
print(y)
