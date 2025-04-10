import argparse
import torch
from torch import nn
from fusion import BertLayer

x = torch.randn(2, 32, 1024)

decoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4, batch_first=True)
decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
y = decoder(x)
print(y)

bertLayer = BertLayer(hidden_size=1024, num_attention_head=4, intermediate_size=2048, hidden_layers=1)
y = bertLayer(x, output_attention=False)
print(y)
