import torch
from torch import nn


class Mamba(nn.Module):
    def __init__(self):
        pass

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape  # [batch, seq_len, dim]
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache0(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                pass

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


import math
from transformers import apply_chunking_to_forward


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_head, num_hidden_layers, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_head = num_attention_head
        self.num_hidden_layers = num_hidden_layers
        assert hidden_size % num_attention_head == 0
        self.num_attention_head = num_attention_head
        self.attention_head_size = int(hidden_size / num_attention_head)
        self.all_head_size = self.num_attention_head * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state, output_attention=False):
        """
        :param hidden_state:    [batch_size, seq_len, hidden_size]
        :param output_attention:
        :return:
        """
        query_layer = self.transpose_for_scores(self.query(hidden_state))
        # [batch_size, num_attention_head, seq_len, attention_head_size]
        key_layer = self.transpose_for_scores(self.key(hidden_state))
        # [batch_size, num_attention_head, seq_len, attention_head_size]
        value_layer = self.transpose_for_scores(self.value(hidden_state))
        # [batch_size, num_attention_head, seq_len, attention_head_size]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [batch_size, num_attention_head, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # [batch_size, num_attention_head, seq_len, seq_len]
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # [batch_size, num_attention_head, seq_len, seq_len]
        attention_probs = self.dropout(attention_probs)
        # [batch_size, num_attention_head, seq_len, seq_len]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [batch_size, num_attention_head, seq_len, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # [batch_size, seq_len, num_attention_head, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # [batch_size, seq_len, all_head_size]
        output = (context_layer, attention_probs) if output_attention else (context_layer, None)
        return output

    def transpose_for_scores(self, x):
        """
        :param x:   [batch_size, seq_len, hidden_size]
        :return:    [batch_size, num_attention_head, seq_len, attention_head_size]
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_head, self.attention_head_size)  # [batch_size, seq_len, hidden_size]
        x = x.view(new_x_shape)  # [batch_size, seq_len, num_attention_head, attention_head_size]
        return x.permute(0, 2, 1, 3)  # [batch_size, num_attention_head, seq_len, attention_head_size]


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state, input_tensor):
        """
        :param hidden_state: [batch_size, seq_len, hidden_size]
        :param input_tensor:
        :return:
        """
        hidden_state = self.dense(hidden_state)  # [batch_size, seq_len, hidden_size]
        hidden_state = self.dropout(hidden_state)  # [batch_size, seq_len, hidden_size]
        hidden_state = self.layer_norm(hidden_state + input_tensor)  # [batch_size, seq_len, hidden_size]
        return hidden_state


class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_head, hidden_layers, dropout=0.1):
        super().__init__()
        self.self_attn = BertSelfAttention(hidden_size, num_attention_head, hidden_layers, dropout)
        self.self_out = BertSelfOutput(hidden_size)

    def forward(self, hidden_state, output_attention=False):
        self_attn_output = self.self_attn(hidden_state, output_attention)
        # ([batch_size, seq_len, hidden_size], [batch_size, num_attention_head, seq_len, seq_len])
        self_out_output = self.self_out(self_attn_output[0], hidden_state)
        output = (self_out_output,) + self_attn_output[1:]
        # ([batch_size, seq_len, hidden_size], [batch_size, num_attention_head, seq_len, seq_len])
        # 保持 BertSelfAttention 的额外输出, 防止信息丢失
        return output


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_state):
        hidden_state = self.dense(hidden_state)  # [batch_size, seq_len, intermediate_size]
        hidden_state = self.intermediate_act_fn(hidden_state)  # [batch_size, seq_len, intermediate_size]
        return hidden_state


class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state, input_tensor):
        hidden_state = self.dense(hidden_state)  # [batch_size, seq_len, hidden_size]
        hidden_state = self.dropout(hidden_state)  # [batch_size, seq_len, hidden_size]
        hidden_state = self.layer_norm(hidden_state + input_tensor)  # [batch_size, seq_len, hidden_size]
        return hidden_state


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_head, intermediate_size, hidden_layers, dropout=0.1,
                 use_intermediate=False):
        super(BertLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_head = num_attention_head
        self.intermediate_size = intermediate_size
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.use_intermediate = use_intermediate
        self.chunk_size_feed_forward = 0  # 划分 seq_len[token数量]
        self.seq_len_dim = 1
        self.attn = BertAttention(hidden_size, num_attention_head, hidden_layers)
        if self.use_intermediate:
            self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size)

    def forward(self, hidden_state, output_attention=False):
        self_attn_output = self.attn(hidden_state, output_attention)
        if not self.use_intermediate:
            return (self_attn_output[0], self_attn_output[1])
        attn_output = self_attn_output[0]  # [batch_size, seq_len, hidden_size]
        output = self_attn_output[1]
        """
            apply_chunking_to_forward 的主要作用是将输入序列分块，减少每次前向传播时的内存占用，特别是当输入序列非常长时
        """
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        output = (layer_output, output)
        return output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # ([batch_size, seq_len, hidden_size], [batch_size, num_attention_head, seq_len, seq_len])
        return layer_output
