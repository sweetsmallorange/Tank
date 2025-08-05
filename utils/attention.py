import math

import numpy as np
import torch
import torch.nn as nn

from utils.masking import TriangularCausalMask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, old=False):
        super(ScaledDotProductAttention, self).__init__()
        self.old = old

    def forward(self, Q, K, V, mask):
        """
        Q: [batch_size, n_heads, len_q, d_k)]
        K: [batch_size, n_heads, len_k(=len_v), d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_k]
        mask: [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32))  # scores : [batch_size, n_heads, len_q, len_k]

        if mask is not None:
            if self.old is True:
                scores += mask
            else:
                scores.masked_fill_(mask.mask, -np.inf)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        # mask必须是一个ByteTensor 而且shape必须和 a一样 并且元素只能是 0或者1 ，是将 mask中为1的 元素所在的索引，
        # 在a中相同的的索引处替换为 value  ,mask value必须同为tensor

        attn = torch.softmax(scores, dim=-1)
        # attn: [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)
        # context: [batch_size, n_heads, len_q, d_k]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.0, mask_flag=True):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self.mask_flag = mask_flag

    def forward(self, input_Q, input_K, input_V, mask=None):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k(=len_v), d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        batch_size, len_q, _ = input_Q.shape
        Q = self.W_Q(input_Q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        if self.mask_flag:
            if mask is None:
                mask = TriangularCausalMask(batch_size, len_q, device=input_Q.device)
        context = ScaledDotProductAttention()(Q, K, V, mask)
        # context: [batch_size, n_heads, len_q, d_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.h * self.d_k)
        output = self.fc(context)
        # output: [batch_size, len_q, d_model]
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        output = self.feed_forward(inputs)
        return output


class sacffn(nn.Module):
    def __init__(self, in_features, out_features):
        super(sacffn, self).__init__()
        self.weight1 = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias1 = torch.nn.Parameter(torch.Tensor(out_features))

        self.weight2 = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias2 = torch.nn.Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight1, 1, 1000)
        torch.nn.init.uniform_(self.weight2, 1, 1000)

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight1)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias1, -bound, bound)

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight2)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias2, -bound, bound)

        return

    def forward(self, input):
        # output = torch.addmm(self.bias1, inputs, self.weight1.t())
        output1 = input.matmul(self.weight1.t())
        output1 += self.bias1

        output2 = output1.matmul(self.weight2.t())
        output2 += self.bias2
        ret = output2
        return ret
