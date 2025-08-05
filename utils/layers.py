import torch
from torch import nn

from utils.attention import MultiHeadAttention, FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_target, dropout_rate=0.0, mask_flag=True):
        super(EncoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention(n_heads, d_model, dropout=dropout_rate, mask_flag=mask_flag)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        # self.project = nn.Linear(d_model, d_target)

    def forward(self, input_q, input_k, input_v, enc_attn_mask=None):
        """
        enc_layer_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """

        residual1 = input_q.clone()
        enc_self_attn_outputs = self.enc_attn(input_q, input_k, input_v, enc_attn_mask)
        outputs1 = self.norm1(enc_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        ffn_outputs = self.ffn(outputs1)
        # ffn_outputs: [batch_size, src_len, d_model]
        ffn_outputs = self.dropout(ffn_outputs)
        outputs2 = self.norm2(ffn_outputs + residual2)

        # final_output = self.project(outputs2)

        return outputs2


class Encoder(nn.Module):
    def __init__(self, layer_num, n_heads, d_model, d_ff, d_target, dropout_rate=0.0, mask_flag=True):
        super(Encoder, self).__init__()
        self.first_layer = EncoderLayer(n_heads, d_model, d_ff, d_model, dropout_rate, mask_flag)

        self.layers = nn.ModuleList(
            [EncoderLayer(n_heads, d_model, d_ff, d_model, dropout_rate, mask_flag) for _ in range(layer_num - 2)])

        self.last_layer = EncoderLayer(n_heads, d_model, d_ff, d_target, dropout_rate, mask_flag)

    def forward(self, input_q, input_k, input_v, enc_self_attn_mask=None):
        """
        enc_inputs: [batch_size, src_len]
        """
        input_q = input_q.clone()
        input_k = input_k.clone()
        input_v = input_v.clone()
        enc_outputs = self.first_layer(input_q, input_k, input_v, enc_self_attn_mask)
        enc_outputs = enc_outputs.clone()
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_outputs, enc_outputs, enc_self_attn_mask)
        enc_outputs = self.last_layer(enc_outputs, enc_outputs, enc_outputs, enc_self_attn_mask)
        return enc_outputs


class EncoderLayer_noNorm(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_target, dropout_rate=0.0, mask_flag=True):
        super(EncoderLayer_noNorm, self).__init__()
        self.enc_attn = MultiHeadAttention(n_heads, d_model, dropout=dropout_rate, mask_flag=mask_flag)
        self.ffn = FeedForward(d_model, d_ff)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm1 = nn.Linear(d_model, d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.project = nn.Linear(d_model, d_target)

    def forward(self, input_q, input_k, input_v, enc_attn_mask=None):
        """
        enc_layer_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """

        residual1 = input_q.clone()
        enc_self_attn_outputs = self.enc_attn(input_q, input_k, input_v, enc_attn_mask)
        # outputs1 = self.norm1(enc_self_attn_outputs + residual1)
        outputs1 = enc_self_attn_outputs + residual1

        residual2 = outputs1.clone()
        ffn_outputs = self.ffn(outputs1)
        # ffn_outputs: [batch_size, src_len, d_model]
        ffn_outputs = self.dropout(ffn_outputs)
        # outputs2 = self.norm2(ffn_outputs + residual2)
        outputs2 = ffn_outputs + residual2

        final_output = self.project(outputs2)

        return final_output


class EncoderLayer_origin(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout_rate=0.0, mask_flag=True):
        super(EncoderLayer_origin, self).__init__()
        self.enc_attn = MultiHeadAttention(n_heads, d_model, dropout=dropout_rate, mask_flag=mask_flag)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        # self.project = nn.Linear(d_model, d_target)

    def forward(self, enc_layer_inputs, enc_attn_mask=None):
        """
        enc_layer_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """

        residual1 = enc_layer_inputs.clone()
        enc_self_attn_outputs = self.enc_attn(enc_layer_inputs, enc_layer_inputs, enc_layer_inputs, enc_attn_mask)
        outputs1 = self.norm1(enc_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        ffn_outputs = self.ffn(outputs1)
        # ffn_outputs: [batch_size, src_len, d_model]
        ffn_outputs = self.dropout(ffn_outputs)
        outputs2 = self.norm2(ffn_outputs + residual2)

        # final_output = self.project(outputs2)

        return outputs2


class Encoder_origin(nn.Module):
    def __init__(self, encoder_num, n_heads, d_model, d_ff, dropout_rate=0.0, mask_flag=True):
        super(Encoder_origin, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer_origin(n_heads, d_model, d_ff, dropout_rate=dropout_rate, mask_flag=mask_flag)
             for _ in range(encoder_num)])

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = enc_inputs.clone()
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs

class DecoderLayer_origin(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout_rate=0.0, mask_flag=True):
        super(DecoderLayer_origin, self).__init__()
        self.dec_self_attn = MultiHeadAttention(n_heads, d_model, dropout=dropout_rate, mask_flag=mask_flag)
        self.norm1 = nn.LayerNorm(d_model)
        self.dec_enc_attn = MultiHeadAttention(n_heads, d_model, dropout=dropout_rate, mask_flag=mask_flag)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, dec_layer_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_layer_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        residual1 = dec_layer_inputs.clone()
        dec_self_attn_outputs = self.dec_self_attn(dec_layer_inputs, dec_layer_inputs, dec_layer_inputs,
                                                   dec_self_attn_mask)
        outputs1 = self.norm1(dec_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        dec_enc_attn_outputs = self.dec_enc_attn(outputs1, enc_outputs, enc_outputs, dec_enc_attn_mask)
        outputs2 = self.norm2(dec_enc_attn_outputs + residual2)

        residual3 = outputs2.clone()
        ffn_outputs = self.ffn(outputs2)
        ffn_outputs = self.dropout(ffn_outputs)
        outputs3 = self.norm3(ffn_outputs + residual3)

        return outputs3

class Decoder_origin(nn.Module):
    def __init__(self, encoder_num, n_heads, d_model, d_ff, dropout_rate=0.0, mask_flag=True):
        super(Decoder_origin, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer_origin(n_heads, d_model, d_ff, dropout_rate=dropout_rate, mask_flag=mask_flag)
             for _ in range(encoder_num)])

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_intpus: [batch_size, src_len, d_model]
        enc_outputs: [batsh_size, src_len, d_model]
        """
        dec_outputs = dec_inputs.clone()
        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs


class Fc(nn.Module):
    def __init__(self, d_model):
        super(Fc, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, inputs):
        output = self.fc(inputs)
        return output


class FcLayer(nn.Module):
    def __init__(self, layer_num, d_model):
        super(FcLayer, self).__init__()
        self.fcs = nn.ModuleList([
            Fc(d_model) for _ in range(layer_num)
        ])

    def forward(self, inputs):
        outputs = inputs.clone()
        for fc in self.fcs:
            outputs = fc(outputs)
        return outputs


class FcBlock(nn.Module):
    def __init__(self, layer_num, d_model):
        super(FcBlock, self).__init__()
        self.fc_layer = FcLayer(layer_num, d_model)
        self.project = nn.Linear(d_model, d_model)

    def forward(self, inputs):
        outputs = inputs.clone()
        t = self.fc_layer(inputs)
        t = self.project(t)
        red = outputs - t
        # outputs = torch.matmul(outputs, t)
        return t, red


class FcStack(nn.Module):
    def __init__(self, block_num, layer_num, d_model):
        super(FcStack, self).__init__()
        self.fc_blocks = nn.ModuleList([
            FcBlock(layer_num, d_model) for _ in range(block_num)
        ])

    def forward(self, inputs):
        res = torch.zeros_like(inputs)
        red = inputs.clone()
        for fc_block in self.fc_blocks:
            t, red = fc_block(red)
            res = res + t
        return res
