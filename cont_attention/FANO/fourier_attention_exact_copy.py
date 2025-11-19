"""
Fourier Attention Neural Operator - EXACT COPY
This file contains the EXACT code from models/transformer_custom.py, word-for-word,
with no modifications except for the necessary imports.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TransformerEncoder_Operator(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder_Operator, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class SpectralConv2d_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, nhead):
        super(SpectralConv2d_Attention, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.nhead = nhead
        self.scale = (1 / (in_channels * out_channels))
        #use torch.view_as_real if using ddp
        #self.weights1 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))
        #self.weights2 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))
        self.weights1 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, num_patches, in_channel, x,y ), (in_channel, out_channel, x,y, nhead) -> (batch, num_patches, out_channel, x,y, nhead)
        return torch.einsum("bnixy,ioxyh->bnoxyh", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        num_patches = x.shape[1]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #####
        x = torch.permute(x, (0,1,4,2,3))
        #x is of shape (batch, num_patches, d_model, x, y)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, num_patches, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, self.nhead, dtype=torch.cfloat, device=x.device)
        #use torch.view_as_complex if using ddp
        #out_ft[:, :, :, :self.modes1, :self.modes2, :] = \
        #    self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], torch.view_as_complex(self.weights1))
        #out_ft[:, :, :, -self.modes1:, :self.modes2, :] = \
        #    self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], torch.view_as_complex(self.weights2))
        out_ft[:, :, :, :self.modes1, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-3, -2))
        #####
        x = torch.permute(x, (0,1,3,4,2,5))
        #x is of shape (batch, num_patches, x, y, d_model, nhead)
        return x

class MultiheadAttention_Conv(nn.Module):

    def __init__(self, d_model, nhead, modes1, modes2, im_size, dropout=0.1):
        super(MultiheadAttention_Conv, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.im_size = im_size

        self.query_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.key_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.value_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)

        self.scaled_dot_product_attention = ScaledDotProductAttention_Conv(d_model, im_size, dropout=dropout)

        self.out_linear = nn.Linear(nhead*self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        batch, num_patches, patch_size, patch_size, d_model = x.size()
        ###should write a split heads function to do the permutation
        query = self.query_operator(x).permute(0,5,1,2,3,4)
        key = self.key_operator(x).permute(0,5,1,2,3,4)
        value = self.value_operator(x).permute(0,5,1,2,3,4)
        # query, key, and value are of shape (batch, nhead, num_patches,x,y , d_model)
        # Scaled Dot Product Attention

        attention_output, _ = self.scaled_dot_product_attention(query, key, value, key_padding_mask=key_padding_mask)

        ######
        ###this should be the combine heads function
        attention_output = attention_output.reshape(batch, num_patches, patch_size, patch_size,-1)
        #####
        output = self.out_linear(attention_output)
        output = self.dropout(output)
        return output

class ScaledDotProductAttention_Conv(nn.Module):

    def __init__(self, d_model, im_size, dropout=0.1):
        super(ScaledDotProductAttention_Conv, self).__init__()
        #d_model* or just d_model?
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([((im_size)**4)])), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # Custom logic for attention calculation
        scores = torch.einsum("bnpxyd,bnqxyd->bnpq", query, key) / self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.einsum("bnpq,bnqxyd->bnpxyd", attention_weights, value)
        output = output.permute(0,2,3,4,5,1)
        return output, attention_weights

class TransformerEncoderLayer_Conv(nn.Module):#

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, modes=None, patch_size=1, im_size=64, batch_first=True):
        super(TransformerEncoderLayer_Conv, self).__init__()
        # Self-attention layer
        if modes is None:
            modes2 = patch_size//2+1
            modes1 = patch_size//2+1
        else:
            modes1 = modes[0]
            modes2 = modes[1]

        self.patch_size = patch_size
        self.im_size = im_size
        self.size_row = im_size
        self.size_col = im_size
        self.d_model = d_model

        #or im_size?
        self.self_attn = MultiheadAttention_Conv(d_model, nhead, modes1, modes2, im_size, dropout=dropout)
        # Feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Activation function

        self.activation = getattr(F, activation)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Whether to normalize before the self-attention layer
        self.norm_first = norm_first
        # Whether to apply layer normalization at the end
        self.do_layer_norm = do_layer_norm


    def forward(self, x, mask=None):
        if self.norm_first:
            x = self.norm1(x)
        # Self-attention
        attn_output = self.self_attn(x, key_padding_mask=mask)
        # Residual connection and layer normalization
        x = x + self.dropout(attn_output)
        if not self.norm_first:
            x = self.norm1(x)
        # Feedforward layer
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Residual connection and layer normalization
        x = x + self.dropout(ff_output)
        if self.do_layer_norm:
            x = self.norm2(x)

        return x

