#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Head Self-Attention Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, n_heads, seq_len, d_k]
            key: [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_v]
            mask: [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        Returns:
            output: [batch_size, n_heads, seq_len, d_v]
            attention: [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # scores: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (for padding and decoder causal mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Weighted sum
        output = torch.matmul(attention, value)
        
        return output, attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Linear projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len] or [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
            attention: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # Linear projection and split into multiple heads
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k]
        # -> [batch_size, n_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Adjust mask shape for multi-head
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, 1, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            elif mask.dim() == 4:  # Already correct shape
                pass
            else:
                raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
        
        # Apply attention
        output, attention = self.attention(Q, K, V, mask)
        
        # Merge multiple heads
        # [batch_size, n_heads, seq_len, d_v] -> [batch_size, seq_len, n_heads, d_v]
        # -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output, attention

