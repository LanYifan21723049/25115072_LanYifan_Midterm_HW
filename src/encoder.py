#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer Encoder Implementation
"""
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer
    
    Structure:
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Position-wise Feed-Forward
    4. Add & Norm
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len] padding mask
        Returns:
            output: [batch_size, seq_len, d_model]
            attention: [batch_size, n_heads, seq_len, seq_len]
        """
        # Self-Attention + Residual + Norm
        attn_output, attention = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        output = self.norm2(x + ff_output)
        
        return output, attention


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder
    
    Composed of N EncoderLayers stacked together
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, 
                 max_len=5000, dropout=0.1, pad_idx=0):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_layers: Number of encoder layers
            max_len: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding token index
        """
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        from positional_encoding import PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # N Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len] token indices
            mask: [batch_size, 1, seq_len] padding mask
        Returns:
            output: [batch_size, seq_len, d_model]
            attentions: list of attention weights from each layer
        """
        # Embedding + scaling
        x = self.embedding(x) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through N Encoder layers
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, mask)
            attentions.append(attention)
        
        return x, attentions
    
    def make_padding_mask(self, x):
        """
        Create padding mask
        
        Args:
            x: [batch_size, seq_len]
        Returns:
            mask: [batch_size, 1, seq_len]
        """
        # Positions with pad_idx are 0, others are 1
        mask = (x != self.pad_idx).unsqueeze(1)  # [batch_size, 1, seq_len]
        return mask

