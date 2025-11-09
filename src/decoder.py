#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer Decoder Implementation
"""
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer
    
    Structure:
    1. Masked Multi-Head Self-Attention
    2. Add & Norm
    3. Multi-Head Cross-Attention (with encoder output)
    4. Add & Norm
    5. Position-wise Feed-Forward
    6. Add & Norm
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super(DecoderLayer, self).__init__()
        
        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention (interact with encoder output)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch_size, tgt_seq_len, d_model]
            encoder_output: [batch_size, src_seq_len, d_model]
            src_mask: [batch_size, 1, src_seq_len] encoder padding mask
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len] decoder causal mask + padding mask
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
            self_attention: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
            cross_attention: [batch_size, n_heads, tgt_seq_len, src_seq_len]
        """
        # Masked Self-Attention + Residual + Norm
        self_attn_output, self_attention = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_output)
        
        # Cross-Attention + Residual + Norm
        # Query from decoder, Key and Value from encoder
        cross_attn_output, cross_attention = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + cross_attn_output)
        
        # Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        output = self.norm3(x + ff_output)
        
        return output, self_attention, cross_attention


class TransformerDecoder(nn.Module):
    """
    Complete Transformer Decoder
    
    Composed of N DecoderLayers stacked together
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_len=5000, dropout=0.1, pad_idx=0):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_layers: Number of decoder layers
            max_len: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding token index
        """
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        from positional_encoding import PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # N Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch_size, tgt_seq_len] target token indices
            encoder_output: [batch_size, src_seq_len, d_model]
            src_mask: [batch_size, 1, src_seq_len] encoder padding mask
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len] decoder mask
        Returns:
            output: [batch_size, tgt_seq_len, vocab_size]
            self_attentions: list of self-attention weights
            cross_attentions: list of cross-attention weights
        """
        # Embedding + scaling
        x = self.embedding(x) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through N Decoder layers
        self_attentions = []
        cross_attentions = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
        
        # Project to vocabulary size
        output = self.fc_out(x)
        
        return output, self_attentions, cross_attentions
    
    def make_causal_mask(self, tgt_len, device):
        """
        Create causal mask (lower triangular matrix) to prevent decoder from seeing future tokens
        
        Args:
            tgt_len: Target sequence length
            device: torch device
        Returns:
            mask: [tgt_len, tgt_len]
        """
        mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device))
        return mask
    
    def make_padding_mask(self, x):
        """
        Create padding mask
        
        Args:
            x: [batch_size, seq_len]
        Returns:
            mask: [batch_size, 1, seq_len]
        """
        mask = (x != self.pad_idx).unsqueeze(1)
        return mask
    
    def make_combined_mask(self, tgt):
        """
        Combine causal mask and padding mask
        
        Args:
            tgt: [batch_size, tgt_seq_len]
        Returns:
            mask: [batch_size, tgt_seq_len, tgt_seq_len]
        """
        batch_size, tgt_len = tgt.size()
        device = tgt.device
        
        # Causal mask: [tgt_len, tgt_len]
        causal_mask = self.make_causal_mask(tgt_len, device)
        
        # Padding mask: [batch_size, 1, tgt_len]
        padding_mask = self.make_padding_mask(tgt)
        
        # Combine two masks: [batch_size, tgt_len, tgt_len]
        # Convert to bool for bitwise operations
        combined_mask = causal_mask.unsqueeze(0).bool() & padding_mask.unsqueeze(1).bool()
        
        return combined_mask

