#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Positional Encoding Implementation
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Advantages:
    1. Can extend to any sequence length
    2. Relative position information can be represented by linear function
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
        # Compute denominator term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin and cos functions
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions use sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions use cos
        
        # Add batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a model parameter, but saved in state_dict)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding
    
    Unlike fixed sinusoidal encoding, this encoding is learned through training.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embedding
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

