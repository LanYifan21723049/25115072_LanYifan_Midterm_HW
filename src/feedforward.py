#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Position-wise Feed-Forward Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Two-layer fully connected network applied independently to each position.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (usually 4 * d_model)
            dropout: Dropout rate
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # First layer + ReLU activation
        output = self.linear1(x)
        output = F.relu(output)
        output = self.dropout(output)
        
        # Second layer
        output = self.linear2(output)
        output = self.dropout(output)
        
        return output

