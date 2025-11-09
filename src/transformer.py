#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Transformer Model
"""
import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Complete Transformer Model (Encoder-Decoder Architecture)
    
    For sequence-to-sequence tasks like machine translation, text summarization, etc.
    """
    def __init__(self, 
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 n_heads=8,
                 d_ff=2048,
                 n_encoder_layers=6,
                 n_decoder_layers=6,
                 max_len=5000,
                 dropout=0.1,
                 src_pad_idx=0,
                 tgt_pad_idx=0):
        """
        Args:
            src_vocab_size: Source language vocabulary size
            tgt_vocab_size: Target language vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            max_len: Maximum sequence length
            dropout: Dropout rate
            src_pad_idx: Source padding token index
            tgt_pad_idx: Target padding token index
        """
        super(Transformer, self).__init__()
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_encoder_layers,
            max_len=max_len,
            dropout=dropout,
            pad_idx=src_pad_idx
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_decoder_layers,
            max_len=max_len,
            dropout=dropout,
            pad_idx=tgt_pad_idx
        )
        
        # Parameter initialization
        self._init_parameters()
        
    def _init_parameters(self):
        """Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt):
        """
        Args:
            src: [batch_size, src_seq_len] source sequence
            tgt: [batch_size, tgt_seq_len] target sequence
        Returns:
            output: [batch_size, tgt_seq_len, tgt_vocab_size]
            encoder_attentions: encoder attention weights
            decoder_self_attentions: decoder self-attention weights
            decoder_cross_attentions: decoder cross-attention weights
        """
        # Create masks
        src_mask = self.encoder.make_padding_mask(src)
        tgt_mask = self.decoder.make_combined_mask(tgt)
        
        # Encoder
        encoder_output, encoder_attentions = self.encoder(src, src_mask)
        
        # Decoder
        output, decoder_self_attentions, decoder_cross_attentions = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        
        return output, encoder_attentions, decoder_self_attentions, decoder_cross_attentions
    
    def encode(self, src):
        """Execute encoding only"""
        src_mask = self.encoder.make_padding_mask(src)
        encoder_output, _ = self.encoder(src, src_mask)
        return encoder_output
    
    def decode(self, tgt, encoder_output, src_mask=None):
        """Execute decoding only"""
        tgt_mask = self.decoder.make_combined_mask(tgt)
        output, _, _ = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return output
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerForLanguageModeling(nn.Module):
    """
    Transformer for Language Modeling (Encoder-only)
    
    Suitable for text classification, language modeling, etc.
    """
    def __init__(self,
                 vocab_size,
                 d_model=512,
                 n_heads=8,
                 d_ff=2048,
                 n_layers=6,
                 max_len=5000,
                 dropout=0.1,
                 pad_idx=0):
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
        super(TransformerForLanguageModeling, self).__init__()
        
        self.pad_idx = pad_idx
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Language modeling head (project to vocabulary)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Parameter initialization
        self._init_parameters()
        
    def _init_parameters(self):
        """Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            output: [batch_size, seq_len, vocab_size]
            attentions: list of attention weights
        """
        # Create padding mask
        mask = self.encoder.make_padding_mask(x)
        
        # Encoder
        encoder_output, attentions = self.encoder(x, mask)
        
        # Project to vocabulary
        output = self.lm_head(encoder_output)
        
        return output, attentions
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

