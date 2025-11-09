#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run PTB Training (Fresh Import - No Cache)

"""
import sys
import os

# Clear any cached modules
if 'data_utils' in sys.modules:
    del sys.modules['data_utils']
if 'train_lm' in sys.modules:
    del sys.modules['train_lm']

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("="*70)
print("  PTB Training (Fresh Import - Forcing Reload)")
print("="*70)
print()

# Import fresh modules
from train_lm import main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer on PTB')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ptb')
    parser.add_argument('--data_format', type=str, default='ptb')
    
    # Model parameters (REDUCED for PTB to prevent overfitting)
    parser.add_argument('--d_model', type=int, default=200)  # Reduced from 256
    parser.add_argument('--n_heads', type=int, default=4)    # Reduced from 8
    parser.add_argument('--d_ff', type=int, default=400)     # Reduced from 512
    parser.add_argument('--n_layers', type=int, default=2)   # Reduced from 4
    parser.add_argument('--dropout', type=float, default=0.5)  # Increased from 0.3
    parser.add_argument('--max_len', type=int, default=70)     # Increased back (compromise)
    
    #Training parameters (adjusted for smaller model)
    parser.add_argument('--batch_size', type=int, default=32)  # Back to 32
    parser.add_argument('--epochs', type=int, default=40)      # More epochs for smaller model
    parser.add_argument('--lr', type=float, default=0.5)       # Reduced learning rate
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--grad_clip', type=float, default=0.25)  # Reduced clip
    parser.add_argument('--seed', type=int, default=42)
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='results/ptb_model_fixed')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Dataset: {args.data_path}")
    print(f"  Data format: {args.data_format}")
    print(f"  Model: d_model={args.d_model}, n_layers={args.n_layers}")
    print(f"  Training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print()
    print("use results/ptb_model_fixed to avoid overfit old results")
    print()
    
    # Start training
    main(args)
