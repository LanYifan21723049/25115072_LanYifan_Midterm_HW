#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run PTB Training with optimal settings
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_lm import main
import argparse

if __name__ == '__main__':
    print("="*70)
    print("  Starting PTB Language Model Training")
    print("="*70)
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train Transformer on PTB')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ptb',
                        help='Path to PTB dataset')
    parser.add_argument('--data_format', type=str, default='ptb',
                        help='Dataset format')
    
    # Model parameters (optimized for PTB)
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=128,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Learning rate (for Noam scheduler)')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Warmup steps for learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='results/ptb_model',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data_path}")
    print(f"  Model: d_model={args.d_model}, n_layers={args.n_layers}")
    print(f"  Training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print()
    
    # Start training
    main(args)
