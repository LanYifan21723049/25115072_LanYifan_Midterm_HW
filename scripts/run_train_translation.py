# -*- coding: utf-8 -*-
"""
Run Translation Model Training in IDE
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class Args:
    # Data parameters
    data_path = 'data/iwslt2017'
    src_lang = 'en'
    tgt_lang = 'de'
    
    # Model parameters
    d_model = 256  # Increased for 50K dataset
    n_heads = 8    # Increased
    d_ff = 1024    # Increased
    n_encoder_layers = 3  # Increased
    n_decoder_layers = 3  # Increased
    dropout = 0.1
    max_len = 100
    
    # Training parameters
    batch_size = 32  # Larger batch for better gradient estimates
    epochs = 30
    lr = 0.0004  # Higher learning rate for faster convergence
    warmup_steps = 2000  # Longer warmup for stability
    grad_clip = 1.0
    seed = 42
    
    # Other parameters
    output_dir = 'results/translation'
    save_every = 5
    num_workers = 0

if __name__ == '__main__':
    print("="*60)
    print("  Starting Translation Model Training (Encoder-Decoder)")
    print("="*60)
    
    from train_translation import main
    
    args = Args()
    main(args)
    
    print("\nTraining completed!")