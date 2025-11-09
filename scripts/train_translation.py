#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Script for Seq2Seq Translation Task (Encoder-Decoder)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformer import Transformer
from data_utils import TranslationDataset, create_translation_dataloaders


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0, scheduler=None):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch_idx, (src, tgt) in enumerate(progress_bar):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # tgt_input is the target sequence without the last token
        # tgt_output is the target sequence without the first token (for loss calculation)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward pass
        output, _, _, _ = model(src, tgt_input)
        
        # Calculate loss
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update learning rate per step (not per epoch)
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc='Evaluating'):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output, _, _, _ = model(src, tgt_input)
            
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, filepath):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def plot_losses(train_losses, val_losses, save_path):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = \
        create_translation_dataloaders(
            args.data_path,
            args.src_lang,
            args.tgt_lang,
            batch_size=args.batch_size,
            max_len=args.max_len,
            num_workers=args.num_workers
        )
    
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    
    # Create model
    print("Creating model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        max_len=args.max_len,
        dropout=args.dropout,
        src_pad_idx=src_vocab['<pad>'],
        tgt_pad_idx=tgt_vocab['<pad>']
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Define loss function (ignore padding token)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Learning rate scheduler - Warmup + Cosine Annealing
    # Warmup: 0 -> peak, then cosine decay to ~0
    total_steps = len(train_loader) * args.epochs
    
    def lr_lambda(step):
        warmup = args.warmup_steps
        if step < warmup:
            # Linear warmup from 0 to 1
            return float(step) / float(max(1, warmup))
        else:
            # Cosine annealing after warmup
            progress = (step - warmup) / (total_steps - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Save configuration
    config = vars(args)
    config['src_vocab_size'] = src_vocab_size
    config['tgt_vocab_size'] = tgt_vocab_size
    config['model_parameters'] = model.count_parameters()
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train (scheduler is called per step inside train_epoch)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip, scheduler)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, config,
                os.path.join(args.output_dir, 'checkpoints', 'best_model.pt')
            )
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, config,
                os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs - 1, train_losses[-1], val_losses[-1], config,
        os.path.join(args.output_dir, 'checkpoints', 'final_model.pt')
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot loss curves
    plot_losses(train_losses, val_losses, os.path.join(args.output_dir, 'loss_curve.png'))
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer for Translation')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/translation',
                        help='Path to dataset')
    parser.add_argument('--src_lang', type=str, default='en',
                        help='Source language')
    parser.add_argument('--tgt_lang', type=str, default='de',
                        help='Target language')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--n_encoder_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--n_decoder_layers', type=int, default=3,
                        help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=100,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='Warmup steps for learning rate')
    parser.addm_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='results/translation',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    main(args)