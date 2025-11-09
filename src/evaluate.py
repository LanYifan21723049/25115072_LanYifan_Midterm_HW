#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation and Inference Script
"""
import torch
import torch.nn as nn
import argparse
import os
import json
from tqdm import tqdm
import numpy as np

from transformer import Transformer, TransformerForLanguageModeling
from data_utils import create_translation_dataloaders, create_lm_dataloaders


def calculate_perplexity(loss):
    """Calculate perplexity"""
    return np.exp(loss)


def evaluate_translation_model(model, dataloader, criterion, device):
    """Evaluate translation model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc='Evaluating'):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output, _, _, _ = model(src, tgt_input)
            
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            # Calculate loss, but do not count padding tokens
            loss = criterion(output, tgt_output)
            
            # Count non-padding tokens
            non_pad_tokens = (tgt_output != model.tgt_pad_idx).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def evaluate_lm_model(model, dataloader, criterion, device):
    """Evaluate language model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            x = batch.to(device)
            
            # Input and target (shifted by 1)
            x_input = x[:, :-1]
            x_target = x[:, 1:]
            
            output, _ = model(x_input)
            
            output = output.reshape(-1, output.shape[-1])
            x_target = x_target.reshape(-1)
            
            loss = criterion(output, x_target)
            
            # Count non-padding tokens
            non_pad_tokens = (x_target != model.pad_idx).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=50, device='cpu'):
    """Greedy decoding (for translation task)"""
    model.eval()
    
    src = src.to(device)
    
    # Encode
    encoder_output = model.encode(src)
    
    # Initialize decoder input (starting with <sos>)
    sos_idx = tgt_vocab['<sos>']
    eos_idx = tgt_vocab['<eos>']
    
    tgt = torch.LongTensor([[sos_idx]]).to(device)
    
    for i in range(max_len):
        # Decode
        src_mask = model.encoder.make_padding_mask(src)
        output = model.decode(tgt, encoder_output, src_mask)
        
        # Get prediction from the last position
        next_token_logits = output[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # If <eos> is predicted, stop
        if next_token.item() == eos_idx:
            break
        
        # Append the predicted token to the sequence
        tgt = torch.cat([tgt, next_token], dim=1)
    
    return tgt.squeeze(0).cpu().numpy()


def generate_text(model, prompt, vocab, idx_to_token, max_len=100, device='cpu'):
    """Generate text (for language model)"""
    model.eval()
    
    # Convert prompt to token indices
    tokens = prompt.split()
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    x = torch.LongTensor([indices]).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            output, _ = model(x)
            
            # Get prediction from the last position
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # If <eos> is predicted, stop
            if next_token.item() == vocab['<eos>']:
                break
            
            x = torch.cat([x, next_token], dim=1)
    
    # Convert back to text
    generated_indices = x.squeeze(0).cpu().numpy()
    generated_text = ' '.join([idx_to_token[idx] for idx in generated_indices])
    
    return generated_text


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open(os.path.join(args.model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    print("Loading data...")
    if args.task == 'translation':
        # Load translation data
        _, test_loader, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = \
            create_translation_dataloaders(
                config['data_path'],
                config['src_lang'],
                config['tgt_lang'],
                batch_size=args.batch_size,
                max_len=config['max_len'],
                test_mode=True
            )
        
        # Create model
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            n_encoder_layers=config['n_encoder_layers'],
            n_decoder_layers=config['n_decoder_layers'],
            max_len=config['max_len'],
            dropout=0.0,  # Do not use dropout during evaluation
            src_pad_idx=src_vocab['<pad>'],
            tgt_pad_idx=tgt_vocab['<pad>']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
        
    else:  # language modeling
        # Load language model data
        _, test_loader, vocab_size, vocab, idx_to_token = \
            create_lm_dataloaders(
                config['data_path'],
                batch_size=args.batch_size,
                max_len=config['max_len'],
                test_mode=True
            )
        
        # Create model
        model = TransformerForLanguageModeling(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            n_layers=config['n_layers'],
            max_len=config['max_len'],
            dropout=0.0,
            pad_idx=vocab['<pad>']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    # Load model weights
    checkpoint_path = os.path.join(args.model_dir, 'checkpoints', args.checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Evaluate
    print("\nEvaluating...")
    if args.task == 'translation':
        loss, perplexity = evaluate_translation_model(model, test_loader, criterion, device)
    else:
        loss, perplexity = evaluate_lm_model(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Perplexity: {perplexity:.4f}")
    
    # Save results
    results = {
        'test_loss': loss,
        'test_perplexity': perplexity,
        'checkpoint': args.checkpoint,
        'task': args.task
    }
    
    results_path = os.path.join(args.model_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Transformer Model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                        help='Checkpoint filename')
    parser.add_argument('--task', type=str, choices=['translation', 'lm'], required=True,
                        help='Task type')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    main(args)