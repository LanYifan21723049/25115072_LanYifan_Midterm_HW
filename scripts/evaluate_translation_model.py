# -*- coding: utf-8 -*-
"""
Comprehensive Translation Model Evaluation
Includes Loss, Perplexity, and other evaluation metrics
"""
import torch
import torch.nn as nn
import sys
import os
import json
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transformer import Transformer
from data_utils import create_translation_dataloaders


def calculate_perplexity(loss):
    """Calculate perplexity"""
    return np.exp(loss)


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc='Testing'):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output, _, _, _ = model(src, tgt_input)
            
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            
            # Only count non-padding tokens
            non_pad_tokens = (tgt_output != model.tgt_pad_idx).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def greedy_translate(model, src, src_vocab, tgt_vocab, max_len=100, device='cpu'):
    """Greedy decoding translation"""
    model.eval()
    
    # Reverse vocabulary (idx->token)
    idx_to_tgt = {v: k for k, v in tgt_vocab.items()}
    
    src = src.to(device)
    
    with torch.no_grad():
        # Encode
        encoder_output = model.encode(src)
        
        # Initialize decoder input
        sos_idx = tgt_vocab['<sos>']
        eos_idx = tgt_vocab['<eos>']
        tgt = torch.LongTensor([[sos_idx]]).to(device)
        
        # Generate step by step
        for _ in range(max_len):
            src_mask = model.encoder.make_padding_mask(src)
            output = model.decode(tgt, encoder_output, src_mask)
            
            # Get prediction from last position
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            if next_token.item() == eos_idx:
                break
            
            tgt = torch.cat([tgt, next_token], dim=1)
    
    # Convert to text
    tgt_tokens = tgt.squeeze(0).cpu().numpy()[1:]  # Skip <sos>
    translation = ' '.join([idx_to_tgt.get(idx, '<unk>') for idx in tgt_tokens if idx != eos_idx])
    
    return translation


def translate_samples(model, test_loader, src_vocab, tgt_vocab, device, num_samples=5):
    """Translate some sample sentences"""
    model.eval()
    
    # Reverse vocabularies
    idx_to_src = {v: k for k, v in src_vocab.items()}
    idx_to_tgt = {v: k for k, v in tgt_vocab.items()}
    
    print("\n" + "="*70)
    print("  Translation Examples")
    print("="*70)
    
    samples = []
    count = 0
    
    for src_batch, tgt_batch in test_loader:
        if count >= num_samples:
            break
        
        for i in range(min(src_batch.size(0), num_samples - count)):
            src = src_batch[i:i+1]
            tgt = tgt_batch[i:i+1]
            
            # Source sentence
            src_tokens = src.squeeze(0).cpu().numpy()
            src_text = ' '.join([idx_to_src.get(idx, '<unk>') for idx in src_tokens 
                                if idx not in [src_vocab['<pad>'], src_vocab['<sos>'], src_vocab['<eos>']]])
            
            # Reference translation
            tgt_tokens = tgt.squeeze(0).cpu().numpy()
            ref_text = ' '.join([idx_to_tgt.get(idx, '<unk>') for idx in tgt_tokens 
                                if idx not in [tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>']]])
            
            # Model translation
            translation = greedy_translate(model, src, src_vocab, tgt_vocab, device=device)
            
            print(f"\nSample {count + 1}:")
            print(f"  Source:      {src_text[:80]}...")
            print(f"  Reference:   {ref_text[:80]}...")
            print(f"  Translation: {translation[:80]}...")
            
            samples.append({
                'source': src_text,
                'reference': ref_text,
                'translation': translation
            })
            
            count += 1
            if count >= num_samples:
                break
    
    return samples


def main():
    print("\n" + "="*70)
    print("  Translation Model Comprehensive Evaluation")
    print("="*70)
    
    # Configuration
    model_dir = 'results/translation'
    checkpoint_name = 'best_model.pt'
    batch_size = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load checkpoint first
    checkpoint_path = os.path.join(model_dir, 'checkpoints', checkpoint_name)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get config from checkpoint, otherwise load from file
    if 'config' in checkpoint and checkpoint['config'] is not None and isinstance(checkpoint['config'], dict):
        config = checkpoint['config']
        print("Using config from checkpoint")
    else:
        print(f"Loading config from {model_dir}/config.json...")
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        print("Warning: Config from file may be incomplete")
    
    # Ensure all required fields exist with default values
    default_config = {
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 1024,
        'n_encoder_layers': 3,
        'n_decoder_layers': 3,
        'max_len': 100,
        'data_path': 'data/iwslt2017',
        'src_lang': 'en',
        'tgt_lang': 'de'
    }
    
    # Merge with defaults
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
            print(f"  Using default {key}: {value}")
    
    print(f"\nModel configuration:")
    print(f"  d_model: {config['d_model']}")
    print(f"  n_heads: {config['n_heads']}")
    print(f"  d_ff: {config['d_ff']}")
    print(f"  n_layers: {config['n_encoder_layers']}")
    print(f"  vocab_size: {config['src_vocab_size']} (src), {config['tgt_vocab_size']} (tgt)")
    
    # Load data
    print(f"\nLoading test data from {config['data_path']}...")
    _, test_loader, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = \
        create_translation_dataloaders(
            config['data_path'],
            config['src_lang'],
            config['tgt_lang'],
            batch_size=batch_size,
            max_len=config['max_len'],
            test_mode=True
        )
    
    print(f"Test set size: {len(test_loader.dataset)} samples")
    
    # Create model
    print("\nCreating model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_encoder_layers=config['n_encoder_layers'],
        n_decoder_layers=config['n_decoder_layers'],
        max_len=config['max_len'],
        dropout=0.0,  # No dropout during evaluation
        src_pad_idx=src_vocab['<pad>'],
        tgt_pad_idx=tgt_vocab['<pad>']
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Load model weights (checkpoint already loaded above)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    
    print("\n" + "="*70)
    print("  Evaluating Performance Metrics")
    print("="*70)
    
    test_loss, test_perplexity = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\n{'Metric':<20} {'Value':<15} {'Interpretation'}")
    print("-" * 70)
    print(f"{'Test Loss':<20} {test_loss:<15.4f} {'Lower is better'}")
    print(f"{'Test Perplexity':<20} {test_perplexity:<15.4f} {'Lower is better (avg ~7 words)'}")
    
    # Translation examples
    samples = translate_samples(model, test_loader, src_vocab, tgt_vocab, device, num_samples=5)
    
    # Save results
    results = {
        'checkpoint': checkpoint_name,
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'test_loss': test_loss,
        'test_perplexity': test_perplexity,
        'model_parameters': model.count_parameters(),
        'config': config,
        'translation_samples': samples
    }
    
    results_path = os.path.join(model_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("  Evaluation Summary")
    print("="*70)
    print(f"\nModel: {checkpoint_name} (Epoch {checkpoint['epoch']})")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"\nPerformance:")
    print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Test Loss:       {test_loss:.4f}")
    print(f"  Test Perplexity: {test_perplexity:.4f}")
    print(f"\nQuality Assessment:")
    if test_perplexity < 10:
        print(f"  ????? Excellent! (Perplexity < 10)")
    elif test_perplexity < 15:
        print(f"  ???? Very Good (Perplexity < 15)")
    elif test_perplexity < 20:
        print(f"  ??? Good (Perplexity < 20)")
    else:
        print(f"  ?? Fair (Consider more training)")
    
    print(f"\nResults saved to: {results_path}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
