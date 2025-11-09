#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Tools for Transformer
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os


def plot_attention_weights(attention, layer_idx, head_idx, src_tokens, tgt_tokens, save_path):
    """
    Visualize attention weights
    
    Args:
        attention: [n_heads, tgt_len, src_len] Attention weights
        layer_idx: Layer index
        head_idx: Head index
        src_tokens: List of source sequence tokens
        tgt_tokens: List of target sequence tokens
        save_path: Save path
    """
    plt.figure(figsize=(10, 8))
    
    # Select attention weights for a specific head
    attn = attention[head_idx].cpu().numpy()
    
    # Create heatmap
    sns.heatmap(attn, 
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap='viridis',
                cbar=True,
                square=False)
    
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multi_head_attention(attention, layer_idx, src_tokens, tgt_tokens, save_dir):
    """
    Visualize multi-head attention
    
    Args:
        attention: [n_heads, tgt_len, src_len]
        layer_idx: Layer index
        src_tokens: Source sequence tokens
        tgt_tokens: Target sequence tokens
        save_dir: Save directory
    """
    n_heads = attention.shape[0]
    
    fig, axes = plt.subplots(2, n_heads // 2, figsize=(20, 8))
    axes = axes.flatten()
    
    for head_idx in range(n_heads):
        attn = attention[head_idx].cpu().numpy()
        
        sns.heatmap(attn,
                    xticklabels=src_tokens,
                    yticklabels=tgt_tokens,
                    cmap='viridis',
                    cbar=True,
                    ax=axes[head_idx],
                    square=False)
        
        axes[head_idx].set_title(f'Head {head_idx}')
        axes[head_idx].set_xlabel('Source')
        axes[head_idx].set_ylabel('Target')
    
    plt.suptitle(f'Multi-Head Attention - Layer {layer_idx}', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'multi_head_attention_layer_{layer_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-head attention visualization to {save_path}")


def plot_training_curves(history_path, save_dir):
    """
    Plot training curves
    
    Args:
        history_path: Path to the training history JSON file
        save_dir: Save directory
    """
    import json
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curve to {save_path}")
    
    # Perplexity curve
    train_perplexity = [np.exp(loss) for loss in train_losses]
    val_perplexity = [np.exp(loss) for loss in val_losses]
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_perplexity, label='Train Perplexity', marker='o', linewidth=2)
    plt.plot(val_perplexity, label='Validation Perplexity', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Training and Validation Perplexity', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'perplexity_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved perplexity curve to {save_path}")


def plot_position_encoding(d_model, max_len, save_path):
    """
    Visualize positional encoding
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        save_path: Save path
    """
    import math
    
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pe.T, cmap='RdBu', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Dimension', fontsize=12)
    plt.title('Sinusoidal Positional Encoding', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved positional encoding visualization to {save_path}")


def plot_model_architecture(config, save_path):
    """
    Plot model architecture (simplified)
    
    Args:
        config: Model configuration dictionary
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Parameter text
    text = f"""
Transformer Architecture

Encoder:
  - Layers: {config.get('n_encoder_layers', config.get('n_layers', 'N/A'))}
  - d_model: {config['d_model']}
  - n_heads: {config['n_heads']}
  - d_ff: {config['d_ff']}
  - dropout: {config['dropout']}

Decoder:
  - Layers: {config.get('n_decoder_layers', 'N/A')}
  - d_model: {config['d_model']}
  - n_heads: {config['n_heads']}
  - d_ff: {config['d_ff']}
  - dropout: {config['dropout']}

Vocabulary:
  - Source: {config.get('src_vocab_size', 'N/A')}
  - Target: {config.get('tgt_vocab_size', config.get('vocab_size', 'N/A'))}

Total Parameters: {config.get('model_parameters', 'N/A'):,}
"""
    
    ax.text(0.5, 0.5, text, ha='center', va='center', 
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved architecture summary to {save_path}")


def analyze_attention_patterns(attentions, save_dir):
    """
    Analyze attention pattern statistics
    
    Args:
        attentions: List of attention weights
        save_dir: Save directory
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[0]
    
    # Calculate average attention entropy for each head in each layer
    entropies = []
    
    for layer_idx, attn in enumerate(attentions):
        layer_entropies = []
        for head_idx in range(n_heads):
            head_attn = attn[head_idx].cpu().numpy()
            # Calculate entropy
            entropy = -np.sum(head_attn * np.log(head_attn + 1e-9), axis=-1).mean()
            layer_entropies.append(entropy)
        entropies.append(layer_entropies)
    
    entropies = np.array(entropies)
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(entropies, 
                xticklabels=[f'Head {i}' for i in range(n_heads)],
                yticklabels=[f'Layer {i}' for i in range(n_layers)],
                cmap='viridis',
                cbar=True,
                annot=True,
                fmt='.2f')
    
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    plt.title('Average Attention Entropy by Layer and Head')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'attention_entropy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved attention entropy analysis to {save_path}")


if __name__ == '__main__':
    # Example: Visualize positional encoding
    import os
    os.makedirs('results/visualizations', exist_ok=True)
    
    plot_position_encoding(
        d_model=256,
        max_len=100,
        save_path='results/visualizations/positional_encoding.png'
    )