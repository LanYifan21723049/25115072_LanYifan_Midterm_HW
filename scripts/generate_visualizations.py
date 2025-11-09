#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Visualizations for Report
Creates all figures needed for the final LaTeX report
"""
import sys
import os
import json

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


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
    Plot model architecture summary
    
    Args:
        config: Model configuration dict
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


def generate_actual_training_curves(save_dir):
    """Generate training curves from actual results"""
    # Load actual training history
    history_file = os.path.join(parent_dir, 'results', 'translation', 'training_history.json')
    
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found, generating example curves")
        generate_example_training_curves(save_dir)
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    epochs = list(range(1, len(train_losses) + 1))
    
    # Loss curve
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', 
             linewidth=2.5, markersize=6, alpha=0.8, color='#2E86AB')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', 
             linewidth=2.5, markersize=6, alpha=0.8, color='#A23B72')
    
    # Mark best validation loss
    best_val_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_idx]
    plt.plot(epochs[best_val_idx], best_val_loss, 'r*', 
             markersize=20, label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {epochs[best_val_idx]})')
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
    plt.title('Translation Model Training Progress (30 Epochs)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'main_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Perplexity curve
    train_perplexity = [np.exp(loss) for loss in train_losses]
    val_perplexity = [np.exp(loss) for loss in val_losses]
    
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_perplexity, label='Train Perplexity', marker='o', 
             linewidth=2.5, markersize=6, alpha=0.8, color='#2E86AB')
    plt.plot(epochs, val_perplexity, label='Validation Perplexity', marker='s', 
             linewidth=2.5, markersize=6, alpha=0.8, color='#A23B72')
    
    # Mark best validation perplexity
    best_val_ppl = val_perplexity[best_val_idx]
    plt.plot(epochs[best_val_idx], best_val_ppl, 'r*', 
             markersize=20, label=f'Best Val PPL: {best_val_ppl:.2f} (Epoch {epochs[best_val_idx]})')
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Perplexity', fontsize=14, fontweight='bold')
    plt.title('Translation Model Perplexity Progress', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'main_training_perplexity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("? Actual training curves generated!")
    print(f"  - Best validation loss: {best_val_loss:.4f} at epoch {epochs[best_val_idx]}")
    print(f"  - Best validation perplexity: {best_val_ppl:.2f}")

def generate_example_training_curves(save_dir):
    """Fallback: Generate example training curves"""
    epochs = 30
    train_losses = [4.5 - i * 0.035 + np.random.rand() * 0.1 for i in range(epochs)]
    val_losses = [4.8 - i * 0.03 + np.random.rand() * 0.15 for i in range(epochs)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss (Example)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'main_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Example training curves generated!")


def generate_attention_visualization(save_path):
    """Generate example attention patterns"""
    seq_len = 10
    n_heads = 8
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head_idx in range(n_heads):
        # Generate different attention patterns
        if head_idx < 2:
            # Local attention pattern
            attn = np.eye(seq_len) + np.eye(seq_len, k=1) * 0.5 + np.eye(seq_len, k=-1) * 0.5
        elif head_idx < 4:
            # Global attention pattern
            attn = np.ones((seq_len, seq_len)) / seq_len
        elif head_idx < 6:
            # Focus on start/end
            attn = np.random.rand(seq_len, seq_len) * 0.1
            attn[:, 0] = 0.5
            attn[:, -1] = 0.5
        else:
            # Random pattern
            attn = np.random.rand(seq_len, seq_len)
        
        # Normalize
        attn = attn / attn.sum(axis=1, keepdims=True)
        
        sns.heatmap(attn, cmap='viridis', cbar=True, ax=axes[head_idx],
                    square=True, vmin=0, vmax=1)
        axes[head_idx].set_title(f'Head {head_idx}', fontsize=10)
        axes[head_idx].set_xlabel('Source', fontsize=8)
        axes[head_idx].set_ylabel('Target', fontsize=8)
    
    plt.suptitle('Multi-Head Attention Patterns (Example)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved attention visualization to {save_path}")


def plot_ablation_full_size(save_dir):
    """Generate plots for full ablation experiments (15 epochs, 200K data)"""
    ablation_dir = os.path.join(parent_dir, 'results', 'ablation_full', 'ablation_size')
    
    if not os.path.exists(ablation_dir):
        print("Warning: Full ablation results not found, skipping...")
        return
    
    # Model configurations
    model_configs = [
        ('d256_h8_l3_dr0.1', 'd=256 (11.7M)', 256, 11.7),
        ('d512_h8_l3_dr0.1', 'd=512 (46.5M)', 512, 46.5),
        ('d768_h12_l3_dr0.1', 'd=768 (104M)', 768, 104.0)
    ]
    
    results = []
    for exp_folder, label, d_model, params_m in model_configs:
        exp_path = os.path.join(ablation_dir, exp_folder)
        history_file = os.path.join(exp_path, 'training_history.json')
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            min_val_loss = min(history['val_losses'])
            results.append({
                'label': label,
                'd_model': d_model,
                'params_m': params_m,
                'val_loss': min_val_loss,
                'train_losses': history['train_losses'],
                'val_losses': history['val_losses']
            })
    
    if not results:
        print("No full ablation results found")
        return
    
    # 1. Bar chart comparing final validation losses
    plt.figure(figsize=(12, 7))
    labels = [r['label'] for r in results]
    val_losses = [r['val_loss'] for r in results]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = plt.bar(range(len(labels)), val_losses, color=colors, 
                   edgecolor='black', linewidth=2, alpha=0.85)
    
    # Add baseline reference line
    plt.axhline(y=1.93, color='orange', linestyle='--', linewidth=2.5, 
                label='Baseline (30 epochs, Val Loss=1.93)', alpha=0.8)
    
    plt.xticks(range(len(labels)), labels, fontsize=13, fontweight='bold')
    plt.ylabel('Best Validation Loss', fontsize=14, fontweight='bold')
    plt.title('Full Ablation: Model Size Impact (200K data, 15 epochs)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.legend(fontsize=12, loc='upper right')
    
    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, val_losses)):
        improvement = ((1.93 - loss) / 1.93) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.3f}\n({improvement:+.1f}%)', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(min(val_losses) * 0.95, max(val_losses) * 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_size_full_15epochs.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("? Full ablation bar chart generated")
    
    # 2. Training curves comparison
    plt.figure(figsize=(14, 8))
    epochs = list(range(1, len(results[0]['train_losses']) + 1))
    
    for i, res in enumerate(results):
        plt.plot(epochs, res['val_losses'], label=res['label'], 
                marker='o', linewidth=2.5, markersize=6, alpha=0.85, color=colors[i])
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Loss', fontsize=14, fontweight='bold')
    plt.title('Full Ablation: Training Progress Comparison (200K data)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=13, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_size_full_curves.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("? Full ablation training curves generated")
    
    # 3. Parameter efficiency chart
    plt.figure(figsize=(12, 7))
    params = [r['params_m'] for r in results]
    improvements = [((1.93 - r['val_loss']) / 1.93) * 100 for r in results]
    
    plt.scatter(params, improvements, s=500, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, res in enumerate(results):
        plt.annotate(res['label'], (params[i], improvements[i]), 
                    fontsize=12, fontweight='bold', ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')
    
    plt.xlabel('Model Parameters (Millions)', fontsize=14, fontweight='bold')
    plt.ylabel('Improvement over Baseline (%)', fontsize=14, fontweight='bold')
    plt.title('Parameter Efficiency Analysis', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_parameter_efficiency.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("? Parameter efficiency chart generated")
    
    print(f"\n? Full ablation visualizations complete!")
    print(f"  Generated 3 plots for full ablation experiments")


def plot_ablation_quick_summary(save_dir):
    """Generate summary plots for quick ablation experiments"""
    ablation_dir = os.path.join(parent_dir, 'results', 'ablation_quick')
    
    if not os.path.exists(ablation_dir):
        print("Warning: Quick ablation results not found, skipping...")
        return
    
    # Read ablation results
    ablation_types = {
        'size': ('Model Size (d_model)', 'd_model'),
        'heads': ('Attention Heads', 'n_heads'),
        'layers': ('Model Depth (Layers)', 'n_layers'),
        'dropout': ('Dropout Rate', 'dropout')
    }
    
    for ablation_name, (title, param_key) in ablation_types.items():
        ablation_path = os.path.join(ablation_dir, f'ablation_{ablation_name}')
        if not os.path.exists(ablation_path):
            continue
        
        results = []
        for exp_folder in os.listdir(ablation_path):
            exp_path = os.path.join(ablation_path, exp_folder)
            history_file = os.path.join(exp_path, 'training_history.json')
            config_file = os.path.join(exp_path, 'config.json')
            
            if os.path.exists(history_file) and os.path.exists(config_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                param_val = config.get(param_key, 'N/A')
                min_val_loss = min(history['val_losses'])
                results.append((param_val, min_val_loss))
        
        if results:
            results.sort(key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))
            params, losses = zip(*results)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(params)), losses, color='skyblue', edgecolor='navy', linewidth=1.5)
            plt.xticks(range(len(params)), [str(p) for p in params], fontsize=12)
            plt.xlabel(title, fontsize=13, fontweight='bold')
            plt.ylabel('Best Validation Loss', fontsize=13, fontweight='bold')
            plt.title(f'Quick Ablation: {title} Impact (5K data, 5 epochs)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, loss) in enumerate(zip(bars, losses)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{loss:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'ablation_quick_{ablation_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"? Quick ablation plot for {ablation_name} generated")

def main():
    # Create visualization directory
    vis_dir = os.path.join(parent_dir, 'report', 'figures')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("="*70)
    print(" Generating Visualizations for Final Report")
    print("="*70)
    
    # 1. Positional encoding visualization
    print("\n[1/6] Generating positional encoding visualization...")
    plot_position_encoding(
        d_model=256,
        max_len=100,
        save_path=os.path.join(vis_dir, 'positional_encoding.png')
    )
    
    # 2. Generate ACTUAL training curves
    print("\n[2/6] Generating actual training curves...")
    generate_actual_training_curves(vis_dir)
    
    # 3. Model architecture summary
    print("\n[3/6] Generating model architecture summary...")
    # Load actual config
    config_file = os.path.join(parent_dir, 'results', 'translation', 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        config['n_encoder_layers'] = 3
        config['n_decoder_layers'] = 3
        config['d_model'] = 256
        config['n_heads'] = 8
        config['d_ff'] = 1024
        config['dropout'] = 0.1
    else:
        config = {
            'd_model': 256,
            'n_heads': 8,
            'd_ff': 1024,
            'n_encoder_layers': 3,
            'n_decoder_layers': 3,
            'dropout': 0.1,
            'src_vocab_size': 8000,
            'tgt_vocab_size': 8000,
            'model_parameters': 11681600
        }
    plot_model_architecture(config, os.path.join(vis_dir, 'architecture_summary.png'))
    
    # 4. Attention patterns
    print("\n[4/6] Generating attention pattern visualization...")
    generate_attention_visualization(os.path.join(vis_dir, 'attention_patterns.png'))
    
    # 5. Quick ablation results
    print("\n[5/6] Generating quick ablation experiment plots...")
    plot_ablation_quick_summary(vis_dir)
    
    # 6. Generate FULL ablation results (15 epochs, 200K data)
    print("\n[6/6] Generating FULL ablation experiment plots...")
    plot_ablation_full_size(vis_dir)
    
    print("\n" + "="*70)
    print(" ? All visualizations generated successfully!")
    print("="*70)
    print(f"\nOutput directory: {vis_dir}")
    print("\nGenerated files:")
    if os.path.exists(vis_dir):
        for filename in sorted(os.listdir(vis_dir)):
            if filename.endswith('.png'):
                print(f"  ? {filename}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

