#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize PTB Training Results
Generate visualization plots for PTB training results
"""
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font for better display (remove Chinese font requirement)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_training_history(result_dir='results/ptb_model_fixed'):
    """Load training history"""
    history_file = os.path.join(result_dir, 'training_history.json')
    
    if not os.path.exists(history_file):
        print(f"[ERROR] Training history file not found: {history_file}")
        return None
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    return history


def plot_training_results(history, output_dir='report/figures'):
    """Generate training results visualization"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    epochs = range(1, len(train_losses) + 1)
    
    # Calculate Perplexity
    train_ppls = [np.exp(loss) for loss in train_losses]
    val_ppls = [np.exp(loss) for loss in val_losses]
    
    # Create main figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # ============= Subplot 1: Loss Curves =============
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, train_losses, 'b-', marker='o', markersize=3, 
             linewidth=2, label='Train Loss', alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', marker='s', markersize=3, 
             linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(epochs) + 1)
    
    # ============= Subplot 2: Perplexity Curves =============
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, train_ppls, 'b-', marker='o', markersize=3, 
             linewidth=2, label='Train PPL', alpha=0.8)
    ax2.plot(epochs, val_ppls, 'r-', marker='s', markersize=3, 
             linewidth=2, label='Validation PPL', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Perplexity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(epochs) + 1)
    
    # ============= Subplot 3: Train vs Val PPL Comparison =============
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(train_ppls, val_ppls, c=epochs, cmap='viridis', 
                s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    # Add diagonal line (Train = Val)
    max_ppl = max(max(train_ppls), max(val_ppls))
    min_ppl = min(min(train_ppls), min(val_ppls))
    ax3.plot([min_ppl, max_ppl], [min_ppl, max_ppl], 'k--', 
             linewidth=1.5, alpha=0.5, label='Train = Val')
    ax3.set_xlabel('Train PPL', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Validation PPL', fontsize=12, fontweight='bold')
    ax3.set_title('Train vs Validation PPL', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Epoch', fontsize=10)
    
    # ============= Subplot 4: First 10 Epochs Detail =============
    ax4 = plt.subplot(2, 3, 4)
    early_epochs = min(10, len(epochs))
    ax4.plot(epochs[:early_epochs], val_ppls[:early_epochs], 
             'ro-', linewidth=2.5, markersize=8, label='Validation PPL')
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Validation Perplexity', fontsize=12, fontweight='bold')
    ax4.set_title('Convergence in First 10 Epochs', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    # Annotate key points
    for i in [0, 4, 9]:
        if i < early_epochs:
            ax4.annotate(f'Epoch {i+1}\nPPL: {val_ppls[i]:.1f}',
                        xy=(i+1, val_ppls[i]),
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=9)
    
    # ============= Subplot 5: Loss Improvement Rate =============
    ax5 = plt.subplot(2, 3, 5)
    val_loss_improvements = [0] + [(val_losses[i-1] - val_losses[i]) / val_losses[i-1] * 100 
                                    for i in range(1, len(val_losses))]
    colors = ['green' if x > 0 else 'red' for x in val_loss_improvements]
    ax5.bar(epochs, val_loss_improvements, color=colors, alpha=0.7, edgecolor='black')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Loss Improvement Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Validation Loss Improvement per Epoch', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xlim(0, len(epochs) + 1)
    
    # ============= Subplot 6: Training Statistics Summary =============
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Prepare statistics
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    best_val_ppl = np.exp(best_val_loss)
    final_val_ppl = val_ppls[-1]
    final_train_ppl = train_ppls[-1]
    
    # Calculate improvement
    initial_val_ppl = val_ppls[0]
    improvement = (initial_val_ppl - final_val_ppl) / initial_val_ppl * 100
    
    summary_text = f"""
    ═══════════════════════════════
        Training Statistics
    ═══════════════════════════════
    
    Total Epochs: {len(epochs)} epochs
    
    Best Validation PPL: {best_val_ppl:.2f}
    (at Epoch {best_epoch})
    
    Final Validation PPL: {final_val_ppl:.2f}
    Final Training PPL: {final_train_ppl:.2f}
    
    Overall Improvement: {improvement:.1f}%
    (from {initial_val_ppl:.1f} to {final_val_ppl:.1f})
    
    Overfitting Check:
    Train/Val Ratio: {final_train_ppl/final_val_ppl:.3f}
    {'OK - Normal' if 0.9 < final_train_ppl/final_val_ppl < 1.2 else 'Warning - Potential Issue'}
    
    Convergence Status:
    Last 5 Epochs PPL Std: {np.std(val_ppls[-5:]):.2f}
    {'OK - Converged' if np.std(val_ppls[-5:]) < 50 else 'Warning - Not Fully Converged'}
    """
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'ptb_training_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training results visualization saved to: {output_path}")
    
    # Close figure to free memory
    plt.close()
    
    return output_path


def plot_simple_comparison(history, output_dir='report/figures'):
    """Generate simplified comparison plot (for reports)"""
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    epochs = range(1, len(train_losses) + 1)
    
    train_ppls = [np.exp(loss) for loss in train_losses]
    val_ppls = [np.exp(loss) for loss in val_losses]
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-o', linewidth=2.5, markersize=4, 
             label='Training', alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-s', linewidth=2.5, markersize=4, 
             label='Validation', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training Process - Loss', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=13, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Perplexity curves
    ax2.plot(epochs, train_ppls, 'b-o', linewidth=2.5, markersize=4, 
             label='Training', alpha=0.8)
    ax2.plot(epochs, val_ppls, 'r-s', linewidth=2.5, markersize=4, 
             label='Validation', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
    ax2.set_title('Training Process - Perplexity', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=13, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'ptb_training_simple.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Simplified training curves saved to: {output_path}")
    
    # Close figure to free memory
    plt.close()
    
    return output_path


def main():
    print("="*70)
    print("  PTB Training Results Visualization")
    print("="*70)
    print()
    
    # Load training history
    history = load_training_history('results/ptb_model_fixed')
    
    if history is None:
        print("Unable to load training history. Please check if training is complete.")
        return
    
    print(f"✓ Successfully loaded training history")
    print(f"  - Total epochs: {len(history['train_losses'])} epochs")
    print(f"  - Final validation loss: {history['val_losses'][-1]:.4f}")
    print(f"  - Final validation PPL: {np.exp(history['val_losses'][-1]):.2f}")
    print()
    
    # Generate detailed visualization
    print("Generating detailed visualization...")
    plot_training_results(history)
    print()
    
    # Generate simplified version
    print("Generating simplified version (for reports)...")
    plot_simple_comparison(history)
    print()
    
    print("="*70)
    print("  ✓ All visualization plots generated successfully!")
    print("="*70)
    print()
    print("Generated files:")
    print("  1. report/figures/ptb_training_results.png  - Detailed 6-subplot analysis")
    print("  2. report/figures/ptb_training_simple.png   - Simplified 2-subplot (for reports)")
    print()


if __name__ == '__main__':
    main()
