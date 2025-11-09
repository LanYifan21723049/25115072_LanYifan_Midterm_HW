#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Ablation Studies for Translation Model (Small Dataset)
Fast testing with reduced dataset size and shorter training time
"""
import os
import json
import argparse
import subprocess
import sys
from datetime import datetime


def run_experiment(config, output_base_dir):
    """Run a single translation training experiment"""
    exp_name = f"d{config['d_model']}_h{config['n_heads']}_l{config['n_layers']}_dr{config['dropout']}"
    output_dir = os.path.join(output_base_dir, exp_name)
    
    print(f"\n{'='*70}")
    print(f"  Running Experiment: {exp_name}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  d_model: {config['d_model']}")
    print(f"  n_heads: {config['n_heads']}")
    print(f"  d_ff: {config['d_ff']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  dropout: {config['dropout']}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Normalize path separators for Windows
    output_dir_normalized = output_dir.replace('\\', '/')
    data_path_normalized = config['data_path'].replace('\\', '/')
    
    # Create temporary training script with this config
    train_script = f"""
# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class Args:
    # Data parameters
    data_path = r'{data_path_normalized}'
    src_lang = '{config['src_lang']}'
    tgt_lang = '{config['tgt_lang']}'
    
    # Model parameters
    d_model = {config['d_model']}
    n_heads = {config['n_heads']}
    d_ff = {config['d_ff']}
    n_encoder_layers = {config['n_layers']}
    n_decoder_layers = {config['n_layers']}
    dropout = {config['dropout']}
    max_len = {config['max_seq_len']}
    
    # Training parameters
    batch_size = {config['batch_size']}
    epochs = {config['epochs']}
    lr = {config['lr']}
    warmup_steps = {config['warmup_steps']}
    grad_clip = 1.0
    seed = 42
    num_workers = 0
    
    # Output
    output_dir = r'{output_dir_normalized}'
    save_every = {config['save_every']}
    
    # Device
    device = 'cuda'

args = Args()

# Import and run training
from train_translation import main as train_main
train_main(args)
"""
    
    # Write temporary script
    temp_script = f"temp_train_{exp_name}.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(train_script)
    
    try:
        # Run training
        result = subprocess.run(
            [sys.executable, temp_script],
            check=True,
            capture_output=False
        )
        print(f"\nExperiment {exp_name} completed successfully!\n")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"\nExperiment {exp_name} failed with error code {e.returncode}\n")
        success = False
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)
    
    return success


def ablation_attention_heads(output_base_dir, base_config):
    """Ablation study on number of attention heads"""
    print("\n" + "="*70)
    print("  ABLATION STUDY: Number of Attention Heads")
    print("="*70)
    
    head_configs = [2, 4, 8]
    results = []
    
    for n_heads in head_configs:
        config = base_config.copy()
        config['n_heads'] = n_heads
        
        success = run_experiment(config, os.path.join(output_base_dir, 'ablation_heads'))
        results.append({
            'n_heads': n_heads,
            'success': success
        })
    
    return results


def ablation_model_depth(output_base_dir, base_config):
    """Ablation study on model depth (number of layers)"""
    print("\n" + "="*70)
    print("  ABLATION STUDY: Model Depth (Number of Layers)")
    print("="*70)
    
    layer_configs = [2, 3, 4]
    results = []
    
    for n_layers in layer_configs:
        config = base_config.copy()
        config['n_layers'] = n_layers
        
        success = run_experiment(config, os.path.join(output_base_dir, 'ablation_layers'))
        results.append({
            'n_layers': n_layers,
            'success': success
        })
    
    return results


def ablation_dropout(output_base_dir, base_config):
    """Ablation study on dropout rate"""
    print("\n" + "="*70)
    print("  ABLATION STUDY: Dropout Rate")
    print("="*70)
    
    dropout_configs = [0.0, 0.1, 0.2, 0.3]
    results = []
    
    for dropout in dropout_configs:
        config = base_config.copy()
        config['dropout'] = dropout
        
        success = run_experiment(config, os.path.join(output_base_dir, 'ablation_dropout'))
        results.append({
            'dropout': dropout,
            'success': success
        })
    
    return results


def ablation_model_size(output_base_dir, base_config):
    """Ablation study on model size (d_model and d_ff)"""
    print("\n" + "="*70)
    print("  ABLATION STUDY: Model Size")
    print("="*70)
    
    size_configs = [
        {'name': 'tiny', 'd_model': 128, 'd_ff': 512, 'n_heads': 4},
        {'name': 'small', 'd_model': 256, 'd_ff': 1024, 'n_heads': 8},
        {'name': 'medium', 'd_model': 512, 'd_ff': 2048, 'n_heads': 8}
    ]
    results = []
    
    for size_cfg in size_configs:
        config = base_config.copy()
        config['d_model'] = size_cfg['d_model']
        config['d_ff'] = size_cfg['d_ff']
        config['n_heads'] = size_cfg['n_heads']
        
        success = run_experiment(config, os.path.join(output_base_dir, 'ablation_size'))
        results.append({
            'model_size': size_cfg['name'],
            'd_model': size_cfg['d_model'],
            'd_ff': size_cfg['d_ff'],
            'n_heads': size_cfg['n_heads'],
            'success': success
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Quick Ablation Studies for Translation Model')
    parser.add_argument('--data_path', type=str, default='data/iwslt2017_small',
                        help='Path to small dataset')
    parser.add_argument('--ablation_type', type=str, default='heads',
                        choices=['heads', 'layers', 'dropout', 'size', 'all'],
                        help='Type of ablation study to run')
    parser.add_argument('--output_dir', type=str, default='results/ablation_quick',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for each experiment (default: 5 for quick testing)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Base configuration for all experiments
    base_config = {
        'data_path': args.data_path,
        'src_lang': 'en',
        'tgt_lang': 'de',
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 1024,
        'n_layers': 3,
        'dropout': 0.1,
        'max_seq_len': 128,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'warmup_steps': 1000,
        'save_every': args.epochs  # Only save at the end for quick testing
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save base configuration
    with open(os.path.join(args.output_dir, 'base_config.json'), 'w', encoding='utf-8') as f:
        json.dump(base_config, f, indent=2)
    
    print("\n" + "="*70)
    print("  QUICK ABLATION STUDY FOR TRANSLATION MODEL")
    print("="*70)
    print(f"\nBase Configuration:")
    print(f"  Data: {args.data_path}")
    print(f"  Language pair: en-de")
    print(f"  Epochs: {args.epochs} (quick testing)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Base model: d_model={base_config['d_model']}, "
          f"n_heads={base_config['n_heads']}, "
          f"n_layers={base_config['n_layers']}, "
          f"dropout={base_config['dropout']}")
    print(f"\nOutput directory: {args.output_dir}")
    print("="*70)
    
    # Run ablation studies
    all_results = {}
    start_time = datetime.now()
    
    if args.ablation_type in ['heads', 'all']:
        print("\nStarting Attention Heads Ablation...")
        all_results['heads'] = ablation_attention_heads(args.output_dir, base_config)
    
    if args.ablation_type in ['layers', 'all']:
        print("\nStarting Model Depth Ablation...")
        all_results['layers'] = ablation_model_depth(args.output_dir, base_config)
    
    if args.ablation_type in ['dropout', 'all']:
        print("\nStarting Dropout Ablation...")
        all_results['dropout'] = ablation_dropout(args.output_dir, base_config)
    
    if args.ablation_type in ['size', 'all']:
        print("\nStarting Model Size Ablation...")
        all_results['size'] = ablation_model_size(args.output_dir, base_config)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Save results
    results_summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'ablation_type': args.ablation_type,
        'base_config': base_config,
        'results': all_results
    }
    
    results_file = os.path.join(args.output_dir, 'ablation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    
    # Generate summary report
    print("\n" + "="*70)
    print("  ABLATION STUDY COMPLETED")
    print("="*70)
    print(f"\nTotal duration: {duration}")
    print(f"Results saved to: {results_file}")
    print("\nNext steps:")
    print("  1. Check the results in each experiment directory")
    print("  2. Compare validation losses across configurations")
    print("  3. Select the best configuration for full training")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
