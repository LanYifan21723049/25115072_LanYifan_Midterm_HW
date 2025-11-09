#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a small subset of translation dataset for quick ablation testing
"""
import os
import json
import random
from datasets import load_from_disk, DatasetDict


def create_small_translation_dataset(
    input_dir='data/iwslt2017',
    output_dir='data/iwslt2017_small',
    train_size=5000,
    val_size=500,
    test_size=500,
    src_lang='en',
    tgt_lang='de',
    seed=42
):
    """
    Create a small translation dataset for quick experiments
    
    Args:
        input_dir: Input directory containing the full dataset
        output_dir: Output directory for the small dataset
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        src_lang: Source language
        tgt_lang: Target language
        seed: Random seed for reproducibility
    """
    print(f"Creating small translation dataset...")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")
    print(f"  Test: {test_size} samples")
    print(f"  Language pair: {src_lang}-{tgt_lang}")
    
    random.seed(seed)
    
    # Load the full dataset from disk
    print(f"\nLoading full IWSLT 2017 dataset from {input_dir}...")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Dataset not found at {input_dir}. Please run download_translation_data.py first.")
    
    dataset = load_from_disk(input_dir)
    
    print(f"Original dataset sizes:")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")
    
    # Create random indices for sampling
    train_indices = random.sample(range(len(dataset['train'])), min(train_size, len(dataset['train'])))
    val_indices = random.sample(range(len(dataset['validation'])), min(val_size, len(dataset['validation'])))
    test_indices = random.sample(range(len(dataset['test'])), min(test_size, len(dataset['test'])))
    
    # Create subset
    small_dataset = DatasetDict({
        'train': dataset['train'].select(train_indices),
        'validation': dataset['validation'].select(val_indices),
        'test': dataset['test'].select(test_indices)
    })
    
    # Save the small dataset
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving small dataset to {output_dir}...")
    small_dataset.save_to_disk(output_dir)
    
    # Save metadata
    metadata = {
        'train_size': len(small_dataset['train']),
        'val_size': len(small_dataset['validation']),
        'test_size': len(small_dataset['test']),
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'seed': seed,
        'created_from': 'iwslt2017',
        'purpose': 'Quick ablation testing'
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nSmall dataset created successfully!")
    print(f"Final sizes:")
    print(f"  Train: {len(small_dataset['train'])} samples")
    print(f"  Validation: {len(small_dataset['validation'])} samples")
    print(f"  Test: {len(small_dataset['test'])} samples")
    
    # Show some sample data
    print("\nSample translation pairs:")
    for i in range(min(3, len(small_dataset['train']))):
        sample = small_dataset['train'][i]
        print(f"\n  Example {i+1}:")
        print(f"    {src_lang.upper()}: {sample['translation'][src_lang][:100]}...")
        print(f"    {tgt_lang.upper()}: {sample['translation'][tgt_lang][:100]}...")
    
    return output_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create small translation dataset')
    parser.add_argument('--input_dir', type=str, default='data/iwslt2017',
                        help='Input directory containing full dataset')
    parser.add_argument('--output_dir', type=str, default='data/iwslt2017_small',
                        help='Output directory')
    parser.add_argument('--train_size', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=500,
                        help='Number of validation samples')
    parser.add_argument('--test_size', type=int, default=500,
                        help='Number of test samples')
    parser.add_argument('--src_lang', type=str, default='en',
                        help='Source language')
    parser.add_argument('--tgt_lang', type=str, default='de',
                        help='Target language')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    create_small_translation_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        seed=args.seed
    )
