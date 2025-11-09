#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download and prepare Penn Treebank (PTB) dataset for language modeling
PTB is the standard benchmark for language modeling
"""
import os
import urllib.request
import zipfile
import shutil

def download_file(url, filepath):
    """Download file with progress"""
    print(f"Downloading from {url}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
    print("\nDownload completed!")


def prepare_ptb_dataset():
    """
    Download and prepare PTB dataset
    
    PTB (Penn Treebank) is the most widely used benchmark for language modeling.
    - Training: ~930K tokens
    - Validation: ~74K tokens  
    - Test: ~82K tokens
    - Vocabulary: ~10K words
    
    Expected performance (your model size):
    - Baseline (no training): PPL ~1000
    - After 10 epochs: PPL ~150-200
    - After 30 epochs: PPL ~100-130
    - Good model: PPL ~80-100
    - State-of-art (large model): PPL ~50-60
    """
    
    print("="*70)
    print("  Penn Treebank (PTB) Dataset Setup")
    print("="*70)
    
    # Create data directory
    data_dir = 'data/ptb'
    os.makedirs(data_dir, exist_ok=True)
    
    # PTB is hosted on multiple sources
    # We'll use a more reliable mirror
    
    # Try multiple sources
    sources = [
        {
            'name': 'Hugging Face Mirror',
            'base_url': 'https://huggingface.co/datasets/ptb_text_only/resolve/main/data/',
            'files': {
                'train': 'train.txt',
                'valid': 'validation.txt', 
                'test': 'test.txt'
            }
        },
        {
            'name': 'Original LSTM Repo',
            'base_url': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/',
            'files': {
                'train': 'ptb.train.txt',
                'valid': 'ptb.valid.txt',
                'test': 'ptb.test.txt'
            }
        }
    ]
    
    print(f"\nDownloading PTB dataset to {data_dir}/...")
    print("Trying multiple mirrors...")
    print()
    
    downloaded = False
    for source in sources:
        print(f"\nTrying {source['name']}...")
        
        success = True
        for split, filename in source['files'].items():
            output_filename = f'ptb.{split}.txt'
            filepath = os.path.join(data_dir, output_filename)
            
            if os.path.exists(filepath):
                print(f"? {output_filename} already exists")
                continue
            
            url = source['base_url'] + filename
            try:
                download_file(url, filepath)
                print(f"? Downloaded {output_filename}")
            except Exception as e:
                print(f"? Failed: {e}")
                success = False
                if os.path.exists(filepath):
                    os.remove(filepath)
                break
        
        if success:
            downloaded = True
            print(f"\n? Successfully downloaded from {source['name']}")
            break
    
    if not downloaded:
        print("\n? All mirrors failed. Creating sample dataset for testing...")
        create_sample_dataset(data_dir)
        return True
    
    print("\n" + "="*70)
    print("Dataset Statistics:")
    print("="*70)
    
    # Count tokens and lines
    ptb_files = {
        'train': 'ptb.train.txt',
        'valid': 'ptb.valid.txt',
        'test': 'ptb.test.txt'
    }
    
    for split, filename in ptb_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = text.split()
                lines = text.split('\n')
                
            print(f"\n{split.upper()} set ({filename}):")
            print(f"  Tokens: {len(tokens):,}")
            print(f"  Lines: {len(lines):,}")
            print(f"  Unique tokens (approx): {len(set(tokens)):,}")
            
            # Show sample
            sample = ' '.join(tokens[:50])
            print(f"  Sample: {sample}...")
    
    print("\n" + "="*70)
    print("? PTB dataset ready!")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Train model:")
    print("     python src/train_lm.py --data_path data/ptb \\")
    print("                            --data_format ptb \\")
    print("                            --epochs 30")
    print()
    print("  2. Or use the run script:")
    print("     python src/run_train_lm.py --use_ptb")
    print()
    
    return True


if __name__ == '__main__':
    success = prepare_ptb_dataset()
    if not success:
        print("\n? Failed to download PTB dataset")
        print("You can also manually download from:")
        print("  https://github.com/wojzaremba/lstm/tree/master/data")
        exit(1)
