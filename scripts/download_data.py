#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download datasets from Hugging Face
"""
import os
import argparse


def download_wikitext2(data_dir='data/wikitext-2'):
    """Download WikiText-2 dataset"""
    print("Downloading WikiText-2 dataset...")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        
        # Save dataset
        dataset.save_to_disk(data_dir)
        print(f"WikiText-2 saved to {data_dir}")
        
        # Print statistics
        print(f"\nDataset statistics:")
        print(f"  Train: {len(dataset['train'])} examples")
        print(f"  Validation: {len(dataset['validation'])} examples")
        print(f"  Test: {len(dataset['test'])} examples")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")


def download_tiny_shakespeare(data_dir='data/tiny_shakespeare'):
    """Download Tiny Shakespeare dataset"""
    print("Downloading Tiny Shakespeare dataset...")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        from datasets import load_dataset
        dataset = load_dataset('tiny_shakespeare')
        
        dataset.save_to_disk(data_dir)
        print(f"Tiny Shakespeare saved to {data_dir}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")


def download_iwslt2017(data_dir='data/iwslt2017', src_lang='en', tgt_lang='de'):
    """Download IWSLT2017 dataset
    
    Note: Since 'iwslt2017' is deprecated in newer datasets library,
    we use 'wmt14' as an alternative translation dataset, or manually download IWSLT2017.
    """
    print(f"Downloading translation dataset ({src_lang}-{tgt_lang})...")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        # Try using WMT14 as alternative (more stable)
        print("Using WMT14 de-en dataset as alternative...")
        dataset = load_dataset('wmt14', 'de-en')
        
        dataset.save_to_disk(data_dir)
        print(f"Translation dataset saved to {data_dir}")
        
        print(f"\nDataset statistics:")
        if 'train' in dataset:
            print(f"  Train: {len(dataset['train'])} examples")
        if 'validation' in dataset:
            print(f"  Validation: {len(dataset['validation'])} examples")
        if 'test' in dataset:
            print(f"  Test: {len(dataset['test'])} examples")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Manually download IWSLT2017 from:")
        print("  https://wit3.fbk.eu/2017-01")
        print("Or use a smaller dataset for testing.")


def download_ag_news(data_dir='data/ag_news'):
    """Download AG News dataset"""
    print("Downloading AG News dataset...")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        from datasets import load_dataset
        dataset = load_dataset('ag_news')
        
        dataset.save_to_disk(data_dir)
        print(f"AG News saved to {data_dir}")
        
        print(f"\nDataset statistics:")
        print(f"  Train: {len(dataset['train'])} examples")
        print(f"  Test: {len(dataset['test'])} examples")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wikitext2', 'tiny_shakespeare', 'iwslt2017', 'ag_news'],
                        help='Dataset to download')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory to save dataset')
    parser.add_argument('--src_lang', type=str, default='en',
                        help='Source language for translation (IWSLT2017)')
    parser.add_argument('--tgt_lang', type=str, default='de',
                        help='Target language for translation (IWSLT2017)')
    
    args = parser.parse_args()
    
    if args.dataset == 'wikitext2':
        data_dir = args.data_dir or 'data/wikitext-2'
        download_wikitext2(data_dir)
    elif args.dataset == 'tiny_shakespeare':
        data_dir = args.data_dir or 'data/tiny_shakespeare'
        download_tiny_shakespeare(data_dir)
    elif args.dataset == 'iwslt2017':
        data_dir = args.data_dir or 'data/iwslt2017'
        download_iwslt2017(data_dir, args.src_lang, args.tgt_lang)
    elif args.dataset == 'ag_news':
        data_dir = args.data_dir or 'data/ag_news'
        download_ag_news(data_dir)
    
    print("\nDownload completed!")


if __name__ == '__main__':
    main()

