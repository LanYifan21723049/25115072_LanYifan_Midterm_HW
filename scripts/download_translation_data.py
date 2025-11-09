#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Translation Dataset - Fixed Version
Uses available datasets from Hugging Face
"""
import os
from datasets import load_dataset


def download_opus_books(data_dir='data/iwslt2017'):
    """
    Download OPUS Books en-de translation dataset
    A smaller but reliable alternative to IWSLT2017
    """
    print("Downloading OPUS Books en-de translation dataset...")
    print("(Using as alternative to deprecated IWSLT2017)")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # OPUS Books is a good alternative for translation
        dataset = load_dataset('opus_books', 'en-de')
        
        # Split into train/val/test
        print("\nSplitting dataset...")
        train_test = dataset['train'].train_test_split(test_size=0.1, seed=42)
        test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)
        
        final_dataset = {
            'train': train_test['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        }
        
        # Save
        from datasets import DatasetDict
        dataset_dict = DatasetDict(final_dataset)
        dataset_dict.save_to_disk(data_dir)
        
        print(f"\n[OK] Translation dataset saved to {data_dir}")
        print(f"\nDataset statistics:")
        print(f"  Train: {len(final_dataset['train'])} examples")
        print(f"  Validation: {len(final_dataset['validation'])} examples")
        print(f"  Test: {len(final_dataset['test'])} examples")
        
        # Show sample
        print("\nSample translation pair:")
        sample = final_dataset['train'][0]
        print(f"  EN: {sample['translation']['en'][:100]}...")
        print(f"  DE: {sample['translation']['de'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def download_wmt14_small(data_dir='data/iwslt2017'):
    """
    Download WMT14 de-en (smaller sample)
    """
    print("Downloading WMT14 de-en translation dataset...")
    print("Note: This is a large dataset, downloading may take time...")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        dataset = load_dataset('wmt14', 'de-en', split='train[:10000]')  # Only first 10k
        
        # Split
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)
        
        from datasets import DatasetDict
        dataset_dict = DatasetDict({
            'train': train_test['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })
        
        dataset_dict.save_to_disk(data_dir)
        
        print(f"\n[OK] WMT14 sample saved to {data_dir}")
        print(f"  Train: {len(train_test['train'])} examples")
        print(f"  Validation: {len(test_val['train'])} examples")
        print(f"  Test: {len(test_val['test'])} examples")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def create_toy_dataset(data_dir='data/iwslt2017'):
    """
    Create a tiny toy translation dataset for testing
    """
    print("Creating toy translation dataset for testing...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Simple parallel sentences
    train_data = {
        'translation': [
            {'en': 'Hello, how are you?', 'de': 'Hallo, wie geht es dir?'},
            {'en': 'I am fine, thank you.', 'de': 'Mir geht es gut, danke.'},
            {'en': 'What is your name?', 'de': 'Wie hei?t du?'},
            {'en': 'My name is John.', 'de': 'Ich hei?e John.'},
            {'en': 'Nice to meet you.', 'de': 'Sch?n dich kennenzulernen.'},
            {'en': 'Good morning!', 'de': 'Guten Morgen!'},
            {'en': 'Good evening!', 'de': 'Guten Abend!'},
            {'en': 'See you later.', 'de': 'Bis sp?ter.'},
            {'en': 'Thank you very much.', 'de': 'Vielen Dank.'},
            {'en': 'You are welcome.', 'de': 'Gern geschehen.'},
        ]
    }
    
    val_data = {
        'translation': [
            {'en': 'How old are you?', 'de': 'Wie alt bist du?'},
            {'en': 'Where are you from?', 'de': 'Woher kommst du?'},
        ]
    }
    
    test_data = {
        'translation': [
            {'en': 'I like coffee.', 'de': 'Ich mag Kaffee.'},
            {'en': 'This is a book.', 'de': 'Das ist ein Buch.'},
        ]
    }
    
    from datasets import Dataset, DatasetDict
    
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'validation': Dataset.from_dict(val_data),
        'test': Dataset.from_dict(test_data)
    })
    
    dataset_dict.save_to_disk(data_dir)
    
    print(f"\n[OK] Toy dataset saved to {data_dir}")
    print(f"  Train: {len(train_data['translation'])} examples")
    print(f"  Validation: {len(val_data['translation'])} examples")
    print(f"  Test: {len(test_data['translation'])} examples")
    print("\nNote: This is a TOY dataset for testing only!")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='opus',
                        choices=['opus', 'wmt14', 'toy'],
                        help='Download method: opus (recommended), wmt14 (large), toy (testing)')
    parser.add_argument('--data_dir', type=str, default='data/iwslt2017',
                        help='Directory to save dataset')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  Translation Dataset Downloader")
    print("="*60)
    print()
    
    if args.method == 'opus':
        success = download_opus_books(args.data_dir)
    elif args.method == 'wmt14':
        success = download_wmt14_small(args.data_dir)
    elif args.method == 'toy':
        success = create_toy_dataset(args.data_dir)
    
    if success:
        print("\n" + "="*60)
        print("  [OK] Download completed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("  [FAILED] Download failed, trying toy dataset...")
        print("="*60)
        create_toy_dataset(args.data_dir)
