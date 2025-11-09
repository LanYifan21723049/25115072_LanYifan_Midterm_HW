#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Europarl raw text files to HuggingFace datasets format
Europarl is a high-quality parallel corpus from European Parliament proceedings
"""
import os
from datasets import Dataset, DatasetDict

def load_europarl(de_file, en_file, max_samples=None):
    """Load Europarl parallel corpus"""
    print(f"Loading Europarl corpus...")
    print(f"  DE file: {de_file}")
    print(f"  EN file: {en_file}")
    
    de_sentences = []
    en_sentences = []
    
    # Read German sentences
    with open(de_file, 'r', encoding='utf-8') as f:
        for line in f:
            de_sentences.append(line.strip())
            if max_samples and len(de_sentences) >= max_samples:
                break
    
    # Read English sentences
    with open(en_file, 'r', encoding='utf-8') as f:
        for line in f:
            en_sentences.append(line.strip())
            if max_samples and len(en_sentences) >= max_samples:
                break
    
    # Ensure equal lengths
    min_len = min(len(de_sentences), len(en_sentences))
    de_sentences = de_sentences[:min_len]
    en_sentences = en_sentences[:min_len]
    
    print(f"  Loaded {len(de_sentences):,} sentence pairs")
    
    return de_sentences, en_sentences


def create_dataset_from_europarl(
    de_file='data/iwslt2017/training/europarl-v7.de-en.de',
    en_file='data/iwslt2017/training/europarl-v7.de-en.en',
    output_dir='data/iwslt2017',
    train_samples=50000,  # Use 50K for training (faster)
    val_samples=2000,
    test_samples=2000
):
    """
    Convert Europarl text files to HuggingFace dataset format
    
    Args:
        de_file: Path to German text file
        en_file: Path to English text file
        output_dir: Output directory for processed dataset
        train_samples: Number of training samples (default: 50K)
        val_samples: Number of validation samples
        test_samples: Number of test samples
    """
    print("="*70)
    print("  Converting Europarl to Dataset Format")
    print("="*70)
    print()
    
    # Check if files exist
    if not os.path.exists(de_file):
        print(f"[ERROR] German file not found: {de_file}")
        return False
    
    if not os.path.exists(en_file):
        print(f"[ERROR] English file not found: {en_file}")
        return False
    
    # Load data
    total_needed = train_samples + val_samples + test_samples
    de_sentences, en_sentences = load_europarl(de_file, en_file, max_samples=total_needed)
    
    if len(de_sentences) < total_needed:
        print(f"[WARNING] Only {len(de_sentences):,} samples available")
        print(f"            Requested {total_needed:,} samples")
        # Adjust sample sizes
        ratio = len(de_sentences) / total_needed
        train_samples = int(train_samples * ratio)
        val_samples = int(val_samples * ratio)
        test_samples = len(de_sentences) - train_samples - val_samples
    
    print()
    print(f"Creating dataset splits:")
    print(f"  Train: {train_samples:,} samples")
    print(f"  Validation: {val_samples:,} samples")
    print(f"  Test: {test_samples:,} samples")
    print()
    
    # Create translation pairs
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    print("Processing data...")
    
    # Split data
    idx = 0
    
    # Training set
    for i in range(train_samples):
        if de_sentences[idx] and en_sentences[idx]:  # Skip empty lines
            train_pairs.append({
                'translation': {
                    'en': en_sentences[idx],
                    'de': de_sentences[idx]
                }
            })
        idx += 1
    
    # Validation set
    for i in range(val_samples):
        if idx < len(de_sentences) and de_sentences[idx] and en_sentences[idx]:
            val_pairs.append({
                'translation': {
                    'en': en_sentences[idx],
                    'de': de_sentences[idx]
                }
            })
        idx += 1
    
    # Test set
    for i in range(test_samples):
        if idx < len(de_sentences) and de_sentences[idx] and en_sentences[idx]:
            test_pairs.append({
                'translation': {
                    'en': en_sentences[idx],
                    'de': de_sentences[idx]
                }
            })
        idx += 1
    
    print(f"  Created {len(train_pairs):,} training pairs")
    print(f"  Created {len(val_pairs):,} validation pairs")
    print(f"  Created {len(test_pairs):,} test pairs")
    print()
    
    # Create datasets
    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)
    test_dataset = Dataset.from_list(test_pairs)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Save
    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove old dataset if exists
    if os.path.exists(output_dir):
        import shutil
        for item in ['train', 'validation', 'test', 'dataset_dict.json']:
            item_path = os.path.join(output_dir, item)
            if os.path.exists(item_path):
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
    
    dataset_dict.save_to_disk(output_dir)
    
    print()
    print("="*70)
    print("  [SUCCESS] Dataset created!")
    print("="*70)
    print()
    print(f"Dataset saved to: {output_dir}")
    print()
    print("Sample translations:")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        en = sample['translation']['en']
        de = sample['translation']['de']
        print(f"  [{i+1}]")
        print(f"    EN: {en[:100]}{'...' if len(en) > 100 else ''}")
        print(f"    DE: {de[:100]}{'...' if len(de) > 100 else ''}")
        print()
    
    print("="*70)
    print("  Ready to train!")
    print("  Run: python src/run_train_translation.py")
    print("="*70)
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Europarl to dataset format')
    parser.add_argument('--de_file', type=str, 
                        default='data/iwslt2017/training/europarl-v7.de-en.de',
                        help='Path to German file')
    parser.add_argument('--en_file', type=str,
                        default='data/iwslt2017/training/europarl-v7.de-en.en',
                        help='Path to English file')
    parser.add_argument('--output_dir', type=str,
                        default='data/iwslt2017',
                        help='Output directory')
    parser.add_argument('--train_samples', type=int, default=50000,
                        help='Number of training samples (default: 50000)')
    parser.add_argument('--val_samples', type=int, default=2000,
                        help='Number of validation samples (default: 2000)')
    parser.add_argument('--test_samples', type=int, default=2000,
                        help='Number of test samples (default: 2000)')
    
    args = parser.parse_args()
    
    success = create_dataset_from_europarl(
        de_file=args.de_file,
        en_file=args.en_file,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples
    )
    
    if not success:
        print("\n[FAILED] Dataset creation failed")
        exit(1)
