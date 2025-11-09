#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create 200K training dataset"""
import os
import shutil
from datasets import Dataset, DatasetDict

print("="*70)
print("  Creating 200K Europarl Dataset")
print("="*70)
print()

de_file = 'data/iwslt2017/training/europarl-v7.de-en.de'
en_file = 'data/iwslt2017/training/europarl-v7.de-en.en'

# Load data
print("Loading Europarl files...")
print(f"  DE: {de_file}")
print(f"  EN: {en_file}")

with open(de_file, 'r', encoding='utf-8') as f:
    de_lines = [line.strip() for line in f.readlines()[:204000]]  # 200K + 4K for val/test

with open(en_file, 'r', encoding='utf-8') as f:
    en_lines = [line.strip() for line in f.readlines()[:204000]]

print(f"  Loaded {len(de_lines):,} lines")
print()

# Create pairs
print("Creating translation pairs...")
pairs = []
for de, en in zip(de_lines, en_lines):
    if de and en:  # Skip empty
        pairs.append({'translation': {'en': en, 'de': de}})

print(f"  Total valid pairs: {len(pairs):,}")
print()

# Split
train_pairs = pairs[:200000]
val_pairs = pairs[200000:202000]
test_pairs = pairs[202000:204000]

print("Dataset splits:")
print(f"  Train:      {len(train_pairs):>8,} samples (x4 increase!)")
print(f"  Validation: {len(val_pairs):>8,} samples")
print(f"  Test:       {len(test_pairs):>8,} samples")
print()

# Create datasets
dataset_dict = DatasetDict({
    'train': Dataset.from_list(train_pairs),
    'validation': Dataset.from_list(val_pairs),
    'test': Dataset.from_list(test_pairs)
})

# Clean old data
output_dir = 'data/iwslt2017'
print(f"Cleaning old dataset at {output_dir}...")
for item in ['train', 'validation', 'test', 'dataset_dict.json']:
    item_path = os.path.join(output_dir, item)
    if os.path.exists(item_path):
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

# Save
print(f"Saving new dataset...")
dataset_dict.save_to_disk(output_dir)

print()
print("="*70)
print("  SUCCESS! 200K Dataset Created!")
print("="*70)
print()
print(f"Location: {output_dir}")
print(f"Training samples: {len(train_pairs):,} (was 50,000)")
print(f"Data increase: 4x")
print()
print("Sample:")
sample = train_pairs[0]
print(f"  EN: {sample['translation']['en'][:80]}...")
print(f"  DE: {sample['translation']['de'][:80]}...")
print()
print("="*70)
print("  Next Steps:")
print("="*70)
print("1. Training will take ~1.5 hours (4x longer)")
print("2. Expected BLEU: 25-30 (5-8 points improvement)")
print("3. Run: python src/run_train_translation.py")
print("="*70)
