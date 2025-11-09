#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Processing Utilities
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re


class Vocabulary:
    """Vocabulary Class"""
    def __init__(self, max_size=None, min_freq=1, use_pretokenized=False):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        self.token_counts = Counter()
        self.use_pretokenized = use_pretokenized  # For PTB: tokens already split by space
        
        # Special tokens
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        # Count token frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.token_counts.update(tokens)
        
        # Add special tokens
        self.token2idx = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        
        # Sort tokens by frequency and add to vocab
        sorted_tokens = sorted(self.token_counts.items(), key=lambda x: x[1], reverse=True)
        
        for token, count in sorted_tokens:
            if count < self.min_freq:
                break
            if self.max_size and len(self.token2idx) >= self.max_size:
                break
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
        
        # Build reverse mapping
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
    
    def tokenize(self, text):
        """Tokenize text"""
        if self.use_pretokenized:
            # PTB format: already tokenized, just split by space
            # Don't lowercase or re-tokenize
            tokens = text.strip().split()
        else:
            # Regular tokenization: lowercase and use regex
            text = text.lower().strip()
            tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to indices"""
        tokens = self.tokenize(text)
        indices = [self.token2idx.get(token, self.token2idx[self.unk_token]) for token in tokens]
        
        if add_special_tokens:
            indices = [self.token2idx[self.sos_token]] + indices + [self.token2idx[self.eos_token]]
        
        return indices
    
    def decode(self, indices, remove_special_tokens=True):
        """Decode indices to text"""
        tokens = [self.idx2token.get(idx, self.unk_token) for idx in indices]
        
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in [self.pad_token, self.sos_token, self.eos_token]]
        
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.token2idx)
    
    def __getitem__(self, token):
        return self.token2idx.get(token, self.token2idx[self.unk_token])


class TextDataset(Dataset):
    """Text Dataset (for Language Modeling)"""
    def __init__(self, texts, vocab, max_len=128, sliding_window=False):
        self.vocab = vocab
        self.max_len = max_len
        self.sliding_window = sliding_window
        
        if sliding_window:
            # Concatenate all texts and create sliding windows
            self.sequences = []
            for text in texts:
                indices = self.vocab.encode(text, add_special_tokens=False)
                # Use stride = max_len for NO OVERLAP (to prevent data leakage)
                # With overlap, model can "cheat" by memorizing what comes after each chunk
                stride = max_len  # Changed from max_len // 2 to max_len
                for i in range(0, len(indices) - max_len + 1, stride):
                    self.sequences.append(indices[i:i + max_len])
            print(f"  Created {len(self.sequences)} sequences from {len(texts)} texts")
        else:
            # Original behavior: each text is a separate sample
            self.texts = texts
            self.sequences = None
        
    def __len__(self):
        if self.sliding_window:
            return len(self.sequences)
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.sliding_window:
            indices = self.sequences[idx]
        else:
            text = self.texts[idx]
            indices = self.vocab.encode(text, add_special_tokens=False)
            
            # Truncate
            if len(indices) > self.max_len:
                indices = indices[:self.max_len]
        
        # Convert to tensor
        return torch.LongTensor(indices)


class TranslationDataset(Dataset):
    """Translation Dataset (Sequence-to-Sequence)"""
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=100):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        src_indices = self.src_vocab.encode(src_text, add_special_tokens=True)
        tgt_indices = self.tgt_vocab.encode(tgt_text, add_special_tokens=True)
        
        # Truncate
        if len(src_indices) > self.max_len:
            src_indices = src_indices[:self.max_len]
        if len(tgt_indices) > self.max_len:
            tgt_indices = tgt_indices[:self.max_len]
        
        return torch.LongTensor(src_indices), torch.LongTensor(tgt_indices)


def collate_fn_lm(batch):
    """Collate function for language modeling"""
    # Find the longest sequence in the batch
    max_len = max(len(item) for item in batch)
    
    # Pad all sequences to the same length
    padded_batch = []
    for item in batch:
        padding = torch.zeros(max_len - len(item), dtype=torch.long)
        padded_item = torch.cat([item, padding])
        padded_batch.append(padded_item)
    
    return torch.stack(padded_batch)


def collate_fn_translation(batch):
    """Collate function for translation task"""
    src_batch, tgt_batch = zip(*batch)
    
    # Find longest sequences
    max_src_len = max(len(item) for item in src_batch)
    max_tgt_len = max(len(item) for item in tgt_batch)
    
    # Pad source sequences
    padded_src = []
    for item in src_batch:
        padding = torch.zeros(max_src_len - len(item), dtype=torch.long)
        padded_item = torch.cat([item, padding])
        padded_src.append(padded_item)
    
    # Pad target sequences
    padded_tgt = []
    for item in tgt_batch:
        padding = torch.zeros(max_tgt_len - len(item), dtype=torch.long)
        padded_item = torch.cat([item, padding])
        padded_tgt.append(padded_item)
    
    return torch.stack(padded_src), torch.stack(padded_tgt)


def create_lm_dataloaders(data_path, batch_size=32, max_len=128, num_workers=0, test_mode=False, data_format='auto'):
    """
    Create data loaders for language modeling
    
    Args:
        data_path: Path to dataset directory
        batch_size: Batch size
        max_len: Maximum sequence length
        num_workers: Number of data loading workers
        test_mode: If True, only return test loader
        data_format: 'auto', 'ptb', or 'wikitext'
    """
    
    # Detect format if auto
    if data_format == 'auto':
        if os.path.exists(os.path.join(data_path, 'ptb.train.txt')):
            data_format = 'ptb'
        elif os.path.exists(os.path.join(data_path, 'dataset_dict.json')):
            data_format = 'wikitext'
        else:
            # Try as simple text files
            data_format = 'text'
    
    # PTB format: each file is space-separated tokens
    if data_format == 'ptb':
        print("Loading PTB dataset...")
        
        train_file = os.path.join(data_path, 'ptb.train.txt')
        val_file = os.path.join(data_path, 'ptb.valid.txt')
        test_file = os.path.join(data_path, 'ptb.test.txt')
        
        # Read and concatenate all tokens
        with open(train_file, 'r', encoding='utf-8') as f:
            train_text = f.read().replace('\n', ' <eos> ')  # Mark sentence boundaries
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_text = f.read().replace('\n', ' <eos> ')
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_text = f.read().replace('\n', ' <eos> ')
        
        print(f"  Train tokens: {len(train_text.split()):,}")
        print(f"  Val tokens: {len(val_text.split()):,}")
        print(f"  Test tokens: {len(test_text.split()):,}")
        
        # Build vocabulary from training set
        # CRITICAL: use_pretokenized=True for PTB (tokens already split by space)
        vocab = Vocabulary(max_size=10000, use_pretokenized=True)
        vocab.build_vocab([train_text])
        
        # Create datasets with NO overlap (stride = max_len)
        train_dataset = TextDataset([train_text], vocab, max_len, sliding_window=True)
        val_dataset = TextDataset([val_text], vocab, max_len, sliding_window=True)
        test_dataset = TextDataset([test_text], vocab, max_len, sliding_window=True)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn_lm, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn_lm, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn_lm, num_workers=num_workers)
        
        if test_mode:
            return None, test_loader, len(vocab), vocab.token2idx, vocab.idx2token
        
        return train_loader, val_loader, len(vocab), vocab.token2idx, vocab.idx2token
    
    # Original WikiText / HuggingFace format
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)
    except:
        # If it's a simple text file
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Simple split
        n_train = int(len(lines) * 0.8)
        n_val = int(len(lines) * 0.1)
        
        train_texts = lines[:n_train]
        val_texts = lines[n_train:n_train+n_val]
        test_texts = lines[n_train+n_val:]
        
        # Build vocabulary
        vocab = Vocabulary(max_size=10000)
        vocab.build_vocab(train_texts)
        
        # Create datasets
        train_dataset = TextDataset(train_texts, vocab, max_len)
        val_dataset = TextDataset(val_texts, vocab, max_len)
        test_dataset = TextDataset(test_texts, vocab, max_len)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  collate_fn=collate_fn_lm, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn_lm, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn_lm, num_workers=num_workers)
        
        if test_mode:
            return None, test_loader, len(vocab), vocab.token2idx, vocab.idx2token
        
        return train_loader, val_loader, len(vocab), vocab.token2idx, vocab.idx2token
    
    # Handle Hugging Face datasets (WikiText-2)
    # IMPORTANT: Concatenate all text instead of treating each line separately
    print("Processing WikiText-2 dataset...")
    
    # Extract non-empty lines
    train_lines = [item['text'] for item in dataset['train'] if item['text'].strip()]
    val_lines = [item['text'] for item in dataset['validation'] if item['text'].strip()]
    test_lines = [item['text'] for item in dataset['test'] if item['text'].strip()]
    
    # Concatenate into single text for each split
    train_text = ' '.join(train_lines)
    val_text = ' '.join(val_lines)
    test_text = ' '.join(test_lines)
    
    print(f"  Train text length: {len(train_text):,} chars")
    print(f"  Val text length: {len(val_text):,} chars")
    
    # Build vocabulary
    vocab = Vocabulary(max_size=10000)
    vocab.build_vocab([train_text])
    
    # Create datasets with sliding window approach
    train_dataset = TextDataset([train_text], vocab, max_len, sliding_window=True)
    val_dataset = TextDataset([val_text], vocab, max_len, sliding_window=True)
    test_dataset = TextDataset([test_text], vocab, max_len, sliding_window=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_lm, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_lm, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn_lm, num_workers=num_workers)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    if test_mode:
        return None, test_loader, len(vocab), vocab.token2idx, vocab.idx2token
    
    return train_loader, val_loader, len(vocab), vocab.token2idx, vocab.idx2token


def create_translation_dataloaders(data_path, src_lang, tgt_lang, batch_size=32, 
                                     max_len=100, num_workers=0, test_mode=False):
    """Create data loaders for translation"""
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)
        
        # Extract texts
        train_src = [item['translation'][src_lang] for item in dataset['train']]
        train_tgt = [item['translation'][tgt_lang] for item in dataset['train']]
        
        val_src = [item['translation'][src_lang] for item in dataset['validation']]
        val_tgt = [item['translation'][tgt_lang] for item in dataset['validation']]
        
        test_src = [item['translation'][src_lang] for item in dataset['test']]
        test_tgt = [item['translation'][tgt_lang] for item in dataset['test']]
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Create dummy data
        train_src = ["Hello world", "How are you", "Good morning"] * 100
        train_tgt = ["Hallo Welt", "Wie geht es dir", "Guten Morgen"] * 100
        val_src = train_src[:50]
        val_tgt = train_tgt[:50]
        test_src = val_src
        test_tgt = val_tgt
    
    # Build vocabularies
    src_vocab = Vocabulary(max_size=8000)
    src_vocab.build_vocab(train_src)
    
    tgt_vocab = Vocabulary(max_size=8000)
    tgt_vocab.build_vocab(train_tgt)
    
    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, max_len)
    test_dataset = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_translation, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_translation, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn_translation, num_workers=num_workers)
    
    if test_mode:
        return None, test_loader, len(src_vocab), len(tgt_vocab), src_vocab.token2idx, tgt_vocab.token2idx
    
    return train_loader, val_loader, len(src_vocab), len(tgt_vocab), src_vocab.token2idx, tgt_vocab.token2idx