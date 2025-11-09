#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run model size ablation on full data (15 epochs - quick validation)
Strategy 3: Fast validation with reduced epochs
Estimated time: ~13.5 hours
"""
import sys
import subprocess

if __name__ == '__main__':
    print("="*70)
    print("  Model Size Ablation - Quick Validation (15 epochs)")
    print("="*70)
    print("\nThis will test 3 model sizes:")
    print("  1. Medium (d=512) - Your baseline")
    print("  2. Large (d=1024) - Potentially better")
    print("  3. Small (d=256) - Lighter alternative")
    print("\nEstimated time: ~13.5 hours")
    print("Data: Full 200K IWSLT2017 dataset")
    print("="*70)
    print("\nStarting ablation study...\n")
    
    cmd = [
        sys.executable,
        'run_ablation_on_full_data.py',
        '--ablation_type', 'size',
        '--epochs', '15'
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("? Ablation study completed successfully!")
        print("Check results in: results/ablation_full/ablation_size/")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("?? Ablation study finished (check results for details)")
        print("="*70)
