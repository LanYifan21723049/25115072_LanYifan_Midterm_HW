#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run all remaining ablation experiments (layers, dropout, size)
"""
import sys
import subprocess

if __name__ == '__main__':
    print("Running all remaining ablation experiments...")
    print("This will take approximately 8-10 minutes\n")
    
    experiments = ['layers', 'dropout', 'size']
    
    for exp_type in experiments:
        print(f"\n{'='*70}")
        print(f"Starting {exp_type.upper()} ablation...")
        print(f"{'='*70}\n")
        
        cmd = [
            sys.executable,
            'run_quick_ablation.py',
            '--ablation_type', exp_type,
            '--epochs', '5'
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"\n? {exp_type.upper()} ablation completed successfully!")
        else:
            print(f"\n?? {exp_type.upper()} ablation finished with warnings (check results)")
    
    print("\n" + "="*70)
    print("All ablation experiments completed!")
    print("Check results in: results/ablation_quick/")
    print("="*70)
