#!/usr/bin/env python3
"""
TallyIO Strategy Core - Fix Return Statements Script
Fixes needless return statements for clippy compliance.
"""

import os
import re
import glob

def fix_needless_returns(content):
    """Fix needless return statements"""
    
    # Fix simple return statements
    content = re.sub(r'        return (Ok\(\(\)\));', r'        \1', content)
    content = re.sub(r'        return (Ok\(Vec::with_capacity\(0\)\));', r'        \1', content)
    content = re.sub(r'        return (Some\(self\.cmp\(other\)\));', r'        \1', content)
    
    # Fix multi-line return statement
    content = re.sub(
        r'        return (other\.priority\.cmp\(&self\.priority\)\s*\.then_with\(\|\| self\.expected_profit\.cmp\(&other\.expected_profit\)\)\s*\.then_with\(\|\| other\.timestamp\.cmp\(&self\.timestamp\)\));',
        r'        \1',
        content,
        flags=re.DOTALL
    )
    
    return content

def process_file(filepath):
    """Process a single Rust file"""
    
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply fixes
    content = fix_needless_returns(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def main():
    """Main function"""
    
    print("🚀 TallyIO Strategy Core - Fix Return Statements")
    print("Fixing needless return statements...")
    
    # Target specific files with return issues
    target_files = [
        'src/coordination/conflict_resolver.rs',
        'src/coordination/parallel_executor.rs', 
        'src/coordination/yield_optimizer.rs',
        'src/priority/mod.rs',
        'src/priority/execution_queue.rs',
        'src/time_bandit/mod.rs',
        'src/zero_risk/mod.rs'
    ]
    
    for filepath in target_files:
        if os.path.exists(filepath):
            process_file(filepath)
    
    print(f"✅ Fixed {len(target_files)} files")
    print("🎯 Ready for final clippy validation!")

if __name__ == '__main__':
    main()
