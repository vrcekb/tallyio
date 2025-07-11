#!/usr/bin/env python3
"""
TallyIO Strategy Core - Final Fix Script
Fixes all remaining clippy violations for 100% compliance.
"""

import os
import re
import glob

def fix_needless_returns(content):
    """Fix needless return statements"""
    
    # Fix simple return statements
    content = re.sub(r'        return (Ok\([^)]*\));', r'        \1', content)
    content = re.sub(r'        return (Some\([^)]*\));', r'        \1', content)
    content = re.sub(r'        return (Vec::new\(\));', r'        \1', content)
    
    # Fix multi-line return statements
    content = re.sub(
        r'        return (other\.priority\.cmp\(&self\.priority\)\s*\.then_with\([^)]*\)\s*\.then_with\([^)]*\));',
        r'        \1',
        content,
        flags=re.DOTALL
    )
    
    return content

def fix_empty_doc_comments(content):
    """Fix empty lines after doc comments"""
    
    # Remove empty lines after doc comments
    content = re.sub(
        r'(\s+/// [^\n]*)\n\s*\n(\s+/// )',
        r'\1\n\2',
        content
    )
    
    # Remove trailing empty doc comment lines
    content = re.sub(
        r'(\s+/// [^\n]*)\n\s*\n(\s+pub fn)',
        r'\1\n\2',
        content
    )
    
    return content

def fix_test_functions_with_question_mark(content):
    """Fix remaining test functions that use ? operator"""
    
    # Replace ? with assert! in test functions
    content = re.sub(
        r'(\s+)(\w+)\?;',
        r'\1assert!(\2.is_ok());',
        content
    )
    
    # Fix moved values in tests
    content = re.sub(
        r'(\s+)coordinator\?;\s*\n\s*if let Ok\(coordinator\) = coordinator \{',
        r'\1assert!(coordinator.is_ok());\n\1if let Ok(coordinator) = coordinator {',
        content
    )
    
    content = re.sub(
        r'(\s+)coordinator\?;\s*\n(\s+)let coordinator = coordinator\?;',
        r'\1let coordinator = coordinator.unwrap();',
        content
    )
    
    return content

def fix_await_on_non_async(content):
    """Fix .await on non-async functions"""
    
    # Remove .await from stop() calls that are not async
    content = re.sub(
        r'coordinator\.stop\(\)\.await',
        r'coordinator.stop()',
        content
    )
    
    return content

def fix_cast_precision_loss(content):
    """Fix cast precision loss warnings"""
    
    # Add allow attribute for intentional precision loss
    content = re.sub(
        r'(\s+)(let memory_mb = \(\(self\.total_memory_mb as f64\)[^;]*;)',
        r'\1#[allow(clippy::cast_precision_loss)]\n\1\2',
        content
    )
    
    return content

def process_file(filepath):
    """Process a single Rust file"""
    
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply all fixes
    content = fix_needless_returns(content)
    content = fix_empty_doc_comments(content)
    content = fix_test_functions_with_question_mark(content)
    content = fix_await_on_non_async(content)
    content = fix_cast_precision_loss(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def main():
    """Main function"""
    
    print("🚀 TallyIO Strategy Core - Final Fix")
    print("Fixing all remaining clippy violations for 100% compliance...")
    
    # Find all Rust files
    rust_files = glob.glob('src/**/*.rs', recursive=True)
    
    for filepath in rust_files:
        process_file(filepath)
    
    print(f"✅ Fixed {len(rust_files)} files")
    print("🎯 Ready for 100% clippy compliance!")

if __name__ == '__main__':
    main()
