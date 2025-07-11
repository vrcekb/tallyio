#!/usr/bin/env python3
"""
TallyIO Strategy Core - Fix Duplicates Script
Fixes duplicate Ok(()) statements and duplicate #[must_use] attributes.
"""

import os
import re
import glob

def fix_duplicate_ok_statements(content):
    """Fix duplicate Ok(()) statements"""
    
    # Fix duplicate Ok(()) on consecutive lines
    content = re.sub(
        r'(\s+)Ok\(\(\)\)\s*\n\s*Ok\(\(\)\)',
        r'\1Ok(())',
        content
    )
    
    return content

def fix_duplicate_must_use(content):
    """Fix duplicate #[must_use] attributes"""
    
    # Remove duplicate #[must_use] attributes
    content = re.sub(
        r'(\s+)#\[must_use\]\s*\n\s*#\[must_use\]',
        r'\1#[must_use]',
        content
    )
    
    return content

def fix_empty_lines_after_attributes(content):
    """Fix empty lines after attributes"""
    
    # Remove empty lines after #[must_use]
    content = re.sub(
        r'(\s+)#\[must_use\]\s*\n\s*\n',
        r'\1#[must_use]\n',
        content
    )
    
    return content

def fix_test_functions_with_question_mark(content):
    """Fix test functions that use ? operator"""
    
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Check if this is a test function that uses ?
        if ('fn ' in line and 'test' in line and 
            '() {' in line and i < len(lines) - 10):
            
            # Look ahead to see if there's a ? operator in the function
            function_content = '\n'.join(lines[i:i+10])
            if '?' in function_content and 'assert!' not in function_content:
                # Convert ? to assert!
                for j in range(i+1, min(i+10, len(lines))):
                    if '?' in lines[j] and 'assert!' not in lines[j]:
                        # Replace result? with assert!(result.is_ok())
                        lines[j] = re.sub(
                            r'(\s+)(\w+)\?;',
                            r'\1assert!(\2.is_ok());',
                            lines[j]
                        )
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_literal_suffixes(content):
    """Fix remaining literal suffixes"""
    
    # Fix remaining unseparated literal suffixes
    content = re.sub(r'(\d+)u8', r'\1_u8', content)
    content = re.sub(r'(\d+)u32', r'\1_u32', content)
    content = re.sub(r'(\d+)u64', r'\1_u64', content)
    
    return content

def process_file(filepath):
    """Process a single Rust file"""
    
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply all fixes
    content = fix_duplicate_ok_statements(content)
    content = fix_duplicate_must_use(content)
    content = fix_empty_lines_after_attributes(content)
    content = fix_test_functions_with_question_mark(content)
    content = fix_literal_suffixes(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def main():
    """Main function"""
    
    print("🚀 TallyIO Strategy Core - Fix Duplicates")
    print("Fixing duplicate statements and attributes...")
    
    # Find all Rust files
    rust_files = glob.glob('src/**/*.rs', recursive=True)
    
    for filepath in rust_files:
        process_file(filepath)
    
    print(f"✅ Fixed {len(rust_files)} files")
    print("🎯 Ready for clippy validation!")

if __name__ == '__main__':
    main()
