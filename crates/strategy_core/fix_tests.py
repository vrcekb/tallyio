#!/usr/bin/env python3
"""
TallyIO Strategy Core - Fix Test Functions Script
Fixes test functions that incorrectly call methods on Result types.
"""

import os
import re
import glob

def fix_test_functions(content):
    """Fix test functions that call methods on Result types"""
    
    # Pattern to match test functions with coordinator issues
    pattern = r'(async fn \w*test\w*\(\) \{[^}]*let coordinator = \w+Coordinator::new\([^)]*\);[^}]*assert!\(coordinator\.is_ok\(\)\);[^}]*)(let \w+_result = coordinator\.(\w+)\(\)(?:\.await)?;[^}]*assert!\(\w+_result\.is_ok\(\)\);[^}]*)(let \w+_result = coordinator\.(\w+)\(\)(?:\.await)?;[^}]*assert!\(\w+_result\.is_ok\(\)\);[^}]*)\}'
    
    def replace_test(match):
        prefix = match.group(1)
        first_call = match.group(2)
        second_call = match.group(3)
        
        # Extract method names
        first_method = match.group(3)
        second_method = match.group(5)
        
        # Build replacement with proper if let structure
        replacement = f"""{prefix}
        
        if let Ok(coordinator) = coordinator {{
            let start_result = coordinator.{first_method}();
            assert!(start_result.is_ok());

            let stop_result = coordinator.{second_method}();
            assert!(stop_result.is_ok());
        }}
    }}"""
        
        return replacement
    
    # Apply the pattern replacement
    content = re.sub(pattern, replace_test, content, flags=re.DOTALL)
    
    return content

def fix_empty_doc_comments(content):
    """Fix empty lines after doc comments"""
    
    # Remove empty doc comment lines followed by empty lines
    content = re.sub(
        r'(\s+/// [^\n]*)\n\s+/// \s*\n\s*\n(\s+/// # Errors[^}]*?)(\s+pub (?:async )?fn)',
        r'\1\n\2\3',
        content,
        flags=re.DOTALL
    )
    
    # Remove duplicate doc comment sections
    content = re.sub(
        r'(\s+/// # Errors\s*\n\s+///\s*\n\s+/// Returns error if [^\n]*)\n\s+/// \s*\n\s*\n\s+/// # Errors\s*\n\s+///\s*\n\s+/// Returns error if [^\n]*',
        r'\1',
        content
    )
    
    return content

def fix_liquidation_test(content):
    """Fix specific liquidation test issues"""
    
    # Fix the liquidation test that has unit type issues
    content = re.sub(
        r'(let coordinator = LiquidationCoordinator::new\([^)]*\);\s*assert!\(coordinator\.is_ok\(\)\);\s*)\n\s*if let Ok\(coordinator\) = coordinator \{\s*\n\s*(let start_result = coordinator\.start\(\);\s*assert!\(start_result\.is_ok\(\)\);\s*let stop_result = coordinator\.stop\(\)\.await;\s*assert!\(stop_result\.is_ok\(\)\);\s*assert!\(coordinator\.shutdown\.load\(Ordering::Relaxed\)\);)\s*\}',
        r'\1\n        if let Ok(coordinator) = coordinator {\n            \2\n        }',
        content,
        flags=re.DOTALL
    )
    
    return content

def fix_resource_allocator_test(content):
    """Fix resource allocator test issues"""
    
    # Fix the resource allocator tests that have unit type issues
    content = re.sub(
        r'(let allocation = allocator\.allocate_resources\([^)]*\);\s*assert!\(allocation\.is_ok\(\)\);\s*)let alloc = allocation\?;',
        r'\1if let Ok(alloc) = allocation {',
        content
    )
    
    # Add closing braces for the if let blocks
    content = re.sub(
        r'(assert_eq!\(alloc\.\w+, [^)]*\); // [^\n]*\n\s*assert_eq!\(alloc\.\w+, [^)]*\); // [^\n]*)',
        r'\1\n        }',
        content
    )
    
    return content

def process_file(filepath):
    """Process a single Rust file"""
    
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply all fixes
    content = fix_test_functions(content)
    content = fix_empty_doc_comments(content)
    content = fix_liquidation_test(content)
    content = fix_resource_allocator_test(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def main():
    """Main function"""
    
    print("🚀 TallyIO Strategy Core - Fix Test Functions")
    print("Fixing test functions with Result type issues...")
    
    # Find all Rust files
    rust_files = glob.glob('src/**/*.rs', recursive=True)
    
    for filepath in rust_files:
        process_file(filepath)
    
    print(f"✅ Fixed {len(rust_files)} files")
    print("🎯 Ready for clippy validation!")

if __name__ == '__main__':
    main()
