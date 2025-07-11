#!/usr/bin/env python3
"""
TallyIO Strategy Core - Fix Compilation Errors Script
Fixes compilation errors caused by ultra-strict clippy fix.
"""

import os
import re
import glob

def fix_empty_doc_lines(content):
    """Fix empty lines after doc comments"""
    
    # Remove empty lines in doc comments
    content = re.sub(
        r'(\s+/// # Errors\s*\n)\s+///\s*\n\s+///\s*\n\s+/// Returns error if operation fails\s*\n\s*\n',
        r'\1    ///\n    /// Returns error if operation fails\n',
        content
    )
    
    return content

def fix_return_types(content):
    """Fix functions that should return Ok(())"""
    
    # Add Ok(()) to functions that need it
    patterns = [
        (r'(pub fn start\(&self\) -> StrategyResult<\(\)> \{\s*tracing::info!\([^)]+\);\s*)\}',
         r'\1\n        Ok(())\n    }'),
        (r'(pub fn stop\(&self\) -> StrategyResult<\(\)> \{\s*tracing::info!\([^)]+\);\s*)\}',
         r'\1\n        Ok(())\n    }'),
        (r'(pub async fn stop\(&self\) -> StrategyResult<\(\)> \{[^}]+)\}',
         r'\1\n        Ok(())\n    }'),
        (r'(pub fn init_strategy_core\([^)]+\) -> StrategyResult<\(\)> \{[^}]+)\}',
         r'\1\n    Ok(())\n}'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    return content

def fix_execution_queue(content):
    """Fix execution queue push method"""
    
    # Fix the push method to return Ok(())
    content = re.sub(
        r'(pub fn push\(&mut self, item: QueueItem\) -> StrategyResult<\(\)> \{[^}]+)\}',
        r'\1\n        Ok(())\n    }',
        content,
        flags=re.DOTALL
    )
    
    return content

def fix_yield_optimizer(content):
    """Fix yield optimizer constructor"""
    
    # Fix the constructor parameter
    content = re.sub(
        r'pub const fn new\(max_parallel: usize\) -> Self \{\s*Self \{ max_parallel \}\s*\}',
        r'pub const fn new(strategy: YieldStrategy) -> Self {\n        Self { strategy }\n    }',
        content
    )
    
    return content

def fix_unused_variables(content):
    """Fix unused variables"""
    
    # Add underscore prefix to unused variables
    content = re.sub(
        r'pub fn resolve_conflicts\(&self, conflicts: &\[StrategyConflict\]\)',
        r'pub fn resolve_conflicts(&self, _conflicts: &[StrategyConflict])',
        content
    )
    
    return content

def process_file(filepath):
    """Process a single Rust file"""
    
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply all fixes
    content = fix_empty_doc_lines(content)
    content = fix_return_types(content)
    content = fix_execution_queue(content)
    content = fix_yield_optimizer(content)
    content = fix_unused_variables(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def main():
    """Main function"""
    
    print("🚀 TallyIO Strategy Core - Fix Compilation Errors")
    print("Fixing compilation errors caused by ultra-strict clippy fix...")
    
    # Find all Rust files
    rust_files = glob.glob('src/**/*.rs', recursive=True)
    
    for filepath in rust_files:
        process_file(filepath)
    
    print(f"✅ Fixed {len(rust_files)} files")
    print("🎯 Ready for compilation!")

if __name__ == '__main__':
    main()
