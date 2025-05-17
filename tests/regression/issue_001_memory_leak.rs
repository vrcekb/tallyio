// Regression test for issue #001: Memory leak in Arena when allocating large objects
//
// This test verifies that a previously fixed memory leak issue doesn't reoccur.
// The issue was that when allocating large objects in the Arena, the memory
// wasn't properly tracked, leading to a memory leak.

use core::Arena;
use std::sync::Arc;
use std::thread;

#[test]
fn test_regression_issue_001() {
    // Create an arena
    let arena = Arc::new(Arena::new());
    
    // Allocate a large number of objects in multiple threads
    let mut handles = vec![];
    for _ in 0..10 {
        let arena_clone = Arc::clone(&arena);
        handles.push(thread::spawn(move || {
            // Allocate 1000 large objects
            for i in 0..1000 {
                // Create a large object (a vector with 1000 elements)
                let large_object = vec![i; 1000];
                // Allocate it in the arena
                arena_clone.alloc(large_object);
            }
        }));
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify that the allocation count is correct
    // 10 threads * 1000 allocations = 10000
    assert_eq!(arena.allocation_count(), 10000);
    
    // In the buggy implementation, the allocation count would be incorrect
    // or the test would crash due to memory issues
}
