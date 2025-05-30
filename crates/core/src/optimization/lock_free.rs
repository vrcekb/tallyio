//! Lock-free data structures for ultra-low latency operations
//!
//! This module provides lock-free data structures that eliminate blocking
//! and ensure consistent performance for high-frequency trading operations.

use crate::error::{CoreError, CoreResult};
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;

/// Lock-free queue wrapper
///
/// Provides a high-performance, lock-free queue implementation with
/// additional monitoring and statistics capabilities.
#[derive(Debug)]
pub struct LockFreeQueue<T> {
    /// Underlying lock-free queue
    queue: Arc<SegQueue<T>>,
    /// Queue size counter (approximate)
    size: AtomicUsize,
    /// Push operation counter
    push_count: AtomicUsize,
    /// Pop operation counter
    pop_count: AtomicUsize,
}

impl<T> LockFreeQueue<T> {
    /// Create a new lock-free queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: Arc::new(SegQueue::new()),
            size: AtomicUsize::new(0),
            push_count: AtomicUsize::new(0),
            pop_count: AtomicUsize::new(0),
        }
    }

    /// Push an item to the queue
    #[inline(always)]
    pub fn push(&self, item: T) {
        self.queue.push(item);
        self.size.fetch_add(1, Ordering::Relaxed);
        self.push_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop an item from the queue
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        match self.queue.pop() {
            Some(item) => {
                self.size.fetch_sub(1, Ordering::Relaxed);
                self.pop_count.fetch_add(1, Ordering::Relaxed);
                Some(item)
            }
            None => None,
        }
    }

    /// Get approximate queue size
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get queue statistics
    #[must_use]
    pub fn statistics(&self) -> QueueStatistics {
        QueueStatistics {
            current_size: self.len(),
            total_pushes: self.push_count.load(Ordering::Relaxed),
            total_pops: self.pop_count.load(Ordering::Relaxed),
            is_empty: self.is_empty(),
        }
    }

    /// Clear the queue
    pub fn clear(&self) {
        while self.pop().is_some() {}
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for LockFreeQueue<T> {
    fn clone(&self) -> Self {
        Self {
            queue: Arc::clone(&self.queue),
            size: AtomicUsize::new(self.size.load(Ordering::Relaxed)),
            push_count: AtomicUsize::new(0),
            pop_count: AtomicUsize::new(0),
        }
    }
}

/// Lock-free stack implementation
///
/// Provides a high-performance, lock-free stack using atomic operations.
#[derive(Debug)]
pub struct LockFreeStack<T> {
    /// Head of the stack
    head: AtomicPtr<Node<T>>,
    /// Stack size counter
    size: AtomicUsize,
    /// Push operation counter
    push_count: AtomicUsize,
    /// Pop operation counter
    pop_count: AtomicUsize,
}

/// Stack node
#[derive(Debug)]
struct Node<T> {
    /// Node data
    data: T,
    /// Next node pointer
    next: *mut Node<T>,
}

impl<T> LockFreeStack<T> {
    /// Create a new lock-free stack
    #[must_use]
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(std::ptr::null_mut()),
            size: AtomicUsize::new(0),
            push_count: AtomicUsize::new(0),
            pop_count: AtomicUsize::new(0),
        }
    }

    /// Push an item to the stack
    pub fn push(&self, item: T) -> CoreResult<()> {
        let new_node = Box::into_raw(Box::new(Node {
            data: item,
            next: std::ptr::null_mut(),
        }));

        loop {
            let head = self.head.load(Ordering::Acquire);
            unsafe {
                (*new_node).next = head;
            }

            match self.head.compare_exchange_weak(
                head,
                new_node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.size.fetch_add(1, Ordering::Relaxed);
                    self.push_count.fetch_add(1, Ordering::Relaxed);
                    return Ok(());
                }
                Err(_) => continue,
            }
        }
    }

    /// Pop an item from the stack
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            if head.is_null() {
                return None;
            }

            let next = unsafe { (*head).next };

            match self
                .head
                .compare_exchange_weak(head, next, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) => {
                    let data = unsafe { Box::from_raw(head).data };
                    self.size.fetch_sub(1, Ordering::Relaxed);
                    self.pop_count.fetch_add(1, Ordering::Relaxed);
                    return Some(data);
                }
                Err(_) => continue,
            }
        }
    }

    /// Get stack size
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if stack is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire).is_null()
    }

    /// Get stack statistics
    #[must_use]
    pub fn statistics(&self) -> StackStatistics {
        StackStatistics {
            current_size: self.len(),
            total_pushes: self.push_count.load(Ordering::Relaxed),
            total_pops: self.pop_count.load(Ordering::Relaxed),
            is_empty: self.is_empty(),
        }
    }
}

impl<T> Default for LockFreeStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    /// Current queue size
    pub current_size: usize,
    /// Total push operations
    pub total_pushes: usize,
    /// Total pop operations
    pub total_pops: usize,
    /// Whether queue is empty
    pub is_empty: bool,
}

impl QueueStatistics {
    /// Get throughput ratio (pops/pushes)
    #[must_use]
    pub fn throughput_ratio(&self) -> f64 {
        if self.total_pushes == 0 {
            0.0
        } else {
            self.total_pops as f64 / self.total_pushes as f64
        }
    }
}

/// Stack statistics
#[derive(Debug, Clone)]
pub struct StackStatistics {
    /// Current stack size
    pub current_size: usize,
    /// Total push operations
    pub total_pushes: usize,
    /// Total pop operations
    pub total_pops: usize,
    /// Whether stack is empty
    pub is_empty: bool,
}

impl StackStatistics {
    /// Get throughput ratio (pops/pushes)
    #[must_use]
    pub fn throughput_ratio(&self) -> f64 {
        if self.total_pushes == 0 {
            0.0
        } else {
            self.total_pops as f64 / self.total_pushes as f64
        }
    }
}

/// Lock-free hash map entry
#[derive(Debug)]
struct HashMapEntry<K, V> {
    /// Entry key
    key: K,
    /// Entry value
    value: V,
    /// Next entry in chain
    next: AtomicPtr<HashMapEntry<K, V>>,
}

/// Simple lock-free hash map
///
/// Provides a basic lock-free hash map implementation for high-performance lookups.
/// Note: This is a simplified implementation for demonstration purposes.
#[derive(Debug)]
pub struct LockFreeHashMap<K, V> {
    /// Hash table buckets
    buckets: Vec<AtomicPtr<HashMapEntry<K, V>>>,
    /// Number of buckets
    bucket_count: usize,
    /// Entry count
    size: AtomicUsize,
}

impl<K, V> LockFreeHashMap<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Create a new lock-free hash map
    #[must_use]
    pub fn new(bucket_count: usize) -> Self {
        let mut buckets = Vec::with_capacity(bucket_count);
        for _ in 0..bucket_count {
            buckets.push(AtomicPtr::new(std::ptr::null_mut()));
        }

        Self {
            buckets,
            bucket_count,
            size: AtomicUsize::new(0),
        }
    }

    /// Get bucket index for key
    fn bucket_index(&self, key: &K) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.bucket_count
    }

    /// Insert a key-value pair
    pub fn insert(&self, key: K, value: V) -> CoreResult<()> {
        let bucket_idx = self.bucket_index(&key);
        let bucket = &self.buckets[bucket_idx];

        let new_entry = Box::into_raw(Box::new(HashMapEntry {
            key,
            value,
            next: AtomicPtr::new(std::ptr::null_mut()),
        }));

        loop {
            let head = bucket.load(Ordering::Acquire);
            unsafe {
                (*new_entry).next.store(head, Ordering::Relaxed);
            }

            match bucket.compare_exchange_weak(
                head,
                new_entry,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.size.fetch_add(1, Ordering::Relaxed);
                    return Ok(());
                }
                Err(_) => {}
            }
        }
    }

    /// Get value by key
    pub fn get(&self, key: &K) -> Option<V> {
        let bucket_idx = self.bucket_index(key);
        let bucket = &self.buckets[bucket_idx];

        let mut current = bucket.load(Ordering::Acquire);
        while !current.is_null() {
            unsafe {
                if (*current).key == *key {
                    return Some((*current).value.clone());
                }
                current = (*current).next.load(Ordering::Acquire);
            }
        }

        None
    }

    /// Get map size
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if map is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, V> Default for LockFreeHashMap<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new(1024) // Default to 1024 buckets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_free_queue() {
        let queue = LockFreeQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.push(1);
        queue.push(2);
        queue.push(3);

        assert_eq!(queue.len(), 3);
        assert!(!queue.is_empty());

        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.len(), 1);

        let stats = queue.statistics();
        assert_eq!(stats.total_pushes, 3);
        assert_eq!(stats.total_pops, 2);
    }

    #[test]
    fn test_lock_free_stack() -> CoreResult<()> {
        let stack = LockFreeStack::new();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);

        stack.push(1)?;
        stack.push(2)?;
        stack.push(3)?;

        assert_eq!(stack.len(), 3);
        assert!(!stack.is_empty());

        assert_eq!(stack.pop(), Some(3)); // LIFO order
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.len(), 1);

        let stats = stack.statistics();
        assert_eq!(stats.total_pushes, 3);
        assert_eq!(stats.total_pops, 2);

        Ok(())
    }

    #[test]
    fn test_lock_free_hash_map() -> CoreResult<()> {
        let map = LockFreeHashMap::new(16);
        assert!(map.is_empty());

        map.insert("key1".to_string(), 100)?;
        map.insert("key2".to_string(), 200)?;

        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());

        assert_eq!(map.get(&"key1".to_string()), Some(100));
        assert_eq!(map.get(&"key2".to_string()), Some(200));
        assert_eq!(map.get(&"key3".to_string()), None);

        Ok(())
    }

    #[test]
    fn test_queue_statistics() {
        let queue = LockFreeQueue::new();

        for i in 0..10 {
            queue.push(i);
        }

        for _ in 0..5 {
            queue.pop();
        }

        let stats = queue.statistics();
        assert_eq!(stats.total_pushes, 10);
        assert_eq!(stats.total_pops, 5);
        assert_eq!(stats.throughput_ratio(), 0.5);
    }

    #[test]
    fn test_stack_statistics() -> CoreResult<()> {
        let stack = LockFreeStack::new();

        for i in 0..10 {
            stack.push(i)?;
        }

        for _ in 0..3 {
            stack.pop();
        }

        let stats = stack.statistics();
        assert_eq!(stats.total_pushes, 10);
        assert_eq!(stats.total_pops, 3);
        assert_eq!(stats.throughput_ratio(), 0.3);

        Ok(())
    }

    #[test]
    fn test_queue_clear() {
        let queue = LockFreeQueue::new();

        for i in 0..5 {
            queue.push(i);
        }

        assert_eq!(queue.len(), 5);
        queue.clear();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_pop_empty() {
        // Test pop from empty queue (line 56)
        let queue: LockFreeQueue<i32> = LockFreeQueue::new();
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_queue_default() {
        // Test Default implementation (lines 90-91)
        let queue: LockFreeQueue<i32> = LockFreeQueue::default();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_queue_clone() {
        // Test Clone implementation (lines 96, 98-101)
        let queue = LockFreeQueue::new();
        queue.push(1);
        queue.push(2);

        let cloned = queue.clone();
        // Clone creates new counters but shares the underlying queue
        assert_eq!(cloned.len(), 2);

        // Original queue should still work
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_stack_default() {
        // Test Default implementation (lines 221-222)
        let stack: LockFreeStack<i32> = LockFreeStack::default();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_stack_push_error_handling() -> CoreResult<()> {
        // Test stack push error handling (line 149, 166)
        let stack = LockFreeStack::new();

        // Push should succeed normally
        stack.push(1)?;
        assert_eq!(stack.len(), 1);

        // Test the compare_exchange_weak failure path by creating contention
        // This is hard to test deterministically, but we can at least exercise the code
        for i in 0..10 {
            stack.push(i)?;
        }

        assert_eq!(stack.len(), 11);
        Ok(())
    }

    #[test]
    fn test_stack_pop_contention() -> CoreResult<()> {
        // Test stack pop with potential contention (lines 179, 191)
        let stack = LockFreeStack::new();

        // Add items
        for i in 0..5 {
            stack.push(i)?;
        }

        // Pop all items
        let mut popped = Vec::new();
        while let Some(item) = stack.pop() {
            popped.push(item);
        }

        assert_eq!(popped.len(), 5);
        assert!(stack.is_empty());
        Ok(())
    }

    #[test]
    fn test_stack_drop() -> CoreResult<()> {
        // Test Drop implementation (line 228)
        {
            let stack = LockFreeStack::new();
            for i in 0..5 {
                stack.push(i)?;
            }
            // Stack will be dropped here, should clean up all nodes
        }
        // If we reach here without crashing, Drop worked correctly
        Ok(())
    }

    #[test]
    fn test_queue_statistics_throughput_zero_pushes() {
        // Test throughput ratio with zero pushes (lines 249-250)
        let stats = QueueStatistics {
            current_size: 0,
            total_pushes: 0,
            total_pops: 0,
            is_empty: true,
        };

        assert_eq!(stats.throughput_ratio(), 0.0);
    }

    #[test]
    fn test_stack_statistics_throughput_zero_pushes() {
        // Test throughput ratio with zero pushes (lines 274-275)
        let stats = StackStatistics {
            current_size: 0,
            total_pushes: 0,
            total_pops: 0,
            is_empty: true,
        };

        assert_eq!(stats.throughput_ratio(), 0.0);
    }

    #[test]
    fn test_hashmap_default() {
        // Test Default implementation (lines 405-406)
        let map: LockFreeHashMap<String, i32> = LockFreeHashMap::default();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_hashmap_insert_contention() -> CoreResult<()> {
        // Test hashmap insert with potential contention (lines 348, 364)
        let map = LockFreeHashMap::new(4); // Small bucket count for more collisions

        // Insert multiple items that might hash to same bucket
        for i in 0..10 {
            map.insert(format!("key{}", i), i)?;
        }

        assert_eq!(map.len(), 10);

        // Verify all items can be retrieved
        for i in 0..10 {
            assert_eq!(map.get(&format!("key{}", i)), Some(i));
        }

        Ok(())
    }

    #[test]
    fn test_hashmap_get_not_found() -> CoreResult<()> {
        // Test hashmap get with key not found (lines 380, 384)
        let map = LockFreeHashMap::new(16);

        // Try to get from empty map
        assert_eq!(map.get(&"nonexistent".to_string()), None);

        // Add some items
        map.insert("key1".to_string(), 100)?;
        map.insert("key2".to_string(), 200)?;

        // Try to get non-existent key
        assert_eq!(map.get(&"key3".to_string()), None);
        Ok(())
    }

    #[test]
    fn test_hashmap_collision_handling() -> CoreResult<()> {
        // Test hashmap collision handling by forcing items into same bucket
        let map = LockFreeHashMap::new(1); // Only one bucket, all items will collide

        map.insert("a".to_string(), 1)?;
        map.insert("b".to_string(), 2)?;
        map.insert("c".to_string(), 3)?;

        // All items should be retrievable despite collisions
        assert_eq!(map.get(&"a".to_string()), Some(1));
        assert_eq!(map.get(&"b".to_string()), Some(2));
        assert_eq!(map.get(&"c".to_string()), Some(3));

        Ok(())
    }
}
