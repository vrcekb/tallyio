//! # Priority-based Execution Queue
//!
//! High-performance priority queue for strategy execution ordering.

use crate::{StrategyResult, StrategyPriority, types::*};
use std::collections::BinaryHeap;

/// Execution queue item
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueueItem {
    /// Strategy priority
    pub priority: StrategyPriority,
    /// Strategy identifier
    pub strategy_id: String,
    /// Expected profit
    pub expected_profit: ProfitAmount,
    /// Timestamp when added to queue
    pub timestamp: Timestamp,
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority (lower number) comes first
        other.priority.cmp(&self.priority)
            .then_with(|| self.expected_profit.cmp(&other.expected_profit))
            .then_with(|| other.timestamp.cmp(&self.timestamp))
    }
}

/// Priority-based execution queue
#[derive(Debug)]
#[non_exhaustive]
pub struct ExecutionQueue {
    /// Internal priority queue
    queue: BinaryHeap<QueueItem>,
    /// Maximum queue size
    max_size: usize,
}

impl ExecutionQueue {
    /// Create new execution queue
    #[inline]
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: BinaryHeap::with_capacity(max_size),
            max_size,
        }
    }
    
    /// Add item to queue
    #[inline]
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn push(&mut self, item: QueueItem) -> StrategyResult<()> {
        if self.queue.len() >= self.max_size {
            // Remove lowest priority item if queue is full
            if let Some(lowest) = self.queue.peek() {
                if item > *lowest {
                    self.queue.pop();
                    self.queue.push(item);
                }
            } else {
                self.queue.push(item);
            }
        } else {
            // Queue is not full, just add the item
            self.queue.push(item);
        }

        Ok(())
    }
    
    /// Pop highest priority item
    #[inline]
    pub fn pop(&mut self) -> Option<QueueItem> {
        self.queue.pop()
    }
    
    /// Get queue length
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    
    /// Check if queue is empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl Default for ExecutionQueue {
    #[inline]
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_queue_creation() {
        let queue = ExecutionQueue::new(100);
        assert_eq!(queue.max_size, 100);
        assert!(queue.is_empty());
    }

    #[test]
    fn execution_queue_push_pop() {
        let mut queue = ExecutionQueue::new(10);
        
        let item = QueueItem {
            priority: StrategyPriority::High,
            strategy_id: "test".to_owned(),
            expected_profit: 1000,
            timestamp: 12345,
        };
        
        let result = queue.push(item.clone());
        assert!(result.is_ok());
        assert_eq!(queue.len(), 1);
        
        let popped = queue.pop();
        assert_eq!(popped, Some(item));
        assert!(queue.is_empty());
    }
}
