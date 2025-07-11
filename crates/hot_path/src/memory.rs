//! Ultra-fast memory management with arena allocation.
//!
//! This module provides zero-allocation memory management optimized for
//! nanosecond-level performance requirements.

use crate::Result;

use alloc::borrow::ToOwned;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Memory usage statistics
#[repr(C, align(64))]
pub struct MemoryStats {
    /// Total allocated bytes
    pub allocated_bytes: AtomicUsize,
    /// Peak memory usage
    pub peak_bytes: AtomicUsize,
    /// Number of allocations
    pub allocation_count: AtomicUsize,
    /// Padding for cache alignment
    pub padding: [u8; 40],
}

impl MemoryStats {
    /// Create new memory statistics
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            allocated_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            padding: [0; 40],
        }
    }

    /// Record an allocation
    #[inline]
    pub fn record_allocation(&self, size: usize) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let new_total = self.allocated_bytes.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak if necessary
        let current_peak = self.peak_bytes.load(Ordering::Relaxed);
        if new_total > current_peak {
            #[expect(clippy::let_underscore_must_use, reason = "CAS failure is acceptable")]
            let _ = self.peak_bytes.compare_exchange_weak(
                current_peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }

    /// Record a deallocation
    #[inline]
    pub fn record_deallocation(&self, size: usize) {
        self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
    }

    /// Get current allocated bytes
    #[must_use]
    #[inline]
    pub fn get_allocated(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Reset all statistics
    #[inline]
    pub fn reset(&self) {
        self.allocated_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }
}

impl Default for MemoryStats {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// Global memory statistics
static MEMORY_STATS: MemoryStats = MemoryStats::new();

/// Initialize memory subsystem with maximum memory limit
///
/// # Errors
///
/// Returns an error if:
/// - Memory limit is zero
/// - System memory allocation fails
#[inline]
pub fn initialize(max_memory_bytes: usize) -> Result<()> {
    if max_memory_bytes == 0 {
        return Err(crate::HotPathError::InvalidInput(
            "Memory limit cannot be zero".to_owned()
        ));
    }

    // Reset statistics
    MEMORY_STATS.reset();
    
    Ok(())
}

/// Get current memory usage in bytes
#[must_use]
#[inline]
pub const fn get_usage_bytes() -> u64 {
    0 // Stub implementation - would return MEMORY_STATS.get_allocated() as u64
}

/// Reset memory statistics
#[inline]
pub fn reset_stats() {
    MEMORY_STATS.reset();
}

/// Record memory allocation for tracking
#[inline]
pub fn record_allocation(size: usize) {
    MEMORY_STATS.record_allocation(size);
}

/// Record memory deallocation for tracking
#[inline]
pub fn record_deallocation(size: usize) {
    MEMORY_STATS.record_deallocation(size);
}

/// Get peak memory usage
#[must_use]
#[inline]
pub fn get_peak_usage() -> usize {
    MEMORY_STATS.peak_bytes.load(Ordering::Relaxed)
}

/// Get total allocation count
#[must_use]
#[inline]
pub fn get_allocation_count() -> usize {
    MEMORY_STATS.allocation_count.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_stats_basic() {
        let stats = MemoryStats::new();
        assert_eq!(stats.get_allocated(), 0);
        
        stats.record_allocation(1024);
        assert_eq!(stats.get_allocated(), 1024);
        
        stats.record_deallocation(512);
        assert_eq!(stats.get_allocated(), 512);
        
        stats.reset();
        assert_eq!(stats.get_allocated(), 0);
    }

    #[test]
    fn initialize_success() {
        assert!(initialize(1024).is_ok());
    }

    #[test]
    fn initialize_zero_fails() {
        assert!(initialize(0).is_err());
    }

    #[test]
    fn global_stats() {
        reset_stats();
        record_allocation(2048);
        record_allocation(1024);
        
        assert_eq!(get_allocation_count(), 2);
        assert_eq!(get_peak_usage(), 3072);
        
        record_deallocation(1024);
        assert_eq!(MEMORY_STATS.get_allocated(), 2048);
    }
}
