//! Atomic operations and lock-free data structures for ultra-high performance.
//!
//! This module provides lock-free atomic primitives optimized for AMD EPYC 9454P
//! with nanosecond-level precision requirements.

use crate::Result;

// Sub-modules
pub mod counters;
pub mod queues;
pub mod state;

// Re-export key types
pub use counters::{AtomicCounter, record_latency};
pub use queues::{LockFreeQueue, QueueError};
pub use state::{AtomicStateMachine, StateTransition};

/// Initialize atomic subsystem
///
/// # Errors
///
/// Returns an error if atomic initialization fails
#[inline]
pub const fn initialize() -> Result<()> {
    return Ok(());
}

/// Get total operation count
#[must_use]
#[inline]
pub const fn get_operation_count() -> u64 {
    return counters::get_operation_count();
}

/// Get average latency in nanoseconds
#[must_use]
#[inline]
pub const fn get_avg_latency() -> u64 {
    return counters::get_avg_latency();
}

/// Get peak latency in nanoseconds
#[must_use]
#[inline]
pub const fn get_peak_latency() -> u64 {
    return counters::get_peak_latency();
}

/// Get operations per second
#[must_use]
#[inline]
pub const fn get_ops_per_second() -> u64 {
    return counters::get_ops_per_second();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_initialization() {
        initialize().unwrap();
    }

    #[test]
    fn test_metrics_functions() {
        let _ = get_operation_count();
        let _ = get_avg_latency();
        let _ = get_peak_latency();
        let _ = get_ops_per_second();
    }
}
