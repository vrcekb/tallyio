//! `TallyIO` Core - Ultra-performant <1ms latency financial engine
//!
//! This crate provides the foundational types and operations for `TallyIO`'s
//! high-frequency trading and MEV extraction systems.

// #[global_allocator]
// static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;  // Disabled on Windows

use std::sync::atomic::{AtomicU64, Ordering};

pub mod engine;
pub mod error;
pub mod types;
pub mod utils;

pub use engine::TallyEngine;
pub use error::{CoreError, CriticalError};
pub use types::*;

/// Global performance counter for monitoring
static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Get current operation count
///
/// Returns the total number of operations processed since startup.
/// This is a lock-free atomic operation suitable for high-frequency monitoring.
#[must_use]
pub fn operation_count() -> u64 {
    OPERATION_COUNTER.load(Ordering::Relaxed)
}

/// Increment operation counter
///
/// Atomically increments the global operation counter.
/// This is called on every critical operation for performance tracking.
pub fn increment_operations() {
    OPERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_operation_counter() {
        let initial = operation_count();
        increment_operations();
        assert_eq!(operation_count(), initial + 1);
    }

    #[test]
    fn test_latency_requirement() {
        let start = Instant::now();
        increment_operations();
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(1),
            "Operation took {elapsed:?}"
        );
    }

    #[test]
    fn test_engine_latency() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(1_000_000_000_000_000_000), // 1 ETH
            Price::new(20_000_000_000),            // 20 gwei
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let start = Instant::now();
        engine.submit_transaction(tx)?;
        let _result = engine.process_next();
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_millis(1),
            "Engine processing took {elapsed:?}"
        );
        Ok(())
    }

    #[test]
    fn test_mev_scanning_latency() {
        let mut data = Vec::with_capacity(100);
        data.extend_from_slice(&[0xa9, 0x05, 0x9c, 0xbb]); // swapExactTokensForTokens
        data.resize(100, 0);

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(2_000_000_000_000_000_000), // 2 ETH
            Price::new(60_000_000_000),            // 60 gwei
            Gas::new(200_000),
            0,
            data,
        );

        let start = Instant::now();
        let _opportunity = TallyEngine::scan_mev_opportunity(&tx);
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_micros(100),
            "MEV scanning took {elapsed:?}"
        );
    }
}
