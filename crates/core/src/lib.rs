//! `TallyIO` Core - Ultra-performant <1ms latency financial engine
//!
//! This crate provides the foundational types and operations for `TallyIO`'s
//! high-frequency trading and MEV extraction systems.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_const_for_fn,
    clippy::derivable_impls
)]
#![allow(
    clippy::inline_always,
    clippy::cast_possible_truncation,
    clippy::doc_markdown
)]
#![allow(
    clippy::uninlined_format_args,
    clippy::single_match,
    clippy::similar_names
)]
#![allow(clippy::needless_pass_by_ref_mut, clippy::unnecessary_wraps)]
#![allow(clippy::unchecked_duration_subtraction, clippy::cast_precision_loss)]
#![allow(
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::redundant_closure_for_method_calls
)]
#![allow(
    clippy::needless_pass_by_value,
    clippy::redundant_clone,
    clippy::unused_self
)]
#![allow(
    dead_code,
    unused_imports,
    unused_mut,
    unused_variables,
    clippy::needless_continue,
    clippy::unreadable_literal,
    clippy::must_use_candidate,
    clippy::match_same_arms,
    clippy::struct_field_names,
    clippy::cast_sign_loss,
    clippy::default_numeric_fallback,
    clippy::if_not_else,
    clippy::trivially_copy_pass_by_ref,
    clippy::struct_excessive_bools,
    clippy::float_cmp,
    clippy::used_underscore_binding,
    clippy::no_effect_underscore_binding,
    clippy::disallowed_methods,
    clippy::field_reassign_with_default,
    clippy::cast_lossless,
    clippy::overly_complex_bool_expr
)]

// #[global_allocator]
// static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;  // Disabled on Windows

use std::sync::atomic::{AtomicU64, Ordering};

pub mod config;
pub mod engine;
pub mod error;
pub mod prelude;
pub mod types;

// Engine modules
pub mod mempool;
pub mod optimization;
pub mod state;
pub mod utils;

// Re-export main types and functions
pub use config::{CoreConfig, CoreConfigBuilder};
pub use engine::TallyEngine;
pub use error::{CoreError, CoreResult, CriticalError, CriticalResult};
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
    use crate::engine::Controllable;
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
        let mut engine = TallyEngine::new()?;

        // Start the engine to make it operational
        engine.start()?;

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
    fn test_mev_scanning_latency() -> Result<(), CoreError> {
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
        let mut engine = TallyEngine::new()?;

        // Start the engine to make it operational
        engine.start()?;

        let _opportunities = engine.scan_mev_opportunity(&tx)?;
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_millis(2),
            "MEV scanning took {elapsed:?}"
        );
        Ok(())
    }
}
