//! # `TallyIO` Hot Path - Ultra-Fast MEV Detection and Execution Engine
//!
//! This crate provides nanosecond-level performance for MEV (Maximal Extractable Value)
//! detection and execution in `DeFi` protocols. It is designed for ultra-high frequency
//! trading with strict latency requirements.
//!
//! ## Performance Targets
//!
//! - MEV Detection: <500ns
//! - Memory Allocation: <5ns
//! - Cross-Chain Operations: <50ns
//! - Crypto Operations: <50μs
//!
//! ## Architecture
//!
//! The crate is organized into several high-performance modules:
//!
//! - `detection`: MEV opportunity scanning and price monitoring
//! - `execution`: Lock-free transaction execution and gas optimization
//! - `memory`: Ultra-fast memory management with arena allocation
//! - `atomic`: Lock-free atomic primitives and data structures
//! - `simd`: SIMD-optimized calculations and operations
//!
//! ## Safety
//!
//! This crate uses `#![forbid(unsafe_code)]` to ensure memory safety while
//! maintaining maximum performance through zero-cost abstractions.

#![forbid(unsafe_code)]
#![deny(
    missing_docs,
    unused_imports,
    unused_variables,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented
)]
#![expect(
    clippy::blanket_clippy_restriction_lints,
    reason = "Ultra-strict clippy profile requires selective restriction allowances"
)]
#![expect(
    clippy::arithmetic_side_effects,
    reason = "Financial calculations require arithmetic operations"
)]
#![expect(
    clippy::integer_division,
    reason = "Performance calculations require integer division"
)]
#![expect(
    clippy::integer_division_remainder_used,
    reason = "Performance calculations require division operations"
)]
#![expect(
    clippy::implicit_return,
    reason = "Rust idiom allows implicit returns"
)]
#![expect(
    clippy::needless_return,
    reason = "Explicit returns improve clarity in financial code"
)]
#![expect(
    clippy::exhaustive_structs,
    reason = "Performance-critical structs need direct field access"
)]
#![expect(
    clippy::absolute_paths,
    reason = "Core library paths are standard and clear"
)]
#![expect(
    clippy::question_mark_used,
    reason = "Error propagation is essential for financial operations"
)]
#![expect(
    clippy::pub_use,
    reason = "Public API convenience re-exports"
)]

#![expect(
    clippy::multiple_crate_versions,
    reason = "Dependency version conflicts are external"
)]
#![expect(
    clippy::arbitrary_source_item_ordering,
    reason = "Logical grouping more important than alphabetical"
)]

#![expect(
    clippy::let_underscore_untyped,
    reason = "Underscore bindings for unused values"
)]
#![expect(
    clippy::unused_trait_names,
    reason = "Traits imported for convenience"
)]
#![expect(
    clippy::missing_const_for_fn,
    reason = "Const fn limitations in complex operations"
)]
#![expect(
    clippy::unnecessary_wraps,
    reason = "Result types for future error handling"
)]
#![expect(
    clippy::default_numeric_fallback,
    reason = "Numeric types are context-appropriate"
)]
#![expect(
    clippy::manual_abs_diff,
    reason = "Manual implementation for clarity"
)]

#![expect(
    clippy::min_ident_chars,
    reason = "Standard single-char parameter names"
)]
#![expect(
    clippy::pattern_type_mismatch,
    reason = "Standard match patterns for error handling"
)]
#![expect(
    clippy::let_underscore_must_use,
    reason = "Intentional discard of return values"
)]

#![cfg_attr(test, expect(
    clippy::assertions_on_result_states,
    reason = "Test assertions require result checking"
))]



#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{borrow::ToOwned as _, string::String};
use core::{fmt::{Display, Formatter, Result as FmtResult}, result::Result as CoreResult};

// Public modules
pub mod atomic;
pub mod detection;
pub mod execution;
pub mod memory;
pub mod simd;
pub mod types;

// Re-export core types for convenience
pub use types::{
    AlignedPrice, AtomicCounter, ExecutionParams, MarketSnapshot,
    Opportunity, TradingPair, get_timestamp_ns, ATOMIC_ORDERING,
    CACHE_LINE_SIZE, MAX_CHAINS, MAX_TRADING_PAIRS
};

/// Result type used throughout the hot path crate
pub type Result<T> = CoreResult<T, HotPathError>;

/// Error types for hot path operations
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HotPathError {
    /// Memory allocation failed
    AllocationFailed(String),

    /// Configuration error
    Configuration(String),

    /// Invalid input parameters
    InvalidInput(String),

    /// Resource exhausted
    ResourceExhausted {
        /// Name of the exhausted resource
        resource: String
    },

    /// Operation timeout
    Timeout {
        /// Timeout duration in nanoseconds
        timeout_ns: u64
    },
}

impl Display for HotPathError {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::AllocationFailed(msg) => return write!(f, "Memory allocation failed: {msg}"),
            Self::Configuration(msg) => return write!(f, "Configuration error: {msg}"),
            Self::InvalidInput(msg) => return write!(f, "Invalid input: {msg}"),
            Self::ResourceExhausted { resource } => return write!(f, "Resource exhausted: {resource}"),
            Self::Timeout { timeout_ns } => return write!(f, "Operation timed out after {timeout_ns}ns"),
        }
    }
}

/// Performance metrics for hot path operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct PerformanceMetrics {
    /// Average latency in nanoseconds
    pub avg_latency_ns: u64,

    /// Memory usage in bytes
    pub memory_usage_bytes: u64,

    /// Operations per second
    pub ops_per_second: u64,

    /// Peak latency in nanoseconds
    pub peak_latency_ns: u64,

    /// Total operations performed
    pub total_operations: u64,
}

/// Configuration for hot path operations
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HotPathConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,

    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,

    /// Target latency in nanoseconds
    pub target_latency_ns: u64,

    /// Number of worker threads
    pub worker_threads: usize,
}

impl Default for HotPathConfig {
    #[inline]
    fn default() -> Self {
        return Self {
            enable_simd: true,
            max_memory_bytes: 1_024 * 1_024 * 1_024, // 1GB
            target_latency_ns: 500, // 500ns target
            worker_threads: num_cpus::get(),
        };
    }
}

/// Initialize the hot path system with given configuration
///
/// # Errors
///
/// Returns an error if:
/// - Configuration validation fails
/// - Memory subsystem initialization fails
/// - Atomic subsystem initialization fails
/// - SIMD initialization fails (when enabled)
#[inline]
pub fn initialize(config: &HotPathConfig) -> Result<()> {
    // Validate configuration
    if config.target_latency_ns == 0 {
        return Err(HotPathError::InvalidInput("Target latency cannot be zero".to_owned()));
    }

    if config.worker_threads == 0 {
        return Err(HotPathError::InvalidInput("Worker threads cannot be zero".to_owned()));
    }

    // Initialize memory subsystem
    #[expect(clippy::question_mark_used, reason = "Error propagation pattern")]
    #[expect(clippy::semicolon_outside_block, reason = "Block scoping for error handling")]
    {
        memory::initialize(config.max_memory_bytes)?;
    }

    // Initialize atomic subsystem
    #[expect(clippy::question_mark_used, reason = "Error propagation pattern")]
    #[expect(clippy::semicolon_outside_block, reason = "Block scoping for error handling")]
    {
        atomic::initialize()?;
    }

    // Initialize SIMD if enabled
    if config.enable_simd {
        #[expect(clippy::question_mark_used, reason = "Error propagation pattern")]
        {
            simd::initialize()?;
        }
    }

    return Ok(());
}

/// Get current performance metrics
#[must_use]
#[inline]
pub const fn get_metrics() -> PerformanceMetrics {
    return PerformanceMetrics {
        total_operations: atomic::get_operation_count(),
        avg_latency_ns: atomic::get_avg_latency(),
        peak_latency_ns: atomic::get_peak_latency(),
        ops_per_second: atomic::get_ops_per_second(),
        memory_usage_bytes: memory::get_usage_bytes(),
    };
}

/// Reset all performance counters
#[inline]
pub fn reset_metrics() {
    atomic::reset_counters();
    memory::reset_stats();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = HotPathConfig::default();
        assert!(config.enable_simd);
        assert!(config.max_memory_bytes > 0);
        assert!(config.target_latency_ns > 0);
        assert!(config.worker_threads > 0);
    }

    #[test]
    fn invalid_config() {
        let config = HotPathConfig {
            target_latency_ns: 0,
            ..Default::default()
        };

        assert!(initialize(&config).is_err());
    }

    #[test]
    fn metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.total_operations, 0);
        assert_eq!(metrics.avg_latency_ns, 0);
    }
}







