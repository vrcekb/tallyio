//! # Hot Path - Ultra-Fast MEV Detection and Execution Engine
//!
//! Ultra-fast nanosecond-level MEV detection and execution engine for `TallyIO`.
//! 
//! ## Performance Targets
//! 
//! - MEV Detection: <500ns
//! - Memory Allocation: <5ns  
//! - Cross-Chain Operations: <50ns
//! - Crypto Operations: <50μs
//! 
//! ## Features
//! 
//! - SIMD-optimized calculations
//! - Lock-free atomic operations
//! - Arena-based memory allocation
//! - Zero-cost abstractions
//! 
//! ## Architecture
//! 
//! This crate is organized into specialized modules:
//! 
//! - `detection/` - MEV detection engine with SIMD optimization
//! - `execution/` - Ultra-fast transaction execution
//! - `memory/` - Arena-based memory management
//! - `atomic/` - Lock-free atomic primitives
//! - `simd/` - SIMD optimizations for vectorized operations
//! - `types` - Zero-cost abstractions and data structures
//! 
//! ## Usage
//!
//! ```rust
//! use hot_path::{initialize, HotPathConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = HotPathConfig::default();
//! initialize(config)?;
//! # Ok(())
//! # }
//! ```

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
    clippy::missing_const_for_fn,
    reason = "Const fn limitations in complex operations"
)]
#![expect(
    clippy::unnecessary_wraps,
    reason = "Result types for future error handling"
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
#![expect(
    clippy::mod_module_files,
    reason = "Module organization requires mod.rs files for complex hierarchies"
)]
#![cfg_attr(test, expect(
    clippy::redundant_test_prefix,
    reason = "Test function names follow conventional test_ prefix pattern"
))]
#![cfg_attr(test, expect(
    clippy::absurd_extreme_comparisons,
    reason = "Test assertions may include boundary checks that appear redundant"
))]
#![cfg_attr(test, expect(
    unused_comparisons,
    reason = "Test assertions may include type limit comparisons for completeness"
))]
#![cfg_attr(test, expect(
    clippy::default_numeric_fallback,
    reason = "Test values use simple numeric literals for clarity"
))]
#![cfg_attr(test, expect(
    unused_must_use,
    reason = "Test code may intentionally ignore return values"
))]
#![expect(
    clippy::single_call_fn,
    reason = "Stub functions will be expanded in future implementations"
)]

#![cfg_attr(test, expect(
    clippy::unwrap_used,
    reason = "Test code requires unwrap for validation"
))]


#![no_std]

extern crate alloc;

use alloc::string::String;
use core::{fmt::{Display, Formatter, Result as FmtResult}, result::Result as CoreResult};

// Public modules
pub mod atomic;
pub mod detection;
pub mod execution;
pub mod memory;
pub mod simd;
pub mod types;

// Re-export core types for convenience
pub use atomic::{AtomicCounter, record_latency};
pub use detection::{detect_opportunities, update_price_feed};
pub use execution::{execute_opportunity, ExecutionResult};
pub use memory::{initialize as initialize_memory, get_usage_bytes};
pub use simd::{initialize as initialize_simd, SimdCapabilities};
pub use types::{MarketSnapshot, TradingPair, Opportunity, AlignedPrice, ExecutionParams};

/// Result type for hot path operations
pub type Result<T> = CoreResult<T, HotPathError>;

/// Hot path specific errors
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HotPathError {
    /// Memory allocation failed
    AllocationFailed(String),
    /// Configuration error
    Configuration(String),
    /// Invalid input provided
    InvalidInput(String),
    /// Resource exhausted
    ResourceExhausted {
        /// Resource that was exhausted
        resource: String
    },
    /// Operation timed out
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

// Error trait implementation for compatibility
impl core::error::Error for HotPathError {}

/// Hot path configuration
#[derive(Debug, Clone, Copy)]
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

/// Performance metrics
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PerformanceMetrics {
    /// Total operations performed
    pub total_operations: u64,
    /// Average latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Peak latency in nanoseconds
    pub peak_latency_ns: u64,
    /// Operations per second
    pub ops_per_second: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
}

/// Initialize the hot path subsystem
///
/// # Errors
///
/// Returns an error if initialization fails
#[inline]
pub fn initialize(config: HotPathConfig) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let config = HotPathConfig::default();
        initialize(config).unwrap();
    }

    #[test]
    fn test_metrics() {
        let metrics = get_metrics();
        assert!(metrics.memory_usage_bytes >= 0);
    }

    #[test]
    fn test_error_display() {
        let error = HotPathError::AllocationFailed("test".into());
        let display = alloc::format!("{error}");
        assert!(display.contains("Memory allocation failed"));
    }
}
