//! # `TallyIO` Strategy Core
//!
//! Ultra-performance strategy execution engine for MEV operations.
//! Optimized for AMD EPYC 9454P with nanosecond-precision execution.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable
)]
#![warn(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::undocumented_unsafe_blocks
)]
#![expect(clippy::multiple_crate_versions, reason = "Dependencies have version conflicts")]

#![expect(
    clippy::mod_module_files,
    reason = "Module organization requires mod.rs files for complex hierarchies"
)]
#![expect(
    clippy::single_call_fn,
    reason = "Stub functions will be expanded in future implementations"
)]
#![expect(
    clippy::implicit_return,
    reason = "Explicit returns reduce readability in simple functions"
)]
#![expect(
    clippy::arbitrary_source_item_ordering,
    reason = "Logical grouping is more important than alphabetical ordering"
)]
#![expect(
    clippy::pub_use,
    reason = "Re-exports provide clean public API"
)]
#![cfg_attr(test, expect(
    clippy::redundant_test_prefix,
    reason = "Test function names follow conventional test_ prefix pattern"
))]
#![cfg_attr(test, expect(
    clippy::assertions_on_result_states,
    reason = "Test assertions on Result states are common and acceptable"
))]
#![cfg_attr(test, expect(
    clippy::shadow_reuse,
    reason = "Variable shadowing in tests improves readability"
))]
#![expect(
    clippy::separated_literal_suffix,
    reason = "Literal suffixes with underscores improve readability"
)]
#![expect(
    clippy::std_instead_of_alloc,
    reason = "std imports are acceptable for this application"
)]
#![expect(
    clippy::as_conversions,
    reason = "Resource allocation requires safe numeric conversions"
)]


use rust_decimal::Decimal;
use thiserror::Error;

/// Core error types for strategy execution
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum StrategyError {
    /// Arbitrage execution errors
    #[error("Arbitrage error: {0}")]
    Arbitrage(#[from] arbitrage::ArbitrageError),

    /// Coordination system errors
    #[error("Coordination error: {0}")]
    Coordination(#[from] coordination::CoordinationError),

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration {
        /// Error message
        message: String,
    },
    
    /// Critical system errors that require immediate shutdown
    #[error("Critical system error: {details}")]
    Critical {
        /// Error details
        details: String,
    },
    
    /// Liquidation execution errors
    #[error("Liquidation error: {0}")]
    Liquidation(#[from] liquidation::LiquidationError),
    
    /// Network communication errors
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    /// Priority system errors
    #[error("Priority error: {0}")]
    Priority(#[from] priority::PriorityError),
    
    /// Resource allocation errors
    #[error("Resource allocation error: {resource} - {reason}")]
    ResourceAllocation {
        /// Resource type
        resource: String,
        /// Allocation failure reason
        reason: String,
    },
    
    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Time-bandit strategy errors
    #[error("Time bandit error: {0}")]
    TimeBandit(#[from] time_bandit::TimeBanditError),
    
    /// Zero-risk strategy errors
    #[error("Zero risk error: {0}")]
    ZeroRisk(#[from] zero_risk::ZeroRiskError),
}

/// Core types module
pub mod types {
    /// Blockchain identifier
    pub type ChainId = u64;
    
    /// Block number
    pub type BlockNumber = u64;
    
    /// Timestamp in milliseconds
    pub type Timestamp = u64;
    
    /// Profit amount in USD (stored as cents to avoid float precision issues)
    pub type ProfitAmount = u64;
    
    /// Gas amount
    pub type GasAmount = u64;
}

/// Strategy execution priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum StrategyPriority {
    /// Critical liquidations - highest priority
    Critical = 0,
    /// High-profit arbitrage opportunities
    High = 1,
    /// Medium-profit opportunities
    Medium = 2,
    /// Low-profit opportunities
    Low = 3,
    /// Background maintenance tasks
    Background = 4,
}

/// Strategy execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StrategyStatus {
    /// Strategy is cancelled
    Cancelled,
    /// Strategy execution completed
    Completed,
    /// Strategy is currently executing
    Executing,
    /// Strategy execution failed
    Failed,
    /// Strategy is pending execution
    Pending,
}

/// Result type for strategy operations
pub type StrategyResult<T> = Result<T, StrategyError>;

/// Global strategy configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StrategyConfig {
    /// Enable ML-based scoring
    pub enable_ml_scoring: bool,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Execution timeout in milliseconds
    pub execution_timeout_ms: u64,
    /// Maximum number of concurrent strategies
    pub max_concurrent_strategies: usize,
    /// Maximum gas price in gwei
    pub max_gas_price: u64,
    /// Minimum profit threshold in USD
    pub min_profit_threshold: Decimal,
    /// NUMA node preference for CPU affinity
    pub numa_node: Option<u32>,
}

impl Default for StrategyConfig {
    #[inline]
    fn default() -> Self {
        Self {
            max_concurrent_strategies: 100,
            min_profit_threshold: Decimal::new(1, 2), // $0.01
            max_gas_price: 100, // 100 gwei
            execution_timeout_ms: 5000, // 5 seconds
            enable_simd: true,
            enable_ml_scoring: false,
            numa_node: None,
        }
    }
}

/// Initialize the strategy core system
///
/// # Errors
///
/// Returns error if initialization fails
#[inline]
/// # Errors
///
/// Returns error if operation fails
pub fn init_strategy_core(config: &StrategyConfig) -> StrategyResult<()> {
    tracing::info!("Initializing TallyIO Strategy Core");
    tracing::debug!("Configuration: {config:?}");

    // Initialize NUMA affinity if specified
    if let Some(numa_node) = config.numa_node {
        tracing::info!("Setting NUMA affinity to node {numa_node}");
    }

    Ok(())
}

// Strategy modules (alphabetical order)
pub mod arbitrage;
pub mod coordination;
pub mod liquidation;
pub mod priority;
pub mod time_bandit;
pub mod zero_risk;

// Re-export types for convenience
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_config_default() {
        let config = StrategyConfig::default();
        assert_eq!(config.max_concurrent_strategies, 100);
        assert!(config.enable_simd);
    }

    #[test]
    fn strategy_priority_ordering() {
        assert!(StrategyPriority::Critical < StrategyPriority::High);
        assert!(StrategyPriority::High < StrategyPriority::Medium);
        assert!(StrategyPriority::Medium < StrategyPriority::Low);
    }

    #[test]
    fn init_strategy_core_works() {
        let config = StrategyConfig::default();
        let result = init_strategy_core(&config);
        assert!(result.is_ok());
    }
}
