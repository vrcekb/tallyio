//! TallyIO Core Prelude
//!
//! This module re-exports the most commonly used types and traits from the core crate.
//! Import this module to get access to all essential TallyIO core functionality.

// Re-export error types
pub use crate::error::{CoreError, CoreResult, CriticalError, CriticalResult};

// Re-export configuration
pub use crate::config::{CoreConfig, CoreConfigBuilder};

// Re-export core types
pub use crate::types::{
    Gas, Opportunity, OpportunityType, Price, ProcessingResult, Transaction, TransactionHash,
    TransactionStatus,
};

// Re-export engine components
pub use crate::engine::{EngineMetrics, EngineStatus, TallyEngine};

// Re-export state management
pub use crate::state::{GlobalState, LocalState, StateManager};

// Re-export mempool components
pub use crate::mempool::{MempoolAnalyzer, MempoolFilter, MempoolWatcher};

// Re-export optimization utilities
pub use crate::optimization::{CpuAffinity, LockFreeQueue, MemoryPool, SimdOps};

// Re-export commonly used standard library types
pub use std::sync::atomic::{AtomicU64, Ordering};
pub use std::time::{Duration, Instant};

// Re-export commonly used external types
pub use crossbeam::queue::SegQueue;
pub use uuid::Uuid;

/// Common result type for operations that can fail
pub type Result<T> = CoreResult<T>;

/// Trait for types that can be processed by the engine
pub trait Processable {
    /// Process this item and return the result
    fn process(&self) -> CoreResult<ProcessingResult>;

    /// Get the priority of this item for scheduling
    fn priority(&self) -> u8;

    /// Check if this item is time-sensitive
    fn is_time_sensitive(&self) -> bool;
}

/// Trait for types that can be analyzed for MEV opportunities
pub trait MevAnalyzable {
    /// Analyze this item for MEV opportunities
    fn analyze_mev(&self) -> CoreResult<Option<Opportunity>>;

    /// Check if this item is DeFi-related
    fn is_defi_related(&self) -> bool;

    /// Get the estimated gas cost
    fn estimated_gas(&self) -> Gas;
}

/// Trait for types that can be cached for performance
pub trait Cacheable {
    /// Get the cache key for this item
    fn cache_key(&self) -> String;

    /// Check if this item should be cached
    fn should_cache(&self) -> bool;

    /// Get the cache TTL for this item
    fn cache_ttl(&self) -> Duration;
}

/// Trait for types that can be monitored for performance metrics
pub trait Monitorable {
    /// Get the metric name for this item
    fn metric_name(&self) -> &'static str;

    /// Get additional metric labels
    fn metric_labels(&self) -> Vec<(&'static str, String)>;

    /// Record processing time for this item
    fn record_processing_time(&self, duration: Duration);
}

/// Macro for creating a new transaction with validation
#[macro_export]
macro_rules! transaction {
    ($from:expr, $to:expr, $value:expr, $gas_price:expr, $gas_limit:expr) => {
        $crate::types::Transaction::new(
            $from,
            Some($to),
            $crate::types::Price::new($value),
            $crate::types::Price::new($gas_price),
            $crate::types::Gas::new($gas_limit),
            0,
            Vec::with_capacity(0),
        )
    };
    ($from:expr, $to:expr, $value:expr, $gas_price:expr, $gas_limit:expr, $data:expr) => {
        $crate::types::Transaction::new(
            $from,
            Some($to),
            $crate::types::Price::new($value),
            $crate::types::Price::new($gas_price),
            $crate::types::Gas::new($gas_limit),
            0,
            $data,
        )
    };
}

/// Macro for creating a new opportunity
#[macro_export]
macro_rules! opportunity {
    ($type:expr, $value:expr, $gas:expr) => {
        $crate::types::Opportunity::new($type, $crate::types::Price::new($value), $gas)
    };
}

/// Macro for timing critical operations
#[macro_export]
macro_rules! time_critical {
    ($operation:expr) => {{
        let start = std::time::Instant::now();
        let result = $operation;
        let elapsed = start.elapsed();

        if elapsed > std::time::Duration::from_millis(1) {
            log::warn!("Critical operation exceeded 1ms: {:?}", elapsed);
        }

        result
    }};
}

/// Macro for creating atomic counters
#[macro_export]
macro_rules! atomic_counter {
    ($name:ident) => {
        static $name: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    };
}

/// Macro for incrementing atomic counters
#[macro_export]
macro_rules! increment_counter {
    ($counter:expr) => {
        $counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    };
}

/// Macro for getting atomic counter values
#[macro_export]
macro_rules! get_counter {
    ($counter:expr) => {
        $counter.load(std::sync::atomic::Ordering::Relaxed)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        // Test that all major types are available
        let _config = CoreConfig::default();
        let _error = CoreError::config("test");
        let _duration = Duration::from_millis(1);
        let _instant = Instant::now();
    }

    #[test]
    fn test_transaction_macro() {
        let tx = transaction!(
            [1u8; 20],
            [2u8; 20],
            1_000_000_000_000_000_000u64, // 1 ETH
            20_000_000_000u64,            // 20 gwei
            21_000u64                     // gas limit
        );

        assert_eq!(tx.value().as_wei(), 1_000_000_000_000_000_000);
        assert_eq!(tx.gas_price().as_wei(), 20_000_000_000);
        assert_eq!(tx.gas_limit().as_units(), 21_000);
    }

    #[test]
    fn test_opportunity_macro() {
        let opp = opportunity!(
            OpportunityType::Arbitrage,
            500_000_000_000_000_000u64, // 0.5 ETH
            Gas::new(100_000)
        );

        assert_eq!(opp.value().as_wei(), 500_000_000_000_000_000);
        assert_eq!(opp.gas_cost().as_units(), 100_000);
    }

    #[test]
    fn test_atomic_counter_macros() {
        atomic_counter!(TEST_COUNTER);

        let initial = get_counter!(TEST_COUNTER);
        increment_counter!(TEST_COUNTER);
        let after = get_counter!(TEST_COUNTER);

        assert_eq!(after, initial + 1);
    }

    #[test]
    fn test_time_critical_macro() {
        let result = time_critical!({
            // Simulate fast operation
            std::thread::sleep(Duration::from_micros(100));
            42
        });

        assert_eq!(result, 42);
    }
}
