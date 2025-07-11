//! Ultra-fast transaction execution engine.
//!
//! This module provides lock-free transaction execution with real-time
//! gas optimization and MEV bundle construction.

use crate::{Result, types::{Opportunity, get_timestamp_ns}};
use alloc::string::String;

// Sub-modules
pub mod atomic_executor;
pub mod gas_optimizer;
pub mod bundle_builder;

// Re-export key types
pub use atomic_executor::{AtomicExecutor, ExecutionContext};
pub use gas_optimizer::{GasOptimizer, GasEstimate};
pub use bundle_builder::{BundleBuilder, Bundle};

/// Execution result
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExecutionResult {
    /// Transaction hash
    pub tx_hash: String,
    /// Gas used
    pub gas_used: u64,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ExecutionResult {
    /// Create a successful execution result
    #[must_use]
    #[inline]
    pub fn success(tx_hash: String, gas_used: u64, execution_time_ns: u64) -> Self {
        return Self {
            tx_hash,
            gas_used,
            execution_time_ns,
            success: true,
            error: None,
        };
    }

    /// Create a failed execution result
    #[must_use]
    #[inline]
    pub fn failure(error: String) -> Self {
        return Self {
            tx_hash: String::with_capacity(66), // 0x + 64 hex chars
            gas_used: 0,
            execution_time_ns: 0,
            success: false,
            error: Some(error),
        };
    }
}

/// Execute MEV opportunity
///
/// # Errors
///
/// Returns an error if execution fails
#[inline]
pub fn execute_opportunity(opportunity: &Opportunity) -> Result<ExecutionResult> {
    let start_time = get_timestamp_ns();
    
    // Use atomic executor for lock-free execution
    let executor = AtomicExecutor::new();
    let result = executor.execute(opportunity)?;
    
    let execution_time_ns = get_timestamp_ns().saturating_sub(start_time);
    
    return Ok(ExecutionResult::success(
        result.tx_hash,
        result.gas_used,
        execution_time_ns,
    ));
}

/// Initialize execution subsystem
///
/// # Errors
///
/// Returns an error if initialization fails
#[inline]
pub fn initialize() -> Result<()> {
    atomic_executor::initialize()?;
    gas_optimizer::initialize()?;
    bundle_builder::initialize()?;
    return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Opportunity;

    #[test]
    fn test_execution_result_success() {
        let result = ExecutionResult::success("0x123".into(), 21000, 1000);
        assert!(result.success);
        assert_eq!(result.gas_used, 21000);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_execution_result_failure() {
        let result = ExecutionResult::failure("Gas limit exceeded".into());
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_execute_opportunity() {
        let opportunity = Opportunity::new(1, "arbitrage".into(), 1000, 95);
        let result = execute_opportunity(&opportunity).unwrap();
        assert!(result.success);
    }
}
