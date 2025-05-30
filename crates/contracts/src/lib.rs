//! `TallyIO` Contracts - Smart contracts integration

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ContractsError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("Contract error: {0}")]
    Contract(String),

    #[error("ABI error: {0}")]
    Abi(String),
}

pub type ContractsResult<T> = Result<T, ContractsError>;

/// Placeholder for contracts functionality
pub struct ContractsManager {
    contract_count: std::sync::atomic::AtomicU64,
}

impl ContractsManager {
    /// Create new contracts manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> ContractsResult<Self> {
        Ok(Self {
            contract_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Deploy a smart contract
    ///
    /// # Errors
    /// Returns error if contract deployment fails
    #[allow(clippy::unnecessary_wraps)] // API consistency with other crates
    pub fn deploy_contract(&self, contract: &str) -> ContractsResult<String> {
        self.contract_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("Deployed contract: {contract}"))
    }
}

impl Default for ContractsManager {
    fn default() -> Self {
        // Use match instead of expect to comply with zero-panic policy
        #[allow(clippy::option_if_let_else)] // Result, not Option
        match Self::new() {
            Ok(manager) => manager,
            Err(_) => {
                // This should never happen in normal circumstances
                // If it does, it's a programming error
                std::process::abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_contracts_manager_creation() -> ContractsResult<()> {
        let manager = ContractsManager::new()?;
        assert_eq!(
            manager
                .contract_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn test_contracts_manager_default() {
        let manager = ContractsManager::default();
        assert_eq!(
            manager
                .contract_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_deploy_contract() -> ContractsResult<()> {
        let manager = ContractsManager::new()?;
        let result = manager.deploy_contract("test_contract")?;

        // Verify contract was deployed
        assert_eq!(result, "Deployed contract: test_contract");
        assert_eq!(
            manager
                .contract_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        Ok(())
    }

    #[test]
    fn test_contracts_latency_requirement() -> ContractsResult<()> {
        let manager = ContractsManager::new()?;
        let start = Instant::now();

        manager.deploy_contract("latency_test")?;

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1,
            "Contract deployment took {}ms, must be <1ms",
            duration.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_contracts() -> ContractsResult<()> {
        let manager = ContractsManager::new()?;

        for i in 0_i32..10_i32 {
            manager.deploy_contract(&format!("contract_{i}"))?;
        }

        assert_eq!(
            manager
                .contract_count
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
        Ok(())
    }

    #[test]
    fn test_contracts_error_display() {
        // Test ContractsError Display implementation (line 57)
        let core_error =
            tallyio_core::CoreError::Critical(tallyio_core::CriticalError::Invalid(789));
        let error = ContractsError::Core(core_error);
        let display_str = format!("{error}");
        assert!(display_str.contains("Core error"));
    }
}
