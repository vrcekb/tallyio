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
pub struct ContractsManager;

impl ContractsManager {
    /// Create new contracts manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> ContractsResult<Self> {
        Ok(Self)
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
