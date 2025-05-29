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
    pub const fn new() -> ContractsResult<Self> {
        Ok(Self)
    }
}

impl Default for ContractsManager {
    fn default() -> Self {
        // This expect is acceptable in Default implementation
        #[allow(clippy::expect_used)]
        Self::new().expect("Failed to create ContractsManager")
    }
}
