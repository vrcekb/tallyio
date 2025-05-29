//! `TallyIO` Blockchain - Multi-chain integration module

use thiserror::Error;

#[derive(Error, Debug)]
pub enum BlockchainError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Chain error: {0}")]
    Chain(String),
}

pub type BlockchainResult<T> = Result<T, BlockchainError>;

/// Placeholder for blockchain functionality
pub struct BlockchainManager;

impl BlockchainManager {
    /// Create new blockchain manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    pub const fn new() -> BlockchainResult<Self> {
        Ok(Self)
    }
}

impl Default for BlockchainManager {
    fn default() -> Self {
        // This expect is acceptable in Default implementation
        #[allow(clippy::expect_used)]
        Self::new().expect("Failed to create BlockchainManager")
    }
}
