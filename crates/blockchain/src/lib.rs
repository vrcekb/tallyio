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
pub struct BlockchainManager {
    block_count: std::sync::atomic::AtomicU64,
}

impl BlockchainManager {
    /// Create new blockchain manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> BlockchainResult<Self> {
        Ok(Self {
            block_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Process a blockchain block
    ///
    /// # Errors
    /// Returns error if block processing fails
    #[allow(clippy::unnecessary_wraps)] // API consistency with other crates
    pub fn process_block(&self, block: &str) -> BlockchainResult<String> {
        self.block_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("Processed block: {block}"))
    }
}

impl Default for BlockchainManager {
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
    fn test_blockchain_manager_creation() -> BlockchainResult<()> {
        let manager = BlockchainManager::new()?;
        assert_eq!(
            manager
                .block_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn test_blockchain_manager_default() {
        let manager = BlockchainManager::default();
        assert_eq!(
            manager
                .block_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_process_block() -> BlockchainResult<()> {
        let manager = BlockchainManager::new()?;
        let result = manager.process_block("test_block")?;

        // Verify block was processed
        assert_eq!(result, "Processed block: test_block");
        assert_eq!(
            manager
                .block_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        Ok(())
    }

    #[test]
    fn test_blockchain_latency_requirement() -> BlockchainResult<()> {
        let manager = BlockchainManager::new()?;
        let start = Instant::now();

        manager.process_block("latency_test")?;

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1,
            "Block processing took {}ms, must be <1ms",
            duration.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_blocks() -> BlockchainResult<()> {
        let manager = BlockchainManager::new()?;

        for i in 0_i32..10_i32 {
            manager.process_block(&format!("block_{i}"))?;
        }

        assert_eq!(
            manager
                .block_count
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
        Ok(())
    }

    #[test]
    fn test_concurrent_block_processing() -> BlockchainResult<()> {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(BlockchainManager::new()?);
        let mut handles = vec![];

        for i in 0_i32..5_i32 {
            let manager_clone = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                manager_clone.process_block(&format!("concurrent_block_{i}"))
            });
            handles.push(handle);
        }

        for handle in handles {
            match handle.join() {
                Ok(result) => {
                    result?; // Process the result but ignore the return value
                }
                Err(_) => return Err(BlockchainError::Network("Thread join failed".to_string())),
            }
        }

        assert_eq!(
            manager
                .block_count
                .load(std::sync::atomic::Ordering::Relaxed),
            5
        );
        Ok(())
    }
}
