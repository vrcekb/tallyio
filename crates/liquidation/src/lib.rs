//! `TallyIO` Liquidation - Liquidation strategies module

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LiquidationError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("Strategy error: {0}")]
    Strategy(String),
}

pub type LiquidationResult<T> = Result<T, LiquidationError>;

/// Placeholder for liquidation functionality
pub struct LiquidationManager {
    liquidation_count: std::sync::atomic::AtomicU64,
}

impl LiquidationManager {
    /// Create new liquidation manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> LiquidationResult<Self> {
        Ok(Self {
            liquidation_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Process a liquidation
    ///
    /// # Errors
    /// Returns error if liquidation processing fails
    #[allow(clippy::unnecessary_wraps)] // API consistency with other crates
    pub fn process_liquidation(&self, liquidation: &str) -> LiquidationResult<String> {
        self.liquidation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("Processed liquidation: {liquidation}"))
    }
}

impl Default for LiquidationManager {
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
    fn test_liquidation_manager_creation() -> LiquidationResult<()> {
        let manager = LiquidationManager::new()?;
        assert_eq!(
            manager
                .liquidation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn test_liquidation_manager_default() {
        let manager = LiquidationManager::default();
        assert_eq!(
            manager
                .liquidation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_process_liquidation() -> LiquidationResult<()> {
        let manager = LiquidationManager::new()?;
        let result = manager.process_liquidation("test_data")?;

        // Verify operation was processed
        assert_eq!(result, "Processed liquidation: test_data");
        assert_eq!(
            manager
                .liquidation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        Ok(())
    }

    #[test]
    fn test_liquidation_latency_requirement() -> LiquidationResult<()> {
        let manager = LiquidationManager::new()?;
        let start = Instant::now();

        manager.process_liquidation("latency_test")?;

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1,
            "liquidation operation took {}ms, must be <1ms",
            duration.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_operations() -> LiquidationResult<()> {
        let manager = LiquidationManager::new()?;

        for i in 0_i32..10_i32 {
            manager.process_liquidation(&format!("operation_{i}"))?;
        }

        assert_eq!(
            manager
                .liquidation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
        Ok(())
    }

    #[test]
    fn test_concurrent_operations() -> LiquidationResult<()> {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(LiquidationManager::new()?);
        let mut handles = vec![];

        for i in 0_i32..5_i32 {
            let manager_clone = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                manager_clone.process_liquidation(&format!("concurrent_{i}"))
            });
            handles.push(handle);
        }

        for handle in handles {
            match handle.join() {
                Ok(result) => {
                    result?; // Process the result but ignore the return value
                }
                Err(_) => return Err(LiquidationError::Strategy("Thread join failed".to_string())),
            }
        }

        assert_eq!(
            manager
                .liquidation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            5
        );
        Ok(())
    }

    #[test]
    fn test_liquidation_error_display() {
        // Test LiquidationError Display implementation (line 54)
        let core_error =
            tallyio_core::CoreError::Critical(tallyio_core::CriticalError::Invalid(202));
        let error = LiquidationError::Core(core_error);
        let display_str = format!("{error}");
        assert!(display_str.contains("Core error"));
    }
}
