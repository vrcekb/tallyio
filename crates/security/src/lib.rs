//! `TallyIO` Security - Security modules

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Encryption error: {0}")]
    Encryption(String),
}

pub type SecurityResult<T> = Result<T, SecurityError>;

/// Placeholder for security functionality
pub struct SecurityManager {
    validation_count: std::sync::atomic::AtomicU64,
}

impl SecurityManager {
    /// Create new security manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> SecurityResult<Self> {
        Ok(Self {
            validation_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Validate a security request
    ///
    /// # Errors
    /// Returns error if request validation fails
    #[allow(clippy::unnecessary_wraps)] // API consistency with other crates
    pub fn validate_request(&self, request: &str) -> SecurityResult<String> {
        self.validation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("Validated request: {request}"))
    }
}

impl Default for SecurityManager {
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
    fn test_security_manager_creation() -> SecurityResult<()> {
        let manager = SecurityManager::new()?;
        assert_eq!(
            manager
                .validation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn test_security_manager_default() {
        let manager = SecurityManager::default();
        assert_eq!(
            manager
                .validation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_validate_request() -> SecurityResult<()> {
        let manager = SecurityManager::new()?;
        let result = manager.validate_request("test_data")?;

        // Verify operation was processed
        assert_eq!(result, "Validated request: test_data");
        assert_eq!(
            manager
                .validation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        Ok(())
    }

    #[test]
    fn test_security_latency_requirement() -> SecurityResult<()> {
        let manager = SecurityManager::new()?;
        let start = Instant::now();

        manager.validate_request("latency_test")?;

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1,
            "security operation took {}ms, must be <1ms",
            duration.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_operations() -> SecurityResult<()> {
        let manager = SecurityManager::new()?;

        for i in 0_i32..10_i32 {
            manager.validate_request(&format!("operation_{i}"))?;
        }

        assert_eq!(
            manager
                .validation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
        Ok(())
    }

    #[test]
    fn test_concurrent_operations() -> SecurityResult<()> {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(SecurityManager::new()?);
        let mut handles = vec![];

        for i in 0_i32..5_i32 {
            let manager_clone = Arc::clone(&manager);
            let handle =
                thread::spawn(move || manager_clone.validate_request(&format!("concurrent_{i}")));
            handles.push(handle);
        }

        for handle in handles {
            match handle.join() {
                Ok(result) => {
                    result?; // Process the result but ignore the return value
                }
                Err(_) => return Err(SecurityError::Auth("Thread join failed".to_string())),
            }
        }

        assert_eq!(
            manager
                .validation_count
                .load(std::sync::atomic::Ordering::Relaxed),
            5
        );
        Ok(())
    }

    #[test]
    fn test_security_error_display() {
        // Test SecurityError Display implementation (line 57)
        let core_error =
            tallyio_core::CoreError::Critical(tallyio_core::CriticalError::Invalid(404));
        let error = SecurityError::Core(core_error);
        let display_str = format!("{error}");
        assert!(display_str.contains("Core error"));
    }
}
