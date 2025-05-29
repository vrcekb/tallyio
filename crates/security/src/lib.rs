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
pub struct SecurityManager;

impl SecurityManager {
    /// Create new security manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> SecurityResult<Self> {
        Ok(Self)
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
