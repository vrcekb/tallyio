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
    pub const fn new() -> SecurityResult<Self> {
        Ok(Self)
    }
}

impl Default for SecurityManager {
    fn default() -> Self {
        #[allow(clippy::expect_used)] Self::new().expect("Failed to create SecurityManager")
    }
}
