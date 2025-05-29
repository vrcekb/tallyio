//! `TallyIO` Web UI - Web UI backend

use thiserror::Error;

#[derive(Error, Debug)]
pub enum WebUiError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("API error: {0}")]
    Api(#[from] tallyio_api::ApiError),

    #[error("UI error: {0}")]
    Ui(String),
}

pub type WebUiResult<T> = Result<T, WebUiError>;

/// Placeholder for web UI functionality
pub struct WebUiManager;

impl WebUiManager {
    /// Create new web UI manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> WebUiResult<Self> {
        Ok(Self)
    }
}

impl Default for WebUiManager {
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
