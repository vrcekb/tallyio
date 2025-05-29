//! `TallyIO` API - REST and WebSocket API

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("HTTP error: {0}")]
    Http(#[from] hyper::Error),

    #[error("WebSocket error: {0}")]
    WebSocket(Box<tungstenite::Error>),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl From<tungstenite::Error> for ApiError {
    fn from(err: tungstenite::Error) -> Self {
        Self::WebSocket(Box::new(err))
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

/// Placeholder for API functionality
pub struct ApiManager;

impl ApiManager {
    /// Create new API manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> ApiResult<Self> {
        Ok(Self)
    }
}

impl Default for ApiManager {
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
