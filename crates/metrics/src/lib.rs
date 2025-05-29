//! `TallyIO` Metrics - Metrics and monitoring

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),

    #[error("Collection error: {0}")]
    Collection(String),
}

pub type MetricsResult<T> = Result<T, MetricsError>;

/// Placeholder for metrics functionality
pub struct MetricsManager;

impl MetricsManager {
    /// Create new metrics manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> MetricsResult<Self> {
        Ok(Self)
    }
}

impl Default for MetricsManager {
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
