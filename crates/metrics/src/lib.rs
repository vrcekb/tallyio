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
    pub const fn new() -> MetricsResult<Self> {
        Ok(Self)
    }
}

impl Default for MetricsManager {
    fn default() -> Self {
        // This expect is acceptable in Default implementation
        #[allow(clippy::expect_used)]
        Self::new().expect("Failed to create MetricsManager")
    }
}
