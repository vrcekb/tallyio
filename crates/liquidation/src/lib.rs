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
pub struct LiquidationManager;

impl LiquidationManager {
    /// Create new liquidation manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    pub const fn new() -> LiquidationResult<Self> {
        Ok(Self)
    }
}

impl Default for LiquidationManager {
    fn default() -> Self {
        // This expect is acceptable in Default implementation
        #[allow(clippy::expect_used)]
        Self::new().expect("Failed to create LiquidationManager")
    }
}
