//! `TallyIO` Database - Database abstractions

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("SQL error: {0}")]
    Sql(#[from] sqlx::Error),

    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    #[error("Connection error: {0}")]
    Connection(String),
}

pub type DatabaseResult<T> = Result<T, DatabaseError>;

/// Placeholder for database functionality
pub struct DatabaseManager;

impl DatabaseManager {
    /// Create new database manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    pub const fn new() -> DatabaseResult<Self> {
        Ok(Self)
    }
}

impl Default for DatabaseManager {
    fn default() -> Self {
        // This expect is acceptable in Default implementation
        #[allow(clippy::expect_used)]
        Self::new().expect("Failed to create DatabaseManager")
    }
}
