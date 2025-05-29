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
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> DatabaseResult<Self> {
        Ok(Self)
    }
}

impl Default for DatabaseManager {
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
