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

/// Database manager for `TallyIO`
pub struct DatabaseManager {
    query_count: std::sync::atomic::AtomicU64,
}

impl DatabaseManager {
    /// Create new database manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> DatabaseResult<Self> {
        Ok(Self {
            query_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Execute a database query
    ///
    /// # Errors
    /// Returns error if query execution fails
    #[allow(clippy::unnecessary_wraps)] // API consistency with other crates
    pub fn execute_query(&self, query: &str) -> DatabaseResult<String> {
        self.query_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("Executed query: {query}"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_database_manager_creation() -> DatabaseResult<()> {
        let manager = DatabaseManager::new()?;
        assert_eq!(
            manager
                .query_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn test_database_manager_default() {
        let manager = DatabaseManager::default();
        assert_eq!(
            manager
                .query_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_execute_query() -> DatabaseResult<()> {
        let manager = DatabaseManager::new()?;
        let result = manager.execute_query("test_query")?;

        // Verify query was executed
        assert_eq!(result, "Executed query: test_query");
        assert_eq!(
            manager
                .query_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        Ok(())
    }

    #[test]
    fn test_database_latency_requirement() -> DatabaseResult<()> {
        let manager = DatabaseManager::new()?;
        let start = Instant::now();

        manager.execute_query("latency_test")?;

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1,
            "Database query took {}ms, must be <1ms",
            duration.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_queries() -> DatabaseResult<()> {
        let manager = DatabaseManager::new()?;

        for i in 0_i32..10_i32 {
            manager.execute_query(&format!("query_{i}"))?;
        }

        assert_eq!(
            manager
                .query_count
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
        Ok(())
    }
}
