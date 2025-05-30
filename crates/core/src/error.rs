//! Error types for TallyIO Core
//!
//! This module defines all error types used throughout the core crate.
//! All errors are designed to be zero-cost and provide detailed context.

use thiserror::Error;

/// Critical errors that require immediate attention
///
/// These errors represent conditions that could affect system stability
/// or performance guarantees. They use Copy semantics for zero-cost propagation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CriticalError {
    /// Invalid input parameter with error code
    Invalid(u16),
    /// Out of memory condition with error code
    OutOfMemory(u16),
    /// Latency requirement violated with microseconds exceeded
    LatencyViolation(u64),
    /// Queue overflow with current size
    QueueOverflow(u32),
    /// Worker thread failure with thread ID
    WorkerFailure(u32),
    /// State corruption detected with state ID
    StateCorruption(u64),
}

impl std::fmt::Display for CriticalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(code) => write!(f, "Invalid parameter (code: {code})"),
            Self::OutOfMemory(code) => write!(f, "Out of memory (code: {code})"),
            Self::LatencyViolation(micros) => write!(f, "Latency violation: {micros}μs"),
            Self::QueueOverflow(size) => write!(f, "Queue overflow at size: {size}"),
            Self::WorkerFailure(id) => write!(f, "Worker thread {id} failed"),
            Self::StateCorruption(id) => write!(f, "State corruption detected (ID: {id})"),
        }
    }
}

impl std::error::Error for CriticalError {}

/// Main error type for the core crate
///
/// This error type provides comprehensive error handling while maintaining
/// performance characteristics required for high-frequency operations.
#[derive(Error, Debug)]
pub enum CoreError {
    /// Critical system error
    #[error("Critical: {0}")]
    Critical(#[from] CriticalError),

    /// Configuration error
    #[error("Configuration: {0}")]
    Config(String),

    /// Engine operation error
    #[error("Engine: {0}")]
    Engine(String),

    /// State management error
    #[error("State: {0}")]
    State(String),

    /// Mempool operation error
    #[error("Mempool: {0}")]
    Mempool(String),

    /// Optimization error
    #[error("Optimization: {0}")]
    Optimization(String),

    /// Worker thread error
    #[error("Worker: {0}")]
    Worker(String),

    /// Scheduler error
    #[error("Scheduler: {0}")]
    Scheduler(String),

    /// Transaction processing error
    #[error("Transaction: {0}")]
    Transaction(String),

    /// MEV opportunity error
    #[error("MEV: {0}")]
    Mev(String),

    /// IO error
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization: {0}")]
    Serialization(#[from] serde_json::Error),

    /// UUID error
    #[error("UUID: {0}")]
    Uuid(#[from] uuid::Error),
}

impl CoreError {
    /// Create a configuration error
    #[must_use]
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create an engine error
    #[must_use]
    pub fn engine(msg: impl Into<String>) -> Self {
        Self::Engine(msg.into())
    }

    /// Create a state error
    #[must_use]
    pub fn state(msg: impl Into<String>) -> Self {
        Self::State(msg.into())
    }

    /// Create a mempool error
    #[must_use]
    pub fn mempool(msg: impl Into<String>) -> Self {
        Self::Mempool(msg.into())
    }

    /// Create an optimization error
    #[must_use]
    pub fn optimization(msg: impl Into<String>) -> Self {
        Self::Optimization(msg.into())
    }

    /// Create a worker error
    #[must_use]
    pub fn worker(msg: impl Into<String>) -> Self {
        Self::Worker(msg.into())
    }

    /// Create a scheduler error
    #[must_use]
    pub fn scheduler(msg: impl Into<String>) -> Self {
        Self::Scheduler(msg.into())
    }

    /// Create a transaction error
    #[must_use]
    pub fn transaction(msg: impl Into<String>) -> Self {
        Self::Transaction(msg.into())
    }

    /// Create a MEV error
    #[must_use]
    pub fn mev(msg: impl Into<String>) -> Self {
        Self::Mev(msg.into())
    }

    /// Check if this is a critical error
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        matches!(self, Self::Critical(_))
    }

    /// Get the error code if this is a critical error
    #[must_use]
    pub const fn critical_code(&self) -> Option<u16> {
        match self {
            Self::Critical(CriticalError::Invalid(code) | CriticalError::OutOfMemory(code)) => {
                Some(*code)
            }
            _ => None,
        }
    }

    /// Check if this error should trigger an immediate shutdown
    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::Critical(CriticalError::StateCorruption(_) | CriticalError::OutOfMemory(_))
        )
    }
}

/// Result type alias for core operations
pub type CoreResult<T> = Result<T, CoreError>;

/// Result type alias for critical operations
pub type CriticalResult<T> = Result<T, CriticalError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_error_display() {
        let err = CriticalError::Invalid(42);
        assert_eq!(err.to_string(), "Invalid parameter (code: 42)");

        let err = CriticalError::LatencyViolation(1500);
        assert_eq!(err.to_string(), "Latency violation: 1500μs");

        let err = CriticalError::WorkerFailure(3);
        assert_eq!(err.to_string(), "Worker thread 3 failed");
    }

    #[test]
    fn test_core_error_creation() {
        let err = CoreError::config("test config error");
        assert!(matches!(err, CoreError::Config(_)));

        let err = CoreError::engine("test engine error");
        assert!(matches!(err, CoreError::Engine(_)));

        let err = CoreError::worker("test worker error");
        assert!(matches!(err, CoreError::Worker(_)));
    }

    #[test]
    fn test_critical_error_detection() {
        let err = CoreError::Critical(CriticalError::Invalid(1));
        assert!(err.is_critical());
        assert_eq!(err.critical_code(), Some(1));

        let err = CoreError::Config("test".to_string());
        assert!(!err.is_critical());
        assert_eq!(err.critical_code(), None);
    }

    #[test]
    fn test_fatal_error_detection() {
        let err = CoreError::Critical(CriticalError::StateCorruption(123));
        assert!(err.is_fatal());

        let err = CoreError::Critical(CriticalError::OutOfMemory(456));
        assert!(err.is_fatal());

        let err = CoreError::Critical(CriticalError::Invalid(789));
        assert!(!err.is_fatal());
    }

    #[test]
    fn test_error_conversion() {
        let critical = CriticalError::OutOfMemory(100);
        let core_err: CoreError = critical.into();
        assert!(core_err.is_critical());
        assert_eq!(core_err.critical_code(), Some(100));
        assert!(core_err.is_fatal());
    }

    #[test]
    fn test_critical_error_display_comprehensive() {
        // Test all CriticalError variants for display (lines 32-36)
        let err = CriticalError::OutOfMemory(200);
        assert_eq!(err.to_string(), "Out of memory (code: 200)");

        let err = CriticalError::QueueOverflow(1000);
        assert_eq!(err.to_string(), "Queue overflow at size: 1000");

        let err = CriticalError::StateCorruption(999);
        assert_eq!(err.to_string(), "State corruption detected (ID: 999)");
    }

    #[test]
    fn test_core_error_creation_comprehensive() {
        // Test all CoreError creation methods (lines 117-154)
        let err = CoreError::state("test state error");
        assert!(matches!(err, CoreError::State(_)));

        let err = CoreError::mempool("test mempool error");
        assert!(matches!(err, CoreError::Mempool(_)));

        let err = CoreError::optimization("test optimization error");
        assert!(matches!(err, CoreError::Optimization(_)));

        let err = CoreError::scheduler("test scheduler error");
        assert!(matches!(err, CoreError::Scheduler(_)));

        let err = CoreError::transaction("test transaction error");
        assert!(matches!(err, CoreError::Transaction(_)));

        let err = CoreError::mev("test mev error");
        assert!(matches!(err, CoreError::Mev(_)));
    }

    #[test]
    fn test_error_from_conversions() {
        // Test From implementations (lines 123-124, 141-142, 147-148, 153-154)
        use std::io;

        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let core_err: CoreError = io_err.into();
        assert!(matches!(core_err, CoreError::Io(_)));

        // Test JSON error conversion
        let json_result = serde_json::from_str::<serde_json::Value>("invalid json");
        assert!(json_result.is_err());
        if let Err(json_err) = json_result {
            let core_err: CoreError = json_err.into();
            assert!(matches!(core_err, CoreError::Serialization(_)));
        }

        // Test UUID error conversion
        let uuid_result = uuid::Uuid::parse_str("invalid-uuid");
        assert!(uuid_result.is_err());
        if let Err(uuid_err) = uuid_result {
            let core_err: CoreError = uuid_err.into();
            assert!(matches!(core_err, CoreError::Uuid(_)));
        }
    }
}
