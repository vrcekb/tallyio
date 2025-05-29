//! Error types for `TallyIO` Core - Zero panic guarantee

use thiserror::Error;

/// Critical errors that require immediate attention
///
/// These are Copy errors designed for ultra-low latency hot paths.
/// All variants carry a u16 error code for monitoring and diagnostics.
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CriticalError {
    /// Invalid input parameter with error code
    #[error("Invalid input: code {0}")]
    Invalid(u16),
    /// Out of memory condition with error code
    #[error("Out of memory: code {0}")]
    OutOfMemory(u16),
    /// Timeout exceeded with error code
    #[error("Timeout: code {0}")]
    Timeout(u16),
    /// Resource exhausted with error code
    #[error("Resource exhausted: code {0}")]
    ResourceExhausted(u16),
}

impl CriticalError {
    /// Get error code for monitoring
    ///
    /// Returns the numeric error code associated with this error.
    /// Used for metrics collection and alerting systems.
    #[must_use]
    pub const fn code(self) -> u16 {
        match self {
            Self::Invalid(code)
            | Self::OutOfMemory(code)
            | Self::Timeout(code)
            | Self::ResourceExhausted(code) => code,
        }
    }

    /// Check if error is recoverable
    ///
    /// Returns `true` if the operation can be retried, `false` if it's a permanent failure.
    /// Invalid input errors are not recoverable, while resource errors typically are.
    #[must_use]
    pub const fn is_recoverable(self) -> bool {
        match self {
            Self::Invalid(_) => false,
            Self::OutOfMemory(_) | Self::Timeout(_) | Self::ResourceExhausted(_) => true,
        }
    }
}

/// Main error type for Core operations
///
/// This enum handles both critical (hot path) and non-critical errors.
/// Critical errors are Copy and designed for ultra-low latency paths.
/// Other errors provide rich context for debugging and diagnostics.
#[derive(Error, Debug)]
pub enum CoreError {
    /// Critical error from hot path operations
    #[error("Critical error: {0:?}")]
    Critical(#[from] CriticalError),

    /// IO operation failed
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization failed
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Data parsing failed
    #[error("Parse error: {message}")]
    Parse { message: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    Config { message: String },
}

impl CoreError {
    /// Create a parse error
    ///
    /// # Arguments
    /// * `message` - Description of what failed to parse
    ///
    /// # Returns
    /// A new `CoreError::Parse` variant
    #[must_use]
    pub fn parse<S: Into<String>>(message: S) -> Self {
        Self::Parse {
            message: message.into(),
        }
    }

    /// Create a config error
    ///
    /// # Arguments
    /// * `message` - Description of the configuration problem
    ///
    /// # Returns
    /// A new `CoreError::Config` variant
    #[must_use]
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Check if error is critical
    ///
    /// Returns `true` if this is a critical error from hot path operations.
    /// Critical errors require immediate attention and may indicate system instability.
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        matches!(self, Self::Critical(_))
    }
}

/// Result type alias for Core operations
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::missing_errors_doc)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_error_codes() {
        let err = CriticalError::Invalid(404);
        assert_eq!(err.code(), 404);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_core_error_creation() {
        let err = CoreError::parse("test message");
        assert!(!err.is_critical());

        let critical = CoreError::Critical(CriticalError::Invalid(500));
        assert!(critical.is_critical());
    }

    #[test]
    fn test_critical_error_variants() {
        let invalid = CriticalError::Invalid(100);
        let timeout = CriticalError::Timeout(200);
        let out_of_memory = CriticalError::OutOfMemory(300);

        assert_eq!(invalid.code(), 100);
        assert_eq!(timeout.code(), 200);
        assert_eq!(out_of_memory.code(), 300);

        assert!(!invalid.is_recoverable());
        assert!(timeout.is_recoverable());
        assert!(out_of_memory.is_recoverable());
    }

    #[test]
    fn test_core_error_config() {
        let config_err = CoreError::config("invalid setting");
        assert!(!config_err.is_critical());

        assert!(matches!(config_err, CoreError::Config { .. }));
        if let CoreError::Config { message } = config_err {
            assert_eq!(message, "invalid setting");
        }
    }

    #[test]
    fn test_core_error_parse() {
        let parse_err = CoreError::parse("failed to parse data");
        assert!(!parse_err.is_critical());

        assert!(matches!(parse_err, CoreError::Parse { .. }));
        if let CoreError::Parse { message } = parse_err {
            assert_eq!(message, "failed to parse data");
        }
    }

    #[test]
    fn test_error_from_conversions() {
        // Test conversion from CriticalError
        let critical = CriticalError::Invalid(404);
        let core_err: CoreError = critical.into();
        assert!(core_err.is_critical());

        // Test conversion from io::Error
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let core_err: CoreError = io_err.into();
        assert!(!core_err.is_critical());
    }

    #[test]
    fn test_error_display() {
        let critical = CriticalError::Invalid(404);
        let display = format!("{critical}");
        assert!(display.contains("404"));

        let core_err = CoreError::Critical(critical);
        let display = format!("{core_err}");
        assert!(display.contains("Critical error"));
    }

    #[test]
    fn test_critical_error_copy() {
        let err1 = CriticalError::Invalid(100);
        let err2 = err1; // Should copy, not move

        // Both should be usable
        assert_eq!(err1.code(), 100);
        assert_eq!(err2.code(), 100);
    }

    #[test]
    fn test_core_result_alias() {
        fn test_function() -> CoreResult<i32> {
            Ok(42)
        }

        let result = test_function();
        assert!(result.is_ok());
        if let Ok(value) = result {
            assert_eq!(value, 42_i32);
        }
    }
}
