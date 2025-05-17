//! Error tipi za core modul

use std::error::Error;
use std::fmt;

/// Core modul error tipi
#[derive(Debug)]
pub enum CoreError {
    /// Napaka pri alokaciji
    AllocationError(String),

    /// Napaka pri sinhronizaciji
    SyncError(String),

    /// Napaka pri validaciji
    ValidationError(String),

    /// Neznana napaka
    Unknown(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocationError(msg) => write!(f, "Allocation error: {msg}"),
            Self::SyncError(msg) => write!(f, "Synchronization error: {msg}"),
            Self::ValidationError(msg) => write!(f, "Validation error: {msg}"),
            Self::Unknown(msg) => write!(f, "Unknown error: {msg}"),
        }
    }
}

impl Error for CoreError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    #[test]
    fn test_error_display() {
        // Preveri vse vrste napak
        let err1 = CoreError::AllocationError("Out of memory".to_string());
        assert_eq!(err1.to_string(), "Allocation error: Out of memory");

        let err2 = CoreError::SyncError("Lock failed".to_string());
        assert_eq!(err2.to_string(), "Synchronization error: Lock failed");

        let err3 = CoreError::ValidationError("Invalid input".to_string());
        assert_eq!(err3.to_string(), "Validation error: Invalid input");

        let err4 = CoreError::Unknown("Something went wrong".to_string());
        assert_eq!(err4.to_string(), "Unknown error: Something went wrong");
    }

    #[test]
    fn test_error_conversion() {
        fn may_fail() -> Result<(), CoreError> {
            Err(CoreError::ValidationError("Invalid input".to_string()))
        }

        let err = may_fail().unwrap_err();
        assert!(matches!(err, CoreError::ValidationError(_)));
    }

    #[test]
    fn test_error_debug() {
        // Preveri Debug implementacijo
        let err = CoreError::AllocationError("test".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("AllocationError"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_error_trait() {
        // Preveri Error trait implementacijo
        let err = CoreError::AllocationError("test error".to_string());
        let dyn_err: &dyn StdError = &err;
        assert!(dyn_err.source().is_none()); // CoreError nima source
    }
}
