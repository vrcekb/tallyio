//! `TallyIO` - A Rust library for handling calculations with proper error handling

#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![warn(missing_docs)]

use thiserror::Error;

/// Custom error type for the `TallyIO` library
#[derive(Error, Debug)]
pub enum TallyError {
    /// I/O related errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse errors
    #[error("Parse error: {0}")]
    Parse(String),

    /// Validation errors
    #[error("Validation error: {0}")]
    Validation(String),
}

/// A result type that uses our custom error type
pub type Result<T> = std::result::Result<T, TallyError>;

/// Adds two numbers together
///
/// # Examples
/// ```
/// use tallyio::sestej;
///
/// assert_eq!(sestej(2, 2), 4);
/// ```
#[must_use]
pub const fn sestej(a: i32, b: i32) -> i32 {
    a + b
}

/// Reads a file and returns its contents as a string
///
/// # Errors
/// Returns `TallyError::Io` if the file cannot be read
pub fn preberi_datoteko(ime_datoteke: &str) -> Result<String> {
    std::fs::read_to_string(ime_datoteke).map_err(Into::into)
}

/// Validates that a number is positive
///
/// # Errors
/// Returns `TallyError::Validation` if the number is not positive
pub fn preveri_pozitivno(stevilo: i32) -> Result<()> {
    if stevilo > 0 {
        Ok(())
    } else {
        Err(TallyError::Validation("Število mora biti pozitivno".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_sestej() {
        assert_eq!(sestej(2, 3), 5);
    }

    #[test]
    fn test_preveri_pozitivno() {
        assert!(preveri_pozitivno(1).is_ok());
        assert!(preveri_pozitivno(0).is_err());
        assert!(preveri_pozitivno(-1).is_err());
    }

    #[test]
    fn test_preberi_datoteko() -> anyhow::Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        write!(temp_file, "Testna vsebina")?;

        let vsebina = preberi_datoteko(temp_file.path().to_str().unwrap())?;
        assert_eq!(vsebina, "Testna vsebina");

        Ok(())
    }

    #[test]
    fn test_preberi_neobstoječo_datoteko() {
        let rezultat = preberi_datoteko("neobstoječa_datoteka.txt");
        assert!(matches!(rezultat, Err(TallyError::Io(_))));
    }
}
