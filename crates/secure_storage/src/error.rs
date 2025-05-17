//! Error tipi za secure storage modul

use std::fmt;
use std::io;

#[derive(Debug)]
pub enum StorageError {
    /// I/O napaka
    Io(io::Error),

    /// Napaka pri šifriranju/dešifriranju
    Encryption,

    /// Podatki niso najdeni
    NotFound,

    /// Neveljaven format podatkov
    InvalidFormat,

    /// Napaka pri serializaciji
    Serialization(serde_json::Error),

    /// Podatki že obstajajo
    AlreadyExists,

    /// Neveljavno ime
    InvalidName,

    /// Napaka pri serializaciji (bincode)
    SerializationError,

    /// Napaka pri deserializaciji (bincode)
    DeserializationError,

    /// Napaka pri šifriranju
    EncryptionError,

    /// Napaka pri dešifriranju
    DecryptionError,
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "IO error: {err}"),
            Self::Encryption | Self::EncryptionError => write!(f, "Encryption error"),
            Self::NotFound => write!(f, "Data not found"),
            Self::InvalidFormat => write!(f, "Invalid data format"),
            Self::Serialization(err) => write!(f, "Serialization error: {err}"),
            Self::AlreadyExists => write!(f, "Data already exists"),
            Self::InvalidName => write!(f, "Invalid name"),
            Self::SerializationError => write!(f, "Serialization error (bincode)"),
            Self::DeserializationError => write!(f, "Deserialization error (bincode)"),
            Self::DecryptionError => write!(f, "Decryption error"),
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Serialization(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<ring::error::Unspecified> for StorageError {
    fn from(_err: ring::error::Unspecified) -> Self {
        Self::EncryptionError
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_display() {
        // Preverimo vse variante Display implementacije
        assert_eq!(format!("{}", StorageError::NotFound), "Data not found");
        assert_eq!(format!("{}", StorageError::Encryption), "Encryption error");
        assert_eq!(format!("{}", StorageError::EncryptionError), "Encryption error");
        assert_eq!(format!("{}", StorageError::InvalidFormat), "Invalid data format");
        assert_eq!(format!("{}", StorageError::AlreadyExists), "Data already exists");
        assert_eq!(format!("{}", StorageError::InvalidName), "Invalid name");
        assert_eq!(
            format!("{}", StorageError::SerializationError),
            "Serialization error (bincode)"
        );
        assert_eq!(
            format!("{}", StorageError::DeserializationError),
            "Deserialization error (bincode)"
        );
        assert_eq!(format!("{}", StorageError::DecryptionError), "Decryption error");

        // Preverimo prikaz gnezdenih napak
        let io_err = io::Error::new(io::ErrorKind::NotFound, "test IO error");
        assert_eq!(format!("{}", StorageError::Io(io_err)), "IO error: test IO error");

        let json_err =
            serde_json::Error::io(io::Error::new(io::ErrorKind::InvalidData, "test JSON error"));
        assert!(format!("{}", StorageError::Serialization(json_err))
            .starts_with("Serialization error:"));
    }

    #[test]
    fn test_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "test error");
        let storage_err = StorageError::from(io_err);
        assert!(matches!(storage_err, StorageError::Io(_)));

        // Test za pretvorbo ring::error::Unspecified v StorageError
        // Ker ring::error::Unspecified nima javnega konstruktorja,
        // bomo testirali pretvorbo posredno preko funkcije, ki vrača ta tip napake
        let ring_err_result: Result<(), ring::error::Unspecified> = Err(ring::error::Unspecified);
        if let Err(ring_err) = ring_err_result {
            let storage_err = StorageError::from(ring_err);
            assert!(matches!(storage_err, StorageError::EncryptionError));
        }
    }

    #[test]
    fn test_error_source() {
        // Preverimo source za IO napako
        let original_error = io::Error::new(io::ErrorKind::NotFound, "source error");
        let error = StorageError::Io(original_error);
        assert!(error.source().is_some());

        // Preverimo source za serializacijsko napako
        let json_err =
            serde_json::Error::io(io::Error::new(io::ErrorKind::InvalidData, "json error"));
        let error = StorageError::Serialization(json_err);
        assert!(error.source().is_some());

        // Preverimo source za ostale napake, ki nimajo source
        assert!(StorageError::NotFound.source().is_none());
        assert!(StorageError::Encryption.source().is_none());
        assert!(StorageError::InvalidFormat.source().is_none());
        assert!(StorageError::AlreadyExists.source().is_none());
    }

    #[test]
    fn test_error_debug() {
        // Preverimo Debug implementacijo
        let error = StorageError::NotFound;
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("NotFound"));
    }
}
