//! Secure Storage modul za varno shranjevanje občutljivih podatkov
//!
//! Ta modul omogoča varno šifriranje in dešifriranje podatkov ter njihovo
//! shranjevanje na disk z uporabo močne kriptografije.

mod error;
mod key;

pub use error::StorageError;
pub use key::Key;

use chacha20poly1305::{AeadInPlace, ChaCha20Poly1305, KeyInit};
use rand::{rngs::OsRng, RngCore};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Secure Storage za varno shranjevanje podatkov
pub struct SecureStorage {
    /// Pot do datoteke za shranjevanje
    storage_path: PathBuf,
    /// Šifrirni ključ
    #[allow(dead_code)]
    key: Key,
    /// `ChaCha20Poly1305` cipher
    cipher: ChaCha20Poly1305,
}

impl SecureStorage {
    /// Ustvari novo instanco `SecureStorage`
    ///
    /// # Errors
    /// Vrne napako, če inicializacija ne uspe
    pub fn new<P: AsRef<Path>>(path: P, key: Key) -> Result<Self, StorageError> {
        let cipher = ChaCha20Poly1305::new_from_slice(key.as_bytes())
            .map_err(|_| StorageError::EncryptionError)?;

        Ok(Self { storage_path: path.as_ref().to_path_buf(), key, cipher })
    }

    /// Šifrira podatke z uporabo `ChaCha20Poly1305`
    ///
    /// # Errors
    /// Vrne napako, če šifriranje ne uspe
    pub fn encrypt(&self, data: &[u8]) -> Result<(Vec<u8>, [u8; 12]), StorageError> {
        let mut nonce = [0u8; 12];
        OsRng.fill_bytes(&mut nonce);

        let mut buffer = data.to_vec();
        self.cipher
            .encrypt_in_place(&nonce.into(), b"", &mut buffer)
            .map_err(|_| StorageError::EncryptionError)?;

        Ok((buffer, nonce))
    }

    /// Dešifrira podatke z uporabo `ChaCha20Poly1305`
    ///
    /// # Errors
    /// Vrne napako, če dešifriranje ne uspe
    pub fn decrypt(&self, encrypted_data: &[u8], nonce: [u8; 12]) -> Result<Vec<u8>, StorageError> {
        let mut buffer = encrypted_data.to_vec();
        self.cipher
            .decrypt_in_place(&nonce.into(), b"", &mut buffer)
            .map_err(|_| StorageError::DecryptionError)?;

        Ok(buffer)
    }

    /// Shrani podatke na disk
    ///
    /// # Errors
    /// Vrne napako, če shranjevanje ne uspe
    pub async fn store<T: Serialize + Sync>(&self, key: &str, value: &T) -> Result<(), StorageError> {
        // Serializiraj podatke
        let serialized = bincode::serialize(value).map_err(|_| StorageError::SerializationError)?;

        // Šifriraj podatke
        let (encrypted, nonce) = self.encrypt(&serialized)?;

        // Ustvari direktorij, če ne obstaja
        if let Some(parent) = self.storage_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Ustvari strukturo za shranjevanje
        let storage_data = StorageData { key: key.to_string(), data: encrypted, nonce };

        // Serializiraj in shrani
        let json = serde_json::to_string(&storage_data).map_err(StorageError::Serialization)?;

        let mut file = fs::File::create(&self.storage_path).await?;
        file.write_all(json.as_bytes()).await?;
        file.flush().await?;

        Ok(())
    }

    /// Naloži podatke z diska
    ///
    /// # Errors
    /// Vrne napako, če nalaganje ne uspe
    pub async fn load<T: DeserializeOwned + Sync>(&self, key: &str) -> Result<T, StorageError> {
        // Preveri, če datoteka obstaja
        if !self.storage_path.exists() {
            return Err(StorageError::NotFound);
        }

        // Preberi datoteko
        let content = fs::read_to_string(&self.storage_path).await?;

        // Deserializiraj vsebino
        let storage_data: StorageData =
            serde_json::from_str(&content).map_err(StorageError::Serialization)?;

        // Preveri, če je ključ pravilen
        if storage_data.key != key {
            return Err(StorageError::NotFound);
        }

        // Dešifriraj podatke
        let decrypted = self.decrypt(&storage_data.data, storage_data.nonce)?;

        // Deserializiraj podatke
        bincode::deserialize(&decrypted).map_err(|_| StorageError::DeserializationError)
    }
}

/// Struktura za shranjevanje podatkov
#[derive(Serialize, Deserialize)]
struct StorageData {
    /// Ključ za dostop do podatkov
    key: String,
    /// Šifrirani podatki
    data: Vec<u8>,
    /// Nonce za šifriranje
    nonce: [u8; 12],
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestData {
        field1: String,
        field2: u32,
    }

    #[tokio::test]
    async fn test_store_and_load() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.json");
        let key = Key::generate().unwrap();
        let storage = SecureStorage::new(&file, key).unwrap();

        let test_data = TestData { field1: "test".to_string(), field2: 42 };

        // Shrani podatke
        storage.store("test_key", &test_data).await.unwrap();

        // Naloži podatke
        let loaded: TestData = storage.load("test_key").await.unwrap();

        assert_eq!(test_data, loaded);
    }

    #[tokio::test]
    async fn test_not_found() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.json");
        let key = Key::generate().unwrap();
        let storage = SecureStorage::new(&file, key).unwrap();

        let result: Result<TestData, _> = storage.load("nonexistent").await;
        assert!(matches!(result, Err(StorageError::NotFound)));
    }

    #[tokio::test]
    async fn test_nested_directory_creation() {
        // Test za ustvarjanje direktorija, če ne obstaja (vrstica 84)
        let dir = tempdir().unwrap();
        let nested_path = dir.path().join("nested/deep/path/test.json");
        let key = Key::generate().unwrap();
        let storage = SecureStorage::new(&nested_path, key).unwrap();

        let test_data = TestData { field1: "test".to_string(), field2: 42 };

        // Shrani podatke - to bo ustvarilo direktorije
        storage.store("test_key", &test_data).await.unwrap();

        // Preveri, če je direktorij bil ustvarjen
        assert!(nested_path.parent().unwrap().exists());
    }

    #[tokio::test]
    async fn test_wrong_key() {
        // Test za preverjanje, če je ključ pravilen (vrstica 115)
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.json");
        let key = Key::generate().unwrap();
        let storage = SecureStorage::new(&file, key).unwrap();

        let test_data = TestData { field1: "test".to_string(), field2: 42 };

        // Shrani podatke z enim ključem
        storage.store("correct_key", &test_data).await.unwrap();

        // Poskusi naložiti z napačnim ključem
        let result: Result<TestData, _> = storage.load("wrong_key").await;
        assert!(matches!(result, Err(StorageError::NotFound)));
    }

    #[test]
    fn test_encrypt_decrypt() {
        let key = Key::generate().unwrap();
        let storage = SecureStorage::new(PathBuf::from("/dev/null"), key).unwrap();

        let data = b"test data";
        let (encrypted, nonce) = storage.encrypt(data).unwrap();
        let decrypted = storage.decrypt(&encrypted, nonce).unwrap();

        assert_eq!(data.to_vec(), decrypted);
    }

    #[tokio::test]
    async fn test_key_mismatch() {
        // Test za preverjanje, če je ključ pravilen (vrstica 115)
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.json");
        let key = Key::generate().unwrap();
        let storage = SecureStorage::new(&file, key).unwrap();

        // Shrani podatke z enim ključem
        let test_data = TestData { field1: "test".to_string(), field2: 42 };
        storage.store("correct_key", &test_data).await.unwrap();

        // Poskusi naložiti z napačnim ključem - to bo sprožilo NotFound napako
        let result: Result<TestData, _> = storage.load("wrong_key").await;
        assert!(matches!(result, Err(StorageError::NotFound)));
    }

    #[tokio::test]
    async fn test_file_operations() {
        // Test za pokritje ustvarjanja direktorija in flush operacije
        let dir = tempdir().unwrap();
        let nested_path = dir.path().join("deep/nested/path/test.json");
        let key = Key::generate().unwrap();
        let storage = SecureStorage::new(&nested_path, key).unwrap();

        // Shrani podatke - to bo ustvarilo direktorije in zapisalo datoteko
        let test_data = TestData { field1: "test".to_string(), field2: 42 };
        storage.store("test_key", &test_data).await.unwrap();

        // Preveri, če je direktorij bil ustvarjen
        assert!(nested_path.parent().unwrap().exists());

        // Preveri, če je datoteka bila ustvarjena
        assert!(nested_path.exists());
    }
}
