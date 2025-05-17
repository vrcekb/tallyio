//! Osnovni unit testi za secure_storage modul
//!
//! Ta modul vsebuje osnovne teste za CRUD operacije in validacijo delovanja
//! SecureStorage implementacije. Testi so optimizirani za hitro izvajanje in
//! preverjanje pravilnosti delovanja.

use secure_storage::{SecureStorage, StorageError};
use std::path::Path;
use tempfile::tempdir;
use tokio::test;
use tokio::time::sleep;
use std::time::Duration;

/// Test osnovnega shranjevanja in nalaganja podatkov
#[test]
async fn test_store_and_load() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = SecureStorage::new(path).await.expect("Failed to create storage");
    
    // Shrani podatke
    let key = "test_key";
    let value = b"test_value".to_vec();
    storage.store(key, &value).await.expect("Failed to store data");
    
    // Naloži podatke
    let loaded = storage.load(key).await.expect("Failed to load data");
    assert_eq!(value, loaded, "Loaded data doesn't match stored data");
}

/// Test preverjanja veljavnosti ključev
#[test]
async fn test_key_validation() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = SecureStorage::new(path).await.expect("Failed to create storage");
    
    // Preveri neveljavne ključe
    let invalid_keys = vec!["", " leading_space", "trailing_space ", "invalid/char", 
                           "another\\invalid", "too_long_key_name_that_exceeds_the_maximum_allowed_length_for_keys_in_secure_storage"];
    
    for key in invalid_keys {
        let result = storage.store(key, b"test").await;
        assert!(result.is_err(), "Expected error for invalid key: {}", key);
        
        if let Err(e) = result {
            assert!(matches!(e, StorageError::InvalidName), "Wrong error type for key: {}", key);
        }
    }
    
    // Preveri veljavne ključe
    let valid_keys = vec!["valid_key", "valid-key", "valid.key", "ValidKey123", "_valid_key"];
    
    for key in valid_keys {
        let result = storage.store(key, b"test").await;
        assert!(result.is_ok(), "Expected success for valid key: {}", key);
    }
}

/// Test brisanja podatkov
#[test]
async fn test_delete() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = SecureStorage::new(path).await.expect("Failed to create storage");
    
    // Shrani podatke
    let key = "test_delete";
    let value = b"test_value".to_vec();
    storage.store(key, &value).await.expect("Failed to store data");
    
    // Preveri, da obstajajo
    let loaded = storage.load(key).await.expect("Failed to load data");
    assert_eq!(value, loaded);
    
    // Izbriši podatke
    storage.delete(key).await.expect("Failed to delete data");
    
    // Preveri, da ne obstajajo več
    let result = storage.load(key).await;
    assert!(result.is_err(), "Expected error when loading deleted key");
    assert!(matches!(result, Err(StorageError::NotFound)), "Wrong error type when loading deleted key");
}

/// Test shranjevanja in nalaganja več ključev
#[test]
async fn test_multiple_keys() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = SecureStorage::new(path).await.expect("Failed to create storage");
    
    // Shrani več ključev
    let keys = vec!["key1", "key2", "key3", "key4", "key5"];
    let values = vec![b"value1".to_vec(), b"value2".to_vec(), b"value3".to_vec(), 
                      b"value4".to_vec(), b"value5".to_vec()];
    
    for (i, key) in keys.iter().enumerate() {
        storage.store(key, &values[i]).await.expect("Failed to store data");
    }
    
    // Naloži in preveri vse ključe
    for (i, key) in keys.iter().enumerate() {
        let loaded = storage.load(key).await.expect("Failed to load data");
        assert_eq!(values[i], loaded, "Loaded data doesn't match for key: {}", key);
    }
}

/// Test odpornosti na napake pri inicializaciji
#[test]
async fn test_init_robustness() {
    // Poskusi inicializirati storage z neobstoječo mapo
    let result = SecureStorage::new(Path::new("/nonexistent/path")).await;
    assert!(result.is_err(), "Expected error when initializing with nonexistent path");
    
    // Poskusi inicializirati storage z neveljavno mapo (datoteka namesto mape)
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("file.txt");
    
    std::fs::write(&file_path, b"test").expect("Failed to create test file");
    
    let result = SecureStorage::new(&file_path).await;
    assert!(result.is_err(), "Expected error when initializing with file path");
}

/// Test posodabljanja obstoječega ključa
#[test]
async fn test_update_existing_key() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = SecureStorage::new(path).await.expect("Failed to create storage");
    
    // Shrani podatke
    let key = "test_update";
    let value1 = b"original_value".to_vec();
    storage.store(key, &value1).await.expect("Failed to store data");
    
    // Posodobi podatke
    let value2 = b"updated_value".to_vec();
    storage.store(key, &value2).await.expect("Failed to update data");
    
    // Preveri, da so podatki posodobljeni
    let loaded = storage.load(key).await.expect("Failed to load data");
    assert_eq!(value2, loaded, "Loaded data doesn't match updated value");
}

/// Test obstojnosti med več instancami
#[test]
async fn test_persistence_across_instances() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    // Prva instanca
    {
        let storage = SecureStorage::new(path).await.expect("Failed to create storage");
        
        // Shrani podatke
        let key = "test_persistence";
        let value = b"persistent_value".to_vec();
        storage.store(key, &value).await.expect("Failed to store data");
    }
    
    // Druga instanca
    {
        let storage = SecureStorage::new(path).await.expect("Failed to create storage");
        
        // Naloži podatke
        let key = "test_persistence";
        let loaded = storage.load(key).await.expect("Failed to load data from new instance");
        assert_eq!(b"persistent_value".to_vec(), loaded, "Loaded data doesn't match across instances");
    }
}

/// Test rokovanja z napakami
#[test]
async fn test_error_handling() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = SecureStorage::new(path).await.expect("Failed to create storage");
    
    // NotFound napaka
    let result = storage.load("nonexistent_key").await;
    assert!(matches!(result, Err(StorageError::NotFound)), "Wrong error type for nonexistent key");
    
    // InvalidName napaka
    let result = storage.store("invalid/key", b"test").await;
    assert!(matches!(result, Err(StorageError::InvalidName)), "Wrong error type for invalid key name");
    
    // NotFound napaka pri brisanju
    let result = storage.delete("nonexistent_key").await;
    assert!(matches!(result, Err(StorageError::NotFound)), "Wrong error type when deleting nonexistent key");
}

/// Test kritične poti manipulacije s ključi
#[test]
async fn test_critical_key_operations() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = SecureStorage::new(path).await.expect("Failed to create storage");
    
    // Operacije, ki so označene kot kritične poti za MEV platformo
    
    // 1. Shranjevanje MEV ključa (nizka latenca kritična)
    let start = std::time::Instant::now();
    storage.store("mev_strategy_key", b"critical_mev_data").await
        .expect("Failed to store MEV key");
    let duration = start.elapsed();
    println!("MEV key storage latency: {:?}", duration);
    assert!(duration < Duration::from_millis(10), "MEV key storage too slow");
    
    // 2. Nalaganje MEV ključa (ultra nizka latenca kritična)
    let start = std::time::Instant::now();
    let _loaded = storage.load("mev_strategy_key").await
        .expect("Failed to load MEV key");
    let duration = start.elapsed();
    println!("MEV key loading latency: {:?}", duration);
    assert!(duration < Duration::from_millis(5), "MEV key loading too slow");
    
    // 3. Kombinirana operacija: Posodobitev in nalaganje (simulacija MEV scenarija)
    let start = std::time::Instant::now();
    
    // Posodobi ključ s svežimi podatki
    storage.store("mev_arbitrage_key", b"fresh_arbitrage_data").await
        .expect("Failed to update arbitrage key");
    
    // Kratka zakasnitev, simulira procesiranje transakcije
    sleep(Duration::from_micros(100)).await;
    
    // Naloži ključ za izvršitev operacije
    let _arbitrage_data = storage.load("mev_arbitrage_key").await
        .expect("Failed to load arbitrage key");
    
    let duration = start.elapsed();
    println!("Combined MEV operation latency: {:?}", duration);
    assert!(duration < Duration::from_millis(15), "Combined MEV operation too slow");
}
