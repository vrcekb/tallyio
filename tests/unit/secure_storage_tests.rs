//! Dodatni testi za izboljšanje pokritosti kode v `secure_storage` modulu.
//! 
//! Ti testi preverjajo osnovno funkcionalnost v `secure_storage` modulu,
//! s poudarkom na varnosti in robustnosti shranjevanja občutljivih podatkov.

use secure_storage::{Key, SecureStorage, StorageError};
use std::fs;
use tempfile::tempdir;
use tokio::runtime::Runtime;

/// Test osnovnih operacij s `SecureStorage`
#[test]
fn test_basic_operations() {
    // Ustvarimo tokio runtime za izvajanje async funkcij
    let rt = Runtime::new().expect("Failed to create runtime");

    // Ustvarimo začasni direktorij za teste
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let storage_path = temp_dir.path().join("secure_data.bin");

    // Ustvarimo ključ
    let key = Key::generate().expect("Failed to generate key");
    
    // Ustvarimo secure storage
    let storage = rt.block_on(async {
        SecureStorage::new(&storage_path, key.clone()).expect("Failed to create storage")
    });
    
    // Test: Shranjevanje podatka
    rt.block_on(async {
        storage.store("test_key", b"testvalue").expect("Failed to store data");
    });

    // Test: Nalaganje podatka
    let loaded_data = rt.block_on(async {
        storage.load("test_key").expect("Failed to load data")
    });
    assert_eq!(loaded_data, b"testvalue");

    // Test: Preverjanje, če podatek obstaja
    let exists = rt.block_on(async {
        storage.exists("test_key").expect("Failed to check existence")
    });
    assert!(exists);

    // Test: Preverjanje, če neobstoječ podatek ne obstaja
    let not_exists = rt.block_on(async {
        storage.exists("nonexistent").expect("Failed to check existence")
    });
    assert!(!not_exists);

    // Test: Brisanje podatka
    rt.block_on(async {
        storage.delete("test_key").expect("Failed to delete data");
    });

    // Test: Preverjanje, da je podatek bil izbrisan
    let exists_after_delete = rt.block_on(async {
        storage.exists("test_key").expect("Failed to check existence")
    });
    assert!(!exists_after_delete);
}

/// Test napačnih vhodnih podatkov
#[test]
fn test_invalid_inputs() {
    // Ustvarimo tokio runtime za izvajanje async funkcij
    let rt = Runtime::new().expect("Failed to create runtime");

    // Ustvarimo začasni direktorij za teste
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let storage_path = temp_dir.path().join("secure_data.bin");

    // Ustvarimo ključ
    let key = Key::generate().expect("Failed to generate key");
    
    // Ustvarimo secure storage
    let storage = rt.block_on(async {
        SecureStorage::new(&storage_path, key.clone()).expect("Failed to create storage")
    });
    
    // Test: Poskus shranjevanja podatka s praznim imenom
    let result = rt.block_on(async {
        storage.store("", b"testvalue").await
    });
    assert!(matches!(result, Err(StorageError::InvalidName)));

    // Test: Poskus shranjevanja podatka z predolgim imenom
    let long_name = "a".repeat(256); // Predolgo ime
    let result = rt.block_on(async {
        storage.store(&long_name, b"testvalue").await
    });
    assert!(matches!(result, Err(StorageError::InvalidName)));

    // Test: Poskus nalaganja neobstoječega podatka
    let result = rt.block_on(async {
        storage.load("nonexistent").await
    });
    assert!(matches!(result, Err(StorageError::NotFound)));

    // Test: Poskus brisanja neobstoječega podatka
    let result = rt.block_on(async {
        storage.delete("nonexistent").await
    });
    assert!(matches!(result, Err(StorageError::NotFound)));

    // Test: Poskus inicializacije s praznim ključem
    let invalid_key_result = rt.block_on(async {
        SecureStorage::new(&storage_path, Key::from_bytes(&[0u8; 0]).unwrap_err())
    });
    assert!(invalid_key_result.is_err());
}

/// Test šifriranja in dešifriranja podatkov
#[test]
fn test_encryption_decryption() -> Result<(), StorageError> {
    // Ustvarimo tokio runtime
    let rt = Runtime::new().expect("Failed to create runtime");

    // Ustvarimo začasno mapo in ključ
    let temp_dir = tempdir()?;
    let path = temp_dir.path().join("crypto_test.bin");
    let key = Key::generate()?;

    // Podatki za test
    let test_data = b"Zelo obcutljivi podatki za MEV operacije";
    
    // Ustvarimo storage, shranimo in naložimo podatke
    rt.block_on(async {
        let storage = SecureStorage::new(&path, key.clone())?;
        
        // Shranimo podatke
        storage.store("crypto_test", test_data).await?;
        
        // Preverimo, da so podatki v datoteki šifrirani
        let raw_data = fs::read(&path)?;
        assert!(!raw_data.windows(test_data.len()).any(|window| window == test_data));
        
        // Naložimo podatke in preverimo, da so pravilno dešifrirani
        let loaded = storage.load("crypto_test").await?;
        assert_eq!(&loaded, test_data);
        
        Ok(())
    })
}

/// Test shranjevanja in nalaganja večje količine podatkov
#[test]
fn test_store_and_load_large_data() -> Result<(), StorageError> {
    // Ustvarimo tokio runtime
    let rt = Runtime::new().expect("Failed to create runtime");

    // Ustvarimo začasno mapo in ključ
    let temp_dir = tempdir()?;
    let path = temp_dir.path().join("large_data_test.bin");
    let key = Key::generate()?;

    // Ustvarimo storage
    let storage = rt.block_on(async {
        SecureStorage::new(&path, key.clone())
    })?;

    // Ustvarimo večjo količino podatkov (1 MB)
    let large_data = vec![0xAA; 1_000_000];
    
    // Shranimo podatke
    rt.block_on(async {
        storage.store("large_test", &large_data).await?;
        
        // Naložimo podatke in preverimo
        let loaded = storage.load("large_test").await?;
        assert_eq!(loaded.len(), large_data.len());
        assert_eq!(loaded, large_data);
        
        Ok(())
    })
}

/// Test za shranjevanje več ključev in persistenco
#[test]
fn test_multiple_keys_and_persistence() -> Result<(), StorageError> {
    // Ustvarimo tokio runtime
    let rt = Runtime::new().expect("Failed to create runtime");

    // Ustvarimo začasno mapo in ključ
    let temp_dir = tempdir()?;
    let path = temp_dir.path().join("persistence_test.bin");
    let key = Key::generate()?;

    // Prvi storage
    {
        let storage = rt.block_on(async {
            SecureStorage::new(&path, key.clone())
        })?;
        
        // Shranimo več različnih ključev
        rt.block_on(async {
            storage.store("key1", b"value1").await?;
            storage.store("key2", b"value2").await?;
            storage.store("key3", b"value3").await?;
            Ok(())
        })?;
    } // Storage bo tu uničen
    
    // Ustvarimo nov storage s istim ključem in preverimo, da so podatki ohranjeni
    {
        let storage = rt.block_on(async {
            SecureStorage::new(&path, key.clone())
        })?;
        
        rt.block_on(async {
            // Preverimo vse tri ključe
            assert_eq!(storage.load("key1").await?, b"value1");
            assert_eq!(storage.load("key2").await?, b"value2");
            assert_eq!(storage.load("key3").await?, b"value3");
            
            // Preverimo, da lahko dodamo nov ključ
            storage.store("key4", b"value4").await?;
            assert_eq!(storage.load("key4").await?, b"value4");
            
            // Preverimo, da lahko izbrišemo ključ
            storage.delete("key2").await?;
            assert!(storage.load("key2").await.is_err());
            
            Ok(())
        })
    }
}

/// Test zaščite pred prepisovanjem podatkov
#[test]
fn test_data_integrity() -> Result<(), StorageError> {
    // Ustvarimo tokio runtime
    let rt = Runtime::new().expect("Failed to create runtime");

    // Ustvarimo začasno mapo in ključa
    let temp_dir = tempdir()?;
    let path = temp_dir.path().join("integrity_test.bin");
    let key1 = Key::generate()?;
    let key2 = Key::generate()?;
    
    // Najprej shranimo podatke s prvim ključem
    {
        let storage = rt.block_on(async {
            SecureStorage::new(&path, key1.clone())
        })?;
        
        rt.block_on(async {
            storage.store("sensitive_key", b"sensitive_value").await?;
            Ok(())
        })?;
    }
    
    // Poskusimo prebrati podatke z napačnim ključem
    {
        let storage = rt.block_on(async {
            SecureStorage::new(&path, key2.clone())
        })?;
        
        let result = rt.block_on(async {
            storage.load("sensitive_key").await
        });
        
        // Pričakujemo napako pri dešifriranju
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(matches!(err, StorageError::DecryptionError));
        }
    }
    
    // Poskusimo prepisati datoteko z novim ključem
    {
        let storage = rt.block_on(async {
            SecureStorage::new(&path, key2.clone())
        })?;
        
        // Shranimo nove podatke
        rt.block_on(async {
            storage.store("new_key", b"new_value").await?;
            Ok(())
        })?;
    }
    
    // Preverimo, da so originalni podatki izgubljeni (ker smo prepisali datoteko)
    {
        let storage = rt.block_on(async {
            SecureStorage::new(&path, key1.clone())
        })?;
        
        let result = rt.block_on(async {
            storage.load("sensitive_key").await
        });
        
        // Pričakujemo napako, ker so podatki prepisani
        assert!(result.is_err());
    }
    
    Ok(())
}

/// Test robnih primerov
#[test]
fn test_edge_cases() -> Result<(), StorageError> {
    // Ustvarimo tokio runtime
    let rt = Runtime::new().expect("Failed to create runtime");

    // Ustvarimo začasno mapo in ključ
    let temp_dir = tempdir()?;
    let path = temp_dir.path().join("edge_cases.bin");
    let key = Key::generate()?;
    
    let storage = rt.block_on(async {
        SecureStorage::new(&path, key.clone())
    })?;
    
    rt.block_on(async {
        // Test: Shranjevanje praznih podatkov
        storage.store("empty", b"").await?;
        let loaded = storage.load("empty").await?;
        assert_eq!(loaded, b"");
        
        // Test: Posebni znaki v imenu ključa
        let special_key = "special!@#$%^&*()_+-=[]{}|;:,./<>?";
        storage.store(special_key, b"special_value").await?;
        assert_eq!(storage.load(special_key).await?, b"special_value");
        
        // Test: Preverjanje obstoja ključa po brisanju in ponovnem ustvarjanju
        storage.store("temp", b"temp_value").await?;
        storage.delete("temp").await?;
        assert!(!storage.exists("temp").await?);
        storage.store("temp", b"new_temp_value").await?;
        assert_eq!(storage.load("temp").await?, b"new_temp_value");
        
        // Test: Posodobitev obstoječega ključa
        storage.store("update_key", b"original").await?;
        storage.store("update_key", b"updated").await?;
        assert_eq!(storage.load("update_key").await?, b"updated");
        
        Ok(())
    })
}
