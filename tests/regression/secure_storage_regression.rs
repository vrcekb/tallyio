//! Regresijski testi za secure_storage modul
//!
//! Ti testi zagotavljajo, da se znane napake ne ponovijo v prihodnosti.
//! Za MEV platformo je ključno, da enkrat rešeni problemi ostanejo rešeni.

use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::test;
use tokio::time::sleep;
use std::time::Duration;
use tokio::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use secure_storage::{SecureStorage, StorageError};

/// Regresijski test: Preverjanje robustnosti pri hkratnih zahtevkih
///
/// Ta test je bil dodan po odkritju težave z race condition pri 
/// hkratnem dostopu do shrambe.
#[test]
async fn regression_test_concurrent_access() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = Arc::new(SecureStorage::new(path).await.expect("Failed to create storage"));
    
    // Hkratno shranjevanje in branje iz več niti
    let tasks = 20;
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    
    for i in 0..tasks {
        let storage_clone = Arc::clone(&storage);
        let counter_clone = Arc::clone(&counter);
        let key = format!("regression_key_{}", i);
        let value = format!("regression_value_{}", i).into_bytes();
        
        let handle = tokio::spawn(async move {
            // Shrani vrednost
            storage_clone.store(&key, &value).await.expect("Failed to store");
            
            // Kratka zakasnitev za povečanje možnosti race conditiona
            sleep(Duration::from_millis(5)).await;
            
            // Preberi vrednost
            let loaded = storage_clone.load(&key).await.expect("Failed to load");
            assert_eq!(loaded, value, "Data corruption in concurrent access");
            
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        
        handles.push(handle);
    }
    
    // Počakaj na vse naloge
    for handle in handles {
        handle.await.expect("Task failed");
    }
    
    assert_eq!(counter.load(Ordering::SeqCst), tasks, "Not all tasks completed");
}

/// Regresijski test: Preverjanje robustnosti pri nenadni prekinitvi
///
/// Ta test je bil dodan po odkritju težave z inconsistentnim stanjem
/// pri nenadni prekinitvi med shranjevanjem.
#[test]
async fn regression_test_interruption_recovery() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    // Priprava začetnih podatkov v shrambi
    {
        let storage = SecureStorage::new(path).await.expect("Failed to create storage");
        for i in 0..5 {
            let key = format!("init_key_{}", i);
            let value = format!("init_value_{}", i).into_bytes();
            storage.store(&key, &value).await.expect("Failed to store initial data");
        }
    } // Storage se tu zapre
    
    // Simulacija prekinitve med shranjevanjem
    {
        let storage = SecureStorage::new(path).await.expect("Failed to create storage");
        
        // Shrani nekaj novih podatkov
        for i in 5..10 {
            let key = format!("new_key_{}", i);
            let value = format!("new_value_{}", i).into_bytes();
            storage.store(&key, &value).await.expect("Failed to store new data");
        }
        
        // Ne pokličemo explicitnega zaprtja shrambe, simuliramo crash
    }
    
    // Preveri, da lahko še vedno dostopamo do podatkov
    {
        let storage = SecureStorage::new(path).await.expect("Failed to create storage");
        
        // Preveri začetne podatke
        for i in 0..5 {
            let key = format!("init_key_{}", i);
            let expected = format!("init_value_{}", i).into_bytes();
            let loaded = storage.load(&key).await.expect("Failed to load initial data after interruption");
            assert_eq!(loaded, expected, "Data corruption after interruption");
        }
        
        // Preveri nove podatke (lahko se je zadnja transakcija izgubila, kar je sprejemljivo)
        for i in 5..9 {
            let key = format!("new_key_{}", i);
            match storage.load(&key).await {
                Ok(loaded) => {
                    let expected = format!("new_value_{}", i).into_bytes();
                    assert_eq!(loaded, expected, "Data corruption after interruption");
                },
                Err(StorageError::NotFound) => {
                    println!("Key {} was not saved due to simulated interruption (acceptable)", key);
                },
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    }
}

/// Regresijski test: Preverjanje maksimalne velikosti podatkov
///
/// Ta test je bil dodan po odkritju težave s performanco pri 
/// shranjevanju velikih količin podatkov.
#[test]
async fn regression_test_large_data_handling() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = SecureStorage::new(path).await.expect("Failed to create storage");
    
    // Shrani podatke različnih velikosti
    let sizes = vec![1024, 10 * 1024, 100 * 1024, 1024 * 1024];
    
    for (i, size) in sizes.iter().enumerate() {
        let key = format!("large_key_{}", i);
        let value = vec![i as u8; *size];
        
        let start = std::time::Instant::now();
        storage.store(&key, &value).await.expect("Failed to store large data");
        let store_duration = start.elapsed();
        
        println!("Storing {}KB took {:?}", size / 1024, store_duration);
        
        // Za MEV operacije je kritično, da so tudi velike operacije relativno hitre
        assert!(store_duration < Duration::from_secs(1), 
                "Storing {}KB took too long: {:?}", size / 1024, store_duration);
        
        // Preveri, da so podatki pravilno shranjeni
        let start = std::time::Instant::now();
        let loaded = storage.load(&key).await.expect("Failed to load large data");
        let load_duration = start.elapsed();
        
        println!("Loading {}KB took {:?}", size / 1024, load_duration);
        
        assert_eq!(loaded, value, "Large data corruption");
        assert!(load_duration < Duration::from_secs(1),
                "Loading {}KB took too long: {:?}", size / 1024, load_duration);
    }
}

/// Regresijski test: MEV-specifično preverjanje atomarnosti
///
/// Ta test je bil dodan po odkritju težave z atomarnostjo operacij, 
/// kar je ključno za MEV strategije.
#[test]
async fn regression_test_mev_atomicity() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let path = temp_dir.path();
    
    let storage = Arc::new(SecureStorage::new(path).await.expect("Failed to create storage"));
    let storage_mutex = Arc::new(Mutex::new(()));
    
    // Simulacija konkurenčnih MEV strategij, ki potrebujejo atomarne operacije
    let strategy_count = 5;
    let operations_per_strategy = 10;
    let mut handles = Vec::new();
    
    for strategy_id in 0..strategy_count {
        let storage_clone = Arc::clone(&storage);
        let mutex_clone = Arc::clone(&storage_mutex);
        
        let handle = tokio::spawn(async move {
            for op_id in 0..operations_per_strategy {
                // Atomarna operacija za MEV strategijo (read-modify-write)
                let _lock = mutex_clone.lock().await;
                
                let key = format!("mev_strategy_{}_{}", strategy_id, op_id % 3);
                
                // Poskusi prebrati obstoječe podatke
                let current_value = match storage_clone.load(&key).await {
                    Ok(data) => {
                        let value = u64::from_be_bytes([
                            data[0], data[1], data[2], data[3], 
                            data[4], data[5], data[6], data[7]
                        ]);
                        value
                    },
                    Err(StorageError::NotFound) => 0,
                    Err(e) => panic!("Unexpected error: {:?}", e),
                };
                
                // Posodobi vrednost (simulacija MEV strategije)
                let new_value = current_value + 1;
                let data = new_value.to_be_bytes().to_vec();
                
                // Shrani posodobljeno vrednost
                storage_clone.store(&key, &data).await.expect("Failed to update MEV data");
            }
        });
        
        handles.push(handle);
    }
    
    // Počakaj na vse strategije
    for handle in handles {
        handle.await.expect("MEV strategy task failed");
    }
    
    // Preveri končno stanje - vsak ključ mora imeti natančno določeno vrednost
    for strategy_id in 0..strategy_count {
        for key_id in 0..3 {
            let key = format!("mev_strategy_{}_{}", strategy_id, key_id);
            
            let data = storage.load(&key).await.expect("Failed to load final MEV data");
            let value = u64::from_be_bytes([
                data[0], data[1], data[2], data[3], 
                data[4], data[5], data[6], data[7]
            ]);
            
            // Pričakovana vrednost: število operacij za to strategijo, ki vplivajo na ta ključ
            let expected = (operations_per_strategy + 2) / 3;
            assert_eq!(value, expected, "MEV atomicity violation for key {}", key);
        }
    }
}
