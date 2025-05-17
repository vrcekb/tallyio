//! Robni primeri za testiranje secure_storage modula
//!
//! Ti testi se osredotočajo na robne primere in izjemne situacije, ki zagotavljajo odpornost
//! secure_storage modula v zahtevnih MEV okoliščinah, kjer je zanesljivost ključna.

use secure_storage::{SecureStorage, StorageError};
use std::fs;
use tempfile::tempdir;
use tokio::test;

/// Test za nalaganje podatkov s poškodovanim JSON formatom
#[test]
async fn test_load_data_corrupted_json() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("corrupted.json");
    
    // Ustvari instanco secure_storage
    let storage = SecureStorage::new(&temp_dir.path()).await.expect("Failed to create storage");
    
    // Najprej shrani nekaj veljavnih podatkov
    storage.store("test_key", b"test_value").await.expect("Failed to store initial data");
    
    // Preberi pot do datoteke (implementacija se lahko razlikuje)
    let actual_file_path = temp_dir.path().join("secure_storage.dat");
    
    // Počakaj, da se podatki zapišejo
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Preberi obstoječe podatke in jih poškoduj
    let content = fs::read(&actual_file_path).expect("Failed to read storage file");
    
    // Poškoduj JSON z rezanjem datoteke na pol
    fs::write(&actual_file_path, &content[0..content.len()/2]).expect("Failed to write corrupted data");
    
    // Poskusi naložiti podatke - pričakujemo napako
    let result = storage.load("test_key").await;
    assert!(result.is_err(), "Expected error when loading from corrupted file");
    
    // Preveri, da je tipa StorageError::LoadError
    if let Err(err) = result {
        match err {
            StorageError::LoadError(_) => {}, // Pričakovan tip napake
            _ => panic!("Unexpected error type: {:?}", err),
        }
    }
}

/// Test za dešifriranje poškodovanih podatkov
#[test]
async fn test_decrypt_corrupted_data() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    
    // Ustvari instanco secure_storage
    let storage = SecureStorage::new(&temp_dir.path()).await.expect("Failed to create storage");
    
    // Shrani nekaj podatkov
    storage.store("corrupt_test", b"original_data").await.expect("Failed to store data");
    
    // Preberi pot do datoteke (implementacija se lahko razlikuje)
    let actual_file_path = temp_dir.path().join("secure_storage.dat");
    
    // Počakaj, da se podatki zapišejo
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Preberi obstoječe podatke
    let content = fs::read(&actual_file_path).expect("Failed to read storage file");
    
    // Poškoduj podatke tako, da spremenimo nekaj bajtov v sredini
    let mut corrupted = content.clone();
    if corrupted.len() > 100 {
        for i in 50..70 {
            if i < corrupted.len() {
                corrupted[i] = corrupted[i].wrapping_add(1);
            }
        }
    }
    
    fs::write(&actual_file_path, &corrupted).expect("Failed to write corrupted data");
    
    // Poskusi naložiti podatke - pričakujemo napako
    let result = storage.load("corrupt_test").await;
    assert!(result.is_err(), "Expected error when decrypting corrupted data");
}

/// Test za različne velikosti podatkov pri šifriranju
#[test]
async fn test_encrypt_various_sizes() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    
    // Ustvari instanco secure_storage
    let storage = SecureStorage::new(&temp_dir.path()).await.expect("Failed to create storage");
    
    // Testiraj različne velikosti podatkov
    let test_sizes = vec![
        0, // Prazni podatki
        1, // En bajt
        16, // Velikost bloka za AES
        32, // Velikost ključa
        64, // Večkratnik bloka
        1023, // Skoraj 1KB
        1024, // Natančno 1KB
        1025, // Tik nad 1KB
        4096, // 4KB (tipična velikost strani)
        1024 * 1024 // 1MB
    ];
    
    for size in test_sizes {
        // Ustvari podatke ustrezne velikosti
        let data = vec![0xAA; size];
        let key = format!("size_test_{}", size);
        
        // Shrani podatke
        storage.store(&key, &data).await.expect(&format!("Failed to store {} bytes", size));
        
        // Naloži podatke
        let loaded = storage.load(&key).await.expect(&format!("Failed to load {} bytes", size));
        
        // Preveri, da so podatki enaki
        assert_eq!(data, loaded, "Data mismatch for size {}", size);
    }
}

/// Test za truncated podatke pri dešifriranju
#[test]
async fn test_decrypt_truncated_data() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    
    // Ustvari instanco secure_storage
    let storage = SecureStorage::new(&temp_dir.path()).await.expect("Failed to create storage");
    
    // Shrani nekaj podatkov
    storage.store("truncate_test", b"data_to_be_truncated").await.expect("Failed to store data");
    
    // Preberi pot do datoteke (implementacija se lahko razlikuje)
    let actual_file_path = temp_dir.path().join("secure_storage.dat");
    
    // Počakaj, da se podatki zapišejo
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Preberi obstoječe podatke
    let content = fs::read(&actual_file_path).expect("Failed to read storage file");
    
    // Skrajšaj podatke
    if content.len() > 30 {
        fs::write(&actual_file_path, &content[0..content.len()-30]).expect("Failed to write truncated data");
    }
    
    // Poskusi naložiti podatke - pričakujemo napako
    let result = storage.load("truncate_test").await;
    assert!(result.is_err(), "Expected error when loading truncated data");
}

/// Test za vzporedno shranjevanje in nalaganje pod obremenitvijo
#[test]
async fn test_concurrent_operations_under_load() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    
    // Ustvari instanco secure_storage
    let storage = Arc::new(SecureStorage::new(&temp_dir.path()).await.expect("Failed to create storage"));
    
    // Število vzporednih operacij
    let concurrent_ops = 100;
    
    // Testiraj vzporedno shranjevanje
    let mut handles = Vec::new();
    
    for i in 0..concurrent_ops {
        let storage_clone = Arc::clone(&storage);
        let key = format!("concurrent_key_{}", i);
        let value = format!("concurrent_value_{}", i).into_bytes();
        
        // Ustvari asinhron task
        let handle = tokio::spawn(async move {
            // Shrani podatke
            storage_clone.store(&key, &value).await.expect("Failed to store in concurrent operation");
            
            // Počakaj naključen čas
            let delay_ms = thread_rng().gen_range(1..50);
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            
            // Naloži podatke
            let loaded = storage_clone.load(&key).await.expect("Failed to load in concurrent operation");
            
            // Preveri
            assert_eq!(value, loaded, "Data mismatch in concurrent operation");
        });
        
        handles.push(handle);
    }
    
    // Počakaj na zaključek vseh operacij
    for handle in handles {
        handle.await.expect("Task failed");
    }
}

/// Test za kritične MEV operacije pod latency pressure
#[test]
async fn test_mev_operations_under_latency_pressure() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    
    // Ustvari instanco secure_storage
    let storage = Arc::new(SecureStorage::new(&temp_dir.path()).await.expect("Failed to create storage"));
    
    // Simuliraj MEV scenarij, kjer je več strategij v konkurenci
    let strategy_count = 5;
    let operations_per_strategy = 20;
    
    let mut handles = Vec::new();
    
    // Meritve latence
    let latencies = Arc::new(Mutex::new(Vec::new()));
    
    for strategy_id in 0..strategy_count {
        let storage_clone = Arc::clone(&storage);
        let latencies_clone = Arc::clone(&latencies);
        
        let handle = tokio::spawn(async move {
            for op_id in 0..operations_per_strategy {
                // Ključ za to operacijo
                let key = format!("mev_latency_strategy_{}_op_{}", strategy_id, op_id);
                let value = format!("value_{}_{}", strategy_id, op_id).into_bytes();
                
                // Merjenje latence
                let start = std::time::Instant::now();
                
                // Shrani podatke (kritična MEV operacija)
                storage_clone.store(&key, &value).await.expect("Failed to store in MEV operation");
                
                // Naloži podatke (kritična MEV operacija)
                let loaded = storage_clone.load(&key).await.expect("Failed to load in MEV operation");
                
                // Izmeri latenco
                let latency = start.elapsed();
                
                // Shrani latenco
                let mut latencies = latencies_clone.lock().await;
                latencies.push(latency);
                
                // Preveri podatke
                assert_eq!(value, loaded, "Data mismatch in MEV operation");
            }
        });
        
        handles.push(handle);
    }
    
    // Počakaj na zaključek vseh operacij
    for handle in handles {
        handle.await.expect("MEV task failed");
    }
    
    // Analiziraj latence
    let latencies = latencies.lock().await;
    
    // Izračunaj p99 latenco
    let mut sorted_latencies = latencies.clone();
    sorted_latencies.sort();
    
    let p99_index = (sorted_latencies.len() as f64 * 0.99) as usize;
    let p99_latency = sorted_latencies.get(p99_index).unwrap_or(&sorted_latencies.last().unwrap_or(&std::time::Duration::from_secs(0))).clone();
    
    println!("MEV operations P99 latency: {:?}", p99_latency);
    
    // Za MEV operacije je kritično, da je latenca pod 10ms
    assert!(p99_latency < std::time::Duration::from_millis(10), 
            "P99 latency for MEV operations too high: {:?}", p99_latency);
}

use std::sync::{Arc, Mutex};
use rand::{thread_rng, Rng};
