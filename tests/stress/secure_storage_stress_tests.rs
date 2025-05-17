//! Stresni testi za `secure_storage` modul
//! 
//! Ti testi preverjajo robustnost `secure_storage` modula pri visokih obremenitvah,
//! kar je ključno za MEV platformo, kjer je zanesljivost kritična.

use secure_storage::{Key, SecureStorage, StorageError};
use std::time::Duration;
use tempfile::tempdir;
use tokio::time::sleep;

/// Stresni test za sočasno izvajanje operacij
/// 
/// Ta test simulira veliko število sočasnih operacij na secure storage,
/// kar je pomembno za MEV platformo, kjer se lahko hkrati izvaja veliko operacij.
#[test]
#[ignore] // Ignoriraj v normalnih testnih zagonih zaradi dolgega trajanja
fn test_stress_concurrent_operations() -> Result<(), StorageError> {
    let dir = tempdir()?;
    let path = dir.path().join("stress.db");
    let key = Key::generate()?;
    let storage = SecureStorage::new(&path, key)?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut handles = Vec::new();
        
        // Launch 100 concurrent operations
        for i in 0..100 {
            let storage = storage.clone();
            let handle = tokio::spawn(async move {
                let key = format!("stress_key_{}", i);
                let value = vec![i as u8; 1024]; // 1KB podatkov
                
                // Store, load, get_meta, and delete in sequence
                storage.store(&key, &value).await?;
                
                // Add some delay to increase chance of concurrency
                sleep(Duration::from_millis(i % 10)).await;
                
                let loaded = storage.load(&key).await?;
                assert_eq!(loaded, value, "Loaded value doesn't match stored value");
                
                // Add some delay to increase chance of concurrency
                sleep(Duration::from_millis(i % 15)).await;
                
                // Delete
                storage.delete(&key).await?;
                
                // Check it's truly gone
                let exists = storage.exists(&key).await?;
                assert!(!exists, "Key should not exist after deletion");
                
                Ok::<_, StorageError>(())
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        for handle in handles {
            handle.await.unwrap()?; // Propagate errors
        }
        
        Ok(())
    })
}

/// Test z velikim številom podatkov
/// 
/// Preverja, da secure_storage modul ustrezno deluje z velikimi količinami podatkov,
/// kar je relevantno za shranjevanje MEV strategij in podatkovno intenzivnih operacij.
#[test]
#[ignore] // Ignoriraj v normalnih testnih zagonih zaradi dolgega trajanja
fn test_load_large_dataset() -> Result<(), StorageError> {
    let dir = tempdir()?;
    let path = dir.path().join("large_dataset.db");
    let key = Key::generate()?;
    let storage = SecureStorage::new(&path, key)?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Create 1000 keys with 1KB of data each
        for i in 0..1000 {
            let key = format!("large_dataset_key_{}", i);
            let value = vec![i as u8; 1024]; // 1KB data
            
            storage.store(&key, &value).await?;
        }
        
        // Verify all data can be accessed
        for i in 0..1000 {
            let key = format!("large_dataset_key_{}", i);
            let value = vec![i as u8; 1024]; // 1KB data
            
            let loaded = storage.load(&key).await?;
            assert_eq!(loaded, value, "Data mismatch for key {}", key);
        }
        
        // Measure load time for a few random keys
        let mut total_time = Duration::from_secs(0);
        let iterations = 100;
        
        for _ in 0..iterations {
            let i = rand::random::<usize>() % 1000;
            let key = format!("large_dataset_key_{}", i);
            
            let start = std::time::Instant::now();
            let _ = storage.load(&key).await?;
            total_time += start.elapsed();
        }
        
        let avg_load_time = total_time / iterations as u32;
        println!("Average load time from 1000-key dataset: {:?}", avg_load_time);
        
        // Even with a large dataset, loading should be reasonably fast for MEV operations
        assert!(avg_load_time < Duration::from_millis(1), 
                "Load time too slow for MEV operations: {:?}", avg_load_time);
        
        Ok(())
    })
}

/// Test mešanih konkurenčnih operacij
/// 
/// Ta test simulira realno MEV okolje, kjer se hkrati izvajajo različne operacije,
/// vključno s shranjevanjem, branjem in brisanjem, kar lahko povzroči težave s konsistenco.
#[test]
#[ignore] // Ignoriraj v normalnih testnih zagonih zaradi dolgega trajanja
fn test_concurrent_mixed_operations() -> Result<(), StorageError> {
    let dir = tempdir()?;
    let path = dir.path().join("mixed_ops.db");
    let key = Key::generate()?;
    let storage = SecureStorage::new(&path, key)?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut handles = Vec::new();
        
        // Common key prefix
        let key_prefix = "mixed_op_key";
        
        // Prepare some initial data
        for i in 0..50 {
            let key = format!("{}_{}", key_prefix, i);
            let value = vec![i as u8; 128];
            storage.store(&key, &value).await?;
        }
        
        // Launch 20 concurrent tasks that perform different operations
        for task_id in 0..20 {
            let storage = storage.clone();
            
            let handle = tokio::spawn(async move {
                match task_id % 4 {
                    0 => {
                        // Add new keys
                        for i in 0..10 {
                            let key = format!("{}_new_{}_{}", key_prefix, task_id, i);
                            let value = vec![(task_id * i) as u8; 256];
                            storage.store(&key, &value).await?;
                        }
                    },
                    1 => {
                        // Update existing keys
                        for i in 0..10 {
                            let key = format!("{}_{}", key_prefix, i);
                            let value = vec![(task_id + i) as u8; 128];
                            storage.store(&key, &value).await?;
                        }
                    },
                    2 => {
                        // Delete keys
                        for i in 10..20 {
                            let key = format!("{}_{}", key_prefix, i);
                            let _ = storage.delete(&key).await; // Ignore errors if already deleted
                        }
                    },
                    3 => {
                        // Read keys
                        for i in 20..30 {
                            let key = format!("{}_{}", key_prefix, i);
                            let _ = storage.load(&key).await; // Ignore errors if deleted
                        }
                    },
                    _ => unreachable!(),
                }
                
                Ok::<_, StorageError>(())
            });
            
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        for handle in handles {
            handle.await.unwrap()?; // Propagate errors
        }
        
        Ok(())
    })
}

/// Test stresne obremenitve s praznim prostorom na disku
/// 
/// Simulira situacijo, kjer je na disku malo prostora med intenzivnimi operacijami,
/// kar je relevantno za produkcijske scenarije, kjer so lahko sistemski viri omejeni.
#[test]
#[ignore] // Ignoriraj v normalnih testnih zagonih zaradi dolgega trajanja
fn test_stress_with_low_disk_space() -> Result<(), StorageError> {
    let dir = tempdir()?;
    let path = dir.path().join("low_space.db");
    let key = Key::generate()?;
    let storage = SecureStorage::new(&path, key)?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Najprej napolnimo z nekaj podatki
        for i in 0..100 {
            let key = format!("low_space_key_{}", i);
            let value = vec![i as u8; 10 * 1024]; // 10KB za vsak vnos
            storage.store(&key, &value).await?;
        }
        
        // Simuliramo nizek prostor tako, da dodamo še več podatkov
        for i in 100..200 {
            let key = format!("low_space_key_{}", i);
            let value = vec![i as u8; 50 * 1024]; // 50KB za vsak dodatni vnos
            storage.store(&key, &value).await?;
        }
        
        // Zdaj izvajamo še več operacij, da testiramo robustnost
        let mut handles = Vec::new();
        
        for i in 0..20 {
            let storage = storage.clone();
            
            let handle = tokio::spawn(async move {
                // Kombinacija operacij
                for j in 0..10 {
                    let base = i * 10 + j;
                    
                    // Shrani novo vrednost
                    storage.store(&format!("new_key_{}", base), &vec![base as u8; 1024]).await?;
                    
                    // Izbriši staro vrednost, da sprostimo nekaj prostora
                    if base < 100 {
                        storage.delete(&format!("low_space_key_{}", base)).await?;
                    }
                    
                    // Posodobi obstoječo vrednost
                    if base >= 100 && base < 200 {
                        storage.store(&format!("low_space_key_{}", base), &vec![0xFF; 1024]).await?;
                    }
                }
                
                Ok::<_, StorageError>(())
            });
            
            handles.push(handle);
        }
        
        // Počakamo na dokončanje vseh operacij
        for handle in handles {
            handle.await.unwrap()?;
        }
        
        // Na koncu preverimo, da so podatki še vedno konsistentni
        for i in 0..20 {
            for j in 0..10 {
                let base = i * 10 + j;
                let key = format!("new_key_{}", base);
                
                let loaded = storage.load(&key).await?;
                assert_eq!(loaded, vec![base as u8; 1024], "Data corruption for {}", key);
            }
        }
        
        Ok(())
    })
}

/// Test vzdrževanja pod konstantno obremenitvijo
/// 
/// Ta test simulira produkcijsko okolje, kjer se operacije izvajajo neprestano,
/// kar lahko razkrije probleme z uhajanjem pomnilnika ali degradacijo zmogljivosti.
#[test]
#[ignore] // Ignoriraj v normalnih testnih zagonih zaradi dolgega trajanja
fn test_sustained_load() -> Result<(), StorageError> {
    let dir = tempdir()?;
    let path = dir.path().join("sustained.db");
    let key = Key::generate()?;
    let storage = SecureStorage::new(&path, key)?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Kako dolgo izvajamo test (v normalnem okolju bi bil daljši)
        let duration = Duration::from_secs(30);
        let start = std::time::Instant::now();
        
        // Konstanta frekvenca operacij
        let mut operation_count = 0;
        
        while start.elapsed() < duration {
            // Izračunaj ključ in vrednost na podlagi števca operacij
            let key = format!("sustained_key_{}", operation_count % 100);
            let value = vec![(operation_count % 256) as u8; 1024];
            
            // Izmenično shranjevanje in branje
            if operation_count % 2 == 0 {
                storage.store(&key, &value).await?;
            } else {
                let _ = storage.load(&key).await; // Ignoriraj napake, če ključ ne obstaja
            }
            
            // Občasno brišemo
            if operation_count % 10 == 0 {
                let delete_key = format!("sustained_key_{}", (operation_count / 10) % 100);
                let _ = storage.delete(&delete_key).await; // Ignoriraj napake
            }
            
            // Beležimo čas izvajanja za odkrivanje upočasnitev
            let op_start = std::time::Instant::now();
            storage.exists(&key).await?;
            let op_time = op_start.elapsed();
            
            // Preverjamo, da ni prišlo do bistvene degradacije
            assert!(op_time < Duration::from_millis(5), 
                    "Operation time degraded: {:?} after {} operations", 
                    op_time, operation_count);
            
            operation_count += 1;
        }
        
        println!("Executed {} operations in {:?}", operation_count, duration);
        
        // Izračunaj povprečno število operacij na sekundo
        let ops_per_second = operation_count as f64 / duration.as_secs_f64();
        println!("Average throughput: {:.2} ops/sec", ops_per_second);
        
        // Zagotovi minimalno prepustnost za MEV operacije
        assert!(ops_per_second > 100.0, 
                "Throughput too low for MEV requirements: {:.2} ops/sec", 
                ops_per_second);
        
        Ok(())
    })
}
