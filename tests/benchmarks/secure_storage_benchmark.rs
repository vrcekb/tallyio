//! Benchmark testi za secure_storage modul
//!
//! Ti testi merijo performančne karakteristike secure_storage modula, s posebnim
//! poudarkom na MEV-kritičnih operacijah, kjer je nizka latenca ključnega pomena.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tempfile::tempdir;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;
use rand::{thread_rng, Rng};

use secure_storage::SecureStorage;

/// Izvedba asinhronih operacij v benchmark testu
fn run_async<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let rt = Runtime::new().expect("Failed to create runtime");
    rt.block_on(fut)
}

/// Benchmark: Shranjevanje podatkov
fn bench_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("SecureStorage Store");
    
    // Benchmark velikosti podatkov
    for size in [32, 64, 128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let temp_dir = tempdir().expect("Failed to create temp directory");
            let path = temp_dir.path();
            
            // Pripravi podatke
            let key = format!("bench_key_{}", size);
            let mut data = vec![0u8; size];
            thread_rng().fill(&mut data[..]);
            
            // Inicializiraj storage
            let storage = Arc::new(run_async(async {
                SecureStorage::new(path).await.expect("Failed to create storage")
            }));
            
            b.iter(|| {
                let storage_clone = Arc::clone(&storage);
                let key_clone = key.clone();
                let data_clone = data.clone();
                
                run_async(async move {
                    black_box(
                        storage_clone.store(&key_clone, &data_clone).await
                            .expect("Failed to store data")
                    )
                })
            });
        });
    }
    
    group.finish();
}

/// Benchmark: Nalaganje podatkov
fn bench_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("SecureStorage Load");
    
    // Benchmark različnih velikosti
    for size in [32, 64, 128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let temp_dir = tempdir().expect("Failed to create temp directory");
            let path = temp_dir.path();
            
            // Pripravi podatke in storage
            let key = format!("bench_key_{}", size);
            let mut data = vec![0u8; size];
            thread_rng().fill(&mut data[..]);
            
            // Inicializiraj storage in shrani podatke za test
            let storage = Arc::new(run_async(async {
                let storage = SecureStorage::new(path).await.expect("Failed to create storage");
                storage.store(&key, &data).await.expect("Failed to store data");
                storage
            }));
            
            b.iter(|| {
                let storage_clone = Arc::clone(&storage);
                let key_clone = key.clone();
                
                run_async(async move {
                    black_box(
                        storage_clone.load(&key_clone).await
                            .expect("Failed to load data")
                    )
                })
            });
        });
    }
    
    group.finish();
}

/// Benchmark: Brisanje podatkov
fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("SecureStorage Delete");
    
    group.bench_function("delete", |b| {
        b.iter(|| {
            let temp_dir = tempdir().expect("Failed to create temp directory");
            let path = temp_dir.path();
            
            // Generiraj unikaten ključ za vsako iteracijo
            let key = format!("bench_delete_{}", thread_rng().gen::<u64>());
            let data = vec![0u8; 64];
            
            // Inicializiraj storage in shrani podatke za brisanje
            run_async(async {
                let storage = SecureStorage::new(path).await.expect("Failed to create storage");
                storage.store(&key, &data).await.expect("Failed to store data");
                
                // Benchmark brisanja
                black_box(
                    storage.delete(&key).await.expect("Failed to delete data")
                )
            })
        });
    });
    
    group.finish();
}

/// Benchmark: MEV specifične operacije
fn bench_mev_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("MEV Operations");
    
    // Benchmark MEV ključni primer uporabe: hitra posodobitev in branje
    group.bench_function("mev_update_and_read", |b| {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let path = temp_dir.path();
        
        // Pripravi podatke
        let key = "mev_strategy_key";
        let initial_data = vec![0u8; 64];
        let updated_data = vec![1u8; 64];
        
        // Inicializiraj storage in shrani začetne podatke
        let storage = Arc::new(run_async(async {
            let storage = SecureStorage::new(path).await.expect("Failed to create storage");
            storage.store(key, &initial_data).await.expect("Failed to store initial data");
            storage
        }));
        
        b.iter(|| {
            let storage_clone = Arc::clone(&storage);
            let updated_data_clone = updated_data.clone();
            
            run_async(async move {
                // Posodobi ključ (simulira posodobitev strategije)
                storage_clone.store(key, &updated_data_clone).await.expect("Failed to update data");
                
                // Takoj preberi podatke (simulira izvedbo strategije)
                black_box(
                    storage_clone.load(key).await.expect("Failed to load data")
                )
            })
        });
    });
    
    // Benchmark konkurenčnih MEV operacij
    group.bench_function("mev_concurrent_operations", |b| {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let path = temp_dir.path();
        
        // Inicializiraj storage
        let storage = Arc::new(run_async(async {
            SecureStorage::new(path).await.expect("Failed to create storage")
        }));
        
        b.iter(|| {
            let storage_clone = Arc::clone(&storage);
            
            run_async(async move {
                let mut handles = Vec::new();
                
                // Ustvari 10 sočasnih operacij (simulira MEV konkurenčne zahteve)
                for i in 0..10 {
                    let storage_task = Arc::clone(&storage_clone);
                    let key = format!("mev_concurrent_key_{}", i);
                    let data = vec![i as u8; 64];
                    
                    let handle = tokio::spawn(async move {
                        // Shrani podatke
                        storage_task.store(&key, &data).await.expect("Failed to store data");
                        
                        // Takoj preberi podatke
                        let loaded = storage_task.load(&key).await.expect("Failed to load data");
                        
                        // Preveri pravilnost
                        assert_eq!(data, loaded, "Data mismatch in concurrent operations");
                    });
                    
                    handles.push(handle);
                }
                
                // Počakaj na vse operacije
                for handle in handles {
                    black_box(handle.await.expect("Task failed"));
                }
            })
        });
    });
    
    group.finish();
}

/// Benchmark: Ekstremne MEV latence
fn bench_mev_latency_critical(c: &mut Criterion) {
    let mut group = c.benchmark_group("MEV Critical Path Latency");
    
    // Nastavitve za latence pod 1ms
    group.sample_size(1000); // Več vzorcev za natančnejše meritve
    
    group.bench_function("mev_ultra_low_latency_load", |b| {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let path = temp_dir.path();
        
        // Priprava: MEV kritični podatki so majhni (< 100 bytes)
        let key = "mev_critical_key";
        let data = vec![1u8; 64];
        
        // Inicializiraj storage in shrani podatke
        let storage = Arc::new(run_async(async {
            let storage = SecureStorage::new(path).await.expect("Failed to create storage");
            // Predgrej storage z nekaj operacijami (simulacija realnega delovanja)
            for i in 0..10 {
                storage.store(&format!("warmup_key_{}", i), &vec![0u8; 32]).await.expect("Failed warmup");
            }
            // Shrani kritični ključ
            storage.store(key, &data).await.expect("Failed to store data");
            storage
        }));
        
        b.iter(|| {
            let storage_clone = Arc::clone(&storage);
            
            run_async(async move {
                // To je kritična MEV pot, kjer pričakujemo <1ms latenco
                black_box(
                    storage_clone.load(key).await.expect("Failed to load critical MEV data")
                )
            })
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_store,
    bench_load,
    bench_delete,
    bench_mev_operations,
    bench_mev_latency_critical
);
criterion_main!(benches);
