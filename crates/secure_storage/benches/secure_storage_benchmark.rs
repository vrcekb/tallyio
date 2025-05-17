use criterion::{black_box, criterion_group, criterion_main, Criterion};
use secure_storage::{Key, SecureStorage};
use std::path::PathBuf;
use tempfile::tempdir;

fn encrypt_benchmark(c: &mut Criterion) {
    let key = Key::generate().unwrap();
    let storage = SecureStorage::new(PathBuf::from("/dev/null"), key).unwrap();

    // Benchmark za različne velikosti podatkov
    let sizes = [64, 256, 1024, 4096, 16384];

    for size in sizes {
        let data = vec![42u8; size];
        c.bench_function(&format!("encrypt_{size}_bytes"), |b| {
            b.iter(|| {
                storage.encrypt(black_box(&data)).unwrap();
            });
        });
    }
}

fn decrypt_benchmark(c: &mut Criterion) {
    let key = Key::generate().unwrap();
    let storage = SecureStorage::new(PathBuf::from("/dev/null"), key).unwrap();

    // Benchmark za različne velikosti podatkov
    let sizes = [64, 256, 1024, 4096, 16384];

    for size in sizes {
        let data = vec![42u8; size];
        let (encrypted, nonce) = storage.encrypt(&data).unwrap();
        c.bench_function(&format!("decrypt_{size}_bytes"), |b| {
            b.iter(|| {
                storage.decrypt(black_box(&encrypted), black_box(nonce)).unwrap();
            });
        });
    }
}

fn store_load_benchmark(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let file = dir.path().join("bench.json");
    let key = Key::generate().unwrap();
    let storage = SecureStorage::new(&file, key).unwrap();

    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();

    // Benchmark za različne velikosti podatkov
    let sizes = [64, 256, 1024, 4096];

    for size in sizes {
        let data = vec![42u8; size];
        c.bench_function(&format!("store_load_{size}_bytes"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    storage.store("test", black_box(&data)).await.unwrap();
                    storage.load::<Vec<u8>>("test").await.unwrap()
                })
            });
        });
    }
}

criterion_group!(benches, encrypt_benchmark, decrypt_benchmark, store_load_benchmark);
criterion_main!(benches);
