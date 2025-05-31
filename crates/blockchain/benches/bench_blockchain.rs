//! Blockchain benchmarks for TallyIO
//!
//! This module provides benchmarks for blockchain operations including
//! block processing and chain management.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    unused_must_use,
    clippy::doc_markdown,
    clippy::semicolon_if_nothing_returned,
    clippy::default_numeric_fallback,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use tallyio_blockchain::BlockchainManager;

/// Benchmark block processing
fn bench_block_processing(c: &mut Criterion) {
    let manager = BlockchainManager::new().expect("Failed to create blockchain manager");

    c.bench_function("process_block", |b| {
        b.iter(|| {
            let result = manager.process_block(black_box("test_block"));
            black_box(result)
        })
    });
}

/// Benchmark batch block processing
fn bench_batch_blocks(c: &mut Criterion) {
    let manager = BlockchainManager::new().expect("Failed to create blockchain manager");

    c.bench_function("batch_process_blocks", |b| {
        b.iter(|| {
            for i in 0..100 {
                let result = manager.process_block(&format!("block_{i}"));
                black_box(result);
            }
        })
    });
}

/// Latency-focused benchmark group
fn latency_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    let manager = BlockchainManager::new().expect("Failed to create blockchain manager");

    group.bench_function("critical_path_block", |b| {
        b.iter(|| {
            let result = manager.process_block(black_box("critical_block"));
            black_box(result)
        })
    });

    group.finish();
}

/// Throughput-focused benchmark group
fn throughput_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(100);

    let manager = BlockchainManager::new().expect("Failed to create blockchain manager");

    group.bench_function("high_volume_blocks", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let result = manager.process_block(&format!("volume_block_{i}"));
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Benchmark concurrent block processing
fn bench_concurrent_blocks(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let manager = Arc::new(BlockchainManager::new().expect("Failed to create blockchain manager"));

    c.bench_function("concurrent_blocks", |b| {
        b.iter(|| {
            let mut handles = vec![];

            for i in 0..4 {
                let manager_clone = Arc::clone(&manager);
                let handle = thread::spawn(move || {
                    for j in 0..10 {
                        let result = manager_clone.process_block(&format!("concurrent_{i}_{j}"));
                        black_box(result);
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().expect("Thread join failed");
            }
        })
    });
}

criterion_group!(
    benches,
    bench_block_processing,
    bench_batch_blocks,
    latency_benchmarks,
    throughput_benchmarks,
    bench_concurrent_blocks
);

criterion_main!(benches);
