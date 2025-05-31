//! Liquidation benchmarks for TallyIO
//!
//! This module provides benchmarks for liquidation strategies and processing.

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
use tallyio_liquidation::LiquidationManager;

/// Benchmark liquidation processing
fn bench_liquidation_processing(c: &mut Criterion) {
    let manager = LiquidationManager::new().expect("Failed to create liquidation manager");

    c.bench_function("process_liquidation", |b| {
        b.iter(|| {
            let result = manager.process_liquidation(black_box("test_liquidation"));
            black_box(result)
        })
    });
}

/// Benchmark batch liquidation processing
fn bench_batch_liquidations(c: &mut Criterion) {
    let manager = LiquidationManager::new().expect("Failed to create liquidation manager");

    c.bench_function("batch_process_liquidations", |b| {
        b.iter(|| {
            for i in 0..100 {
                let result = manager.process_liquidation(&format!("liquidation_{i}"));
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

    let manager = LiquidationManager::new().expect("Failed to create liquidation manager");

    group.bench_function("critical_path_liquidation", |b| {
        b.iter(|| {
            let result = manager.process_liquidation(black_box("critical_liquidation"));
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

    let manager = LiquidationManager::new().expect("Failed to create liquidation manager");

    group.bench_function("high_volume_liquidations", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let result = manager.process_liquidation(&format!("volume_liquidation_{i}"));
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Benchmark concurrent liquidation processing
fn bench_concurrent_liquidations(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let manager =
        Arc::new(LiquidationManager::new().expect("Failed to create liquidation manager"));

    c.bench_function("concurrent_liquidations", |b| {
        b.iter(|| {
            let mut handles = vec![];

            for i in 0..4 {
                let manager_clone = Arc::clone(&manager);
                let handle = thread::spawn(move || {
                    for j in 0..10 {
                        let result =
                            manager_clone.process_liquidation(&format!("concurrent_{i}_{j}"));
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
    bench_liquidation_processing,
    bench_batch_liquidations,
    latency_benchmarks,
    throughput_benchmarks,
    bench_concurrent_liquidations
);

criterion_main!(benches);
