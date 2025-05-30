//! Engine benchmarks for TallyIO core
//!
//! This module provides comprehensive benchmarks for the TallyIO engine
//! to ensure <1ms latency requirements are met.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::doc_markdown,
    clippy::semicolon_if_nothing_returned,
    clippy::default_numeric_fallback,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use tallyio_core::engine::{Controllable, Monitorable, TallyEngine, TransactionProcessor};
use tallyio_core::types::{Gas, Price, Transaction};

/// Benchmark transaction processing
fn bench_transaction_processing(c: &mut Criterion) {
    let mut engine = TallyEngine::new().expect("Failed to create engine");
    engine.start().expect("Failed to start engine");

    let tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    c.bench_function("process_transaction", |b| {
        b.iter(|| {
            let result = engine.process_transaction(black_box(tx.clone()));
            black_box(result)
        })
    });
}

/// Benchmark MEV opportunity detection
fn bench_mev_detection(c: &mut Criterion) {
    let engine = TallyEngine::new().expect("Failed to create engine");

    let defi_tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(2),
        Price::from_gwei(50),
        Gas::new(150_000),
        0,
        vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method signature
    );

    c.bench_function("mev_detection", |b| {
        b.iter(|| {
            let result = engine.scan_mev_opportunity(black_box(&defi_tx));
            black_box(result)
        })
    });
}

/// Benchmark transaction submission
fn bench_transaction_submission(c: &mut Criterion) {
    let mut engine = TallyEngine::new().expect("Failed to create engine");
    engine.start().expect("Failed to start engine");

    c.bench_function("submit_transaction", |b| {
        b.iter(|| {
            let tx = Transaction::new(
                [1u8; 20],
                Some([2u8; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(21_000),
                0,
                Vec::with_capacity(0),
            );
            let result = engine.submit_transaction(black_box(tx));
            black_box(result)
        })
    });
}

/// Benchmark batch processing
fn bench_batch_processing(c: &mut Criterion) {
    let engine = TallyEngine::new().expect("Failed to create engine");

    let transactions: Vec<Transaction> = (0..100)
        .map(|i| {
            Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(21_000),
                i as u64,
                Vec::with_capacity(0),
            )
        })
        .collect();

    c.bench_function("batch_processing", |b| {
        b.iter(|| {
            let result = engine.process_batch(black_box(transactions.clone()));
            black_box(result)
        })
    });
}

/// Benchmark engine metrics collection
fn bench_metrics_collection(c: &mut Criterion) {
    let engine = TallyEngine::new().expect("Failed to create engine");

    c.bench_function("metrics_collection", |b| {
        b.iter(|| {
            let result = engine.metrics();
            black_box(result)
        })
    });
}

/// Benchmark health check
fn bench_health_check(c: &mut Criterion) {
    let engine = TallyEngine::new().expect("Failed to create engine");

    c.bench_function("health_check", |b| {
        b.iter(|| {
            let result = engine.health_check();
            black_box(result)
        })
    });
}

/// Latency-focused benchmark group
fn latency_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    let mut engine = TallyEngine::new().expect("Failed to create engine");
    engine.start().expect("Failed to start engine");

    let tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    group.bench_function("critical_path", |b| {
        b.iter(|| {
            engine.submit_transaction(black_box(tx.clone())).unwrap();
            let result = engine.process_next();
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

    let mut engine = TallyEngine::new().expect("Failed to create engine");
    engine.start().expect("Failed to start engine");

    group.bench_function("high_volume", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let tx = Transaction::new(
                    [i as u8; 20],
                    Some([(i + 1) as u8; 20]),
                    Price::from_ether(1),
                    Price::from_gwei(20),
                    Gas::new(21_000),
                    i as u64,
                    Vec::with_capacity(0),
                );
                engine.submit_transaction(tx).unwrap();
            }

            // Process all transactions
            while engine.queue_size() > 0 {
                let _ = engine.process_next();
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_transaction_processing,
    bench_mev_detection,
    bench_transaction_submission,
    bench_batch_processing,
    bench_metrics_collection,
    bench_health_check,
    latency_benchmarks,
    throughput_benchmarks
);

criterion_main!(benches);
