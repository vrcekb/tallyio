//! Metrics benchmarks for TallyIO
//!
//! This module provides benchmarks for metrics collection and recording operations.

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
use tallyio_metrics::MetricsManager;

/// Benchmark metrics collection
fn bench_metrics_collection(c: &mut Criterion) {
    let manager = MetricsManager::new().expect("Failed to create metrics manager");

    c.bench_function("record_metric", |b| {
        b.iter(|| {
            let result = manager.record_metric(black_box("test_metric"));
            black_box(result)
        })
    });
}

/// Benchmark batch metrics recording
fn bench_batch_metrics(c: &mut Criterion) {
    let manager = MetricsManager::new().expect("Failed to create metrics manager");

    c.bench_function("batch_record_metrics", |b| {
        b.iter(|| {
            for i in 0..100 {
                let result = manager.record_metric(&format!("metric_{i}"));
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

    let manager = MetricsManager::new().expect("Failed to create metrics manager");

    group.bench_function("critical_path_recording", |b| {
        b.iter(|| {
            let result = manager.record_metric(black_box("critical_metric"));
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

    let manager = MetricsManager::new().expect("Failed to create metrics manager");

    group.bench_function("high_volume_recording", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let result = manager.record_metric(&format!("volume_metric_{i}"));
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Benchmark concurrent metrics recording
fn bench_concurrent_metrics(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let manager = Arc::new(MetricsManager::new().expect("Failed to create metrics manager"));

    c.bench_function("concurrent_recording", |b| {
        b.iter(|| {
            let mut handles = vec![];

            for i in 0..4 {
                let manager_clone = Arc::clone(&manager);
                let handle = thread::spawn(move || {
                    for j in 0..10 {
                        let result = manager_clone.record_metric(&format!("concurrent_{i}_{j}"));
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
    bench_metrics_collection,
    bench_batch_metrics,
    latency_benchmarks,
    throughput_benchmarks,
    bench_concurrent_metrics
);

criterion_main!(benches);
