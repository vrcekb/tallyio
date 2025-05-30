//! Memory optimization benchmarks for TallyIO core
//!
//! This module provides benchmarks for memory pool operations
//! and other memory-related optimizations.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::uninlined_format_args,
    clippy::doc_markdown,
    clippy::semicolon_if_nothing_returned,
    clippy::default_numeric_fallback,
    clippy::explicit_iter_loop
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use tallyio_core::optimization::{memory_pool::MemoryPoolConfig, MemoryPool};

/// Benchmark memory pool allocation
fn bench_memory_pool_allocation(c: &mut Criterion) {
    let pool = MemoryPool::new(64 * 1024 * 1024).expect("Failed to create memory pool");

    c.bench_function("memory_pool_get_buffer", |b| {
        b.iter(|| {
            let buffer = pool.get_buffer(black_box(4096));
            black_box(buffer)
        })
    });
}

/// Benchmark memory pool vs standard allocation
fn bench_allocation_comparison(c: &mut Criterion) {
    let pool = MemoryPool::new(64 * 1024 * 1024).expect("Failed to create memory pool");

    let mut group = c.benchmark_group("allocation_comparison");

    group.bench_function("memory_pool", |b| {
        b.iter(|| {
            let _buffer = pool.get_buffer(black_box(4096)).unwrap();
        })
    });

    group.bench_function("standard_allocation", |b| {
        b.iter(|| {
            let _buffer = vec![0u8; black_box(4096)];
        })
    });

    group.finish();
}

/// Benchmark different buffer sizes
fn bench_buffer_sizes(c: &mut Criterion) {
    let pool = MemoryPool::new(64 * 1024 * 1024).expect("Failed to create memory pool");

    let mut group = c.benchmark_group("buffer_sizes");

    for size in [1024, 4096, 16384, 65536].iter() {
        group.bench_with_input(format!("size_{}", size), size, |b, &size| {
            b.iter(|| {
                let _buffer = pool.get_buffer(black_box(size)).unwrap();
            })
        });
    }

    group.finish();
}

/// Benchmark memory pool configuration impact
fn bench_pool_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_configurations");

    // Small pool
    let small_config = MemoryPoolConfig {
        pool_size: 1024 * 1024, // 1MB
        block_size: 1024,       // 1KB blocks
        ..Default::default()
    };
    let small_pool = MemoryPool::with_config(small_config).unwrap();

    // Large pool
    let large_config = MemoryPoolConfig {
        pool_size: 64 * 1024 * 1024, // 64MB
        block_size: 4096,            // 4KB blocks
        ..Default::default()
    };
    let large_pool = MemoryPool::with_config(large_config).unwrap();

    group.bench_function("small_pool", |b| {
        b.iter(|| {
            let _buffer = small_pool.get_buffer(black_box(1024)).unwrap();
        })
    });

    group.bench_function("large_pool", |b| {
        b.iter(|| {
            let _buffer = large_pool.get_buffer(black_box(4096)).unwrap();
        })
    });

    group.finish();
}

/// Benchmark memory pool statistics collection
fn bench_pool_statistics(c: &mut Criterion) {
    let pool = MemoryPool::new(64 * 1024 * 1024).expect("Failed to create memory pool");

    // Generate some activity
    for _ in 0..100 {
        let _buffer = pool.get_buffer(4096).unwrap();
    }

    c.bench_function("pool_statistics", |b| {
        b.iter(|| {
            let stats = pool.statistics();
            black_box(stats)
        })
    });
}

/// Benchmark high-frequency allocations
fn bench_high_frequency_allocations(c: &mut Criterion) {
    let pool = MemoryPool::new(64 * 1024 * 1024).expect("Failed to create memory pool");

    let mut group = c.benchmark_group("high_frequency");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("rapid_allocations", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let _buffer = pool.get_buffer(black_box(4096)).unwrap();
            }
        })
    });

    group.finish();
}

/// Benchmark memory pool under contention
fn bench_pool_contention(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let pool = Arc::new(MemoryPool::new(64 * 1024 * 1024).expect("Failed to create memory pool"));

    c.bench_function("pool_contention", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let pool_clone = Arc::clone(&pool);
                    thread::spawn(move || {
                        for _ in 0..100 {
                            let _buffer = pool_clone.get_buffer(4096).unwrap();
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        })
    });
}

/// Benchmark buffer operations
fn bench_buffer_operations(c: &mut Criterion) {
    let pool = MemoryPool::new(64 * 1024 * 1024).expect("Failed to create memory pool");

    let mut group = c.benchmark_group("buffer_operations");

    group.bench_function("buffer_fill", |b| {
        b.iter(|| {
            let mut buffer = pool.get_buffer(4096).unwrap();
            buffer.fill(black_box(42));
        })
    });

    group.bench_function("buffer_resize", |b| {
        b.iter(|| {
            let mut buffer = pool.get_buffer(1024).unwrap();
            buffer.resize(black_box(2048));
        })
    });

    group.bench_function("buffer_clear", |b| {
        b.iter(|| {
            let mut buffer = pool.get_buffer(4096).unwrap();
            buffer.clear();
        })
    });

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_usage_patterns(c: &mut Criterion) {
    let pool = MemoryPool::new(64 * 1024 * 1024).expect("Failed to create memory pool");

    let mut group = c.benchmark_group("usage_patterns");

    // Sequential pattern
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let buffers: Vec<_> = (0..100).map(|_| pool.get_buffer(4096).unwrap()).collect();
            black_box(buffers);
        })
    });

    // Random sizes pattern
    group.bench_function("random_sizes", |b| {
        b.iter(|| {
            let sizes = [1024, 2048, 4096, 8192];
            let buffers: Vec<_> = (0..100)
                .map(|i| pool.get_buffer(sizes[i % sizes.len()]).unwrap())
                .collect();
            black_box(buffers);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_pool_allocation,
    bench_allocation_comparison,
    bench_buffer_sizes,
    bench_pool_configurations,
    bench_pool_statistics,
    bench_high_frequency_allocations,
    bench_pool_contention,
    bench_buffer_operations,
    bench_usage_patterns
);

criterion_main!(benches);
