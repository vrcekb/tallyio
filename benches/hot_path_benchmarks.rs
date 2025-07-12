//! # TallyIO Hot Path Performance Benchmarks
//!
//! Ultra-precise benchmarks for the hot_path crate measuring nanosecond-level performance.
//! These benchmarks validate that the system meets the strict latency requirements
//! for handling millions of dollars in daily trading volume.
//!
//! ## Performance Requirements:
//! - MEV Detection: <500ns latency
//! - Cross-chain Execution: <50ns latency
//! - Memory Allocation: <5ns overhead
//! - Crypto Operations: <50μs processing
//! - End-to-end Latency: <10ms total

#![allow(missing_docs)]

use criterion::{
    criterion_group, criterion_main, Criterion, BenchmarkId, Throughput,
    black_box, BatchSize,
};
use std::time::Duration;

use hot_path::{
    HotPathConfig, initialize,
    detection::{detect_opportunities, update_price_feed, OpportunityScanner},
    execution::{execute_opportunity, AtomicExecutor, GasOptimizer, BundleBuilder},
    types::{MarketSnapshot, TradingPair, AlignedPrice, Opportunity},
    atomic::{AtomicCounter, LockFreeQueue},
    memory::{ArenaAllocator, ObjectPool, RingBuffer},
    simd::{SimdCapabilities},
};

/// Initialize hot path for benchmarks
fn setup_hot_path() {
    let config = HotPathConfig::default();
    let _ = initialize(config);
}

/// Benchmark MEV detection pipeline with various market sizes
fn bench_mev_detection(c: &mut Criterion) {
    setup_hot_path();
    
    let mut group = c.benchmark_group("mev_detection");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);
    
    // Test different market sizes
    for size in [100, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("detect_opportunities", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut snapshot = MarketSnapshot::new(size);
                        for i in 0..size {
                            let pair = TradingPair::new(i as u32, (i + 1) as u32, (100 + i) as u16, (200 + i) as u16);
                            let price = AlignedPrice::new(1000 + i as u64, 123_456_789, 500);
                            snapshot.add_price(pair, price);
                        }
                        snapshot
                    },
                    |snapshot| {
                        black_box(detect_opportunities(&snapshot).unwrap())
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark opportunity execution performance
fn bench_opportunity_execution(c: &mut Criterion) {
    setup_hot_path();
    
    let mut group = c.benchmark_group("opportunity_execution");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10000);
    
    group.bench_function("execute_single_opportunity", |b| {
        b.iter_batched(
            || Opportunity::new(1, "arbitrage".to_string(), 1000, 95),
            |opportunity| {
                black_box(execute_opportunity(&opportunity).unwrap())
            },
            BatchSize::SmallInput,
        );
    });
    
    // Benchmark atomic executor directly
    group.bench_function("atomic_executor", |b| {
        let executor = AtomicExecutor::new();
        b.iter_batched(
            || Opportunity::new(1, "arbitrage".to_string(), 1000, 95),
            |opportunity| {
                black_box(executor.execute(&opportunity).unwrap())
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Benchmark atomic operations performance
fn bench_atomic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("atomic_operations");
    group.measurement_time(Duration::from_secs(5));

    // Atomic counter benchmarks
    group.bench_function("atomic_counter_increment", |b| {
        let counter = AtomicCounter::new();
        b.iter(|| {
            black_box(counter.increment())
        });
    });

    group.bench_function("atomic_counter_get", |b| {
        let counter = AtomicCounter::new();
        b.iter(|| {
            black_box(counter.get())
        });
    });

    // Lock-free queue benchmarks
    group.bench_function("lockfree_queue_enqueue_dequeue", |b| {
        let queue = LockFreeQueue::new(10000);
        let mut counter = 0u64;

        b.iter(|| {
            counter += 1;
            let _ = black_box(queue.enqueue(counter));
            let _ = black_box(queue.dequeue());
        });
    });

    group.finish();
}

/// Benchmark memory management performance
fn bench_memory_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_management");
    group.measurement_time(Duration::from_secs(5));

    // Arena allocator creation benchmark
    group.bench_function("arena_allocator_creation", |b| {
        b.iter(|| {
            black_box(ArenaAllocator::new(1024 * 1024))
        });
    });

    // Object pool creation benchmark
    group.bench_function("object_pool_creation", |b| {
        b.iter(|| {
            black_box(ObjectPool::<u64>::new(1000))
        });
    });

    // Ring buffer creation benchmark
    group.bench_function("ring_buffer_creation", |b| {
        b.iter(|| {
            black_box(RingBuffer::<u64>::new(1024))
        });
    });

    group.finish();
}

/// Benchmark SIMD operations performance
fn bench_simd_operations(c: &mut Criterion) {
    let caps = SimdCapabilities::detect();

    if !caps.has_simd() {
        return; // Skip SIMD benchmarks if not supported
    }

    let mut group = c.benchmark_group("simd_operations");
    group.measurement_time(Duration::from_secs(5));

    // Basic SIMD capability test
    group.bench_function("simd_capabilities_detection", |b| {
        b.iter(|| {
            black_box(SimdCapabilities::detect())
        });
    });

    group.finish();
}

/// Benchmark price feed updates
fn bench_price_feed_updates(c: &mut Criterion) {
    setup_hot_path();
    
    let mut group = c.benchmark_group("price_feed_updates");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10000);
    
    group.bench_function("single_price_update", |b| {
        let mut counter = 0u64;
        b.iter(|| {
            counter += 1;
            let price = AlignedPrice::new(1000 + counter, 123_456_789, 500);
            black_box(update_price_feed(counter as u32, price).unwrap())
        });
    });
    
    // Batch price updates
    for batch_size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_price_updates", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    for i in 0..batch_size {
                        let price = AlignedPrice::new(1000 + i as u64, 123_456_789, 500);
                        let _ = black_box(update_price_feed(i as u32, price));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark gas optimization
fn bench_gas_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("gas_optimization");
    group.measurement_time(Duration::from_secs(5));
    
    group.bench_function("gas_optimizer", |b| {
        let optimizer = GasOptimizer::new();
        let tx_data = b"dummy_transaction_data_for_gas_optimization";
        
        b.iter(|| {
            black_box(optimizer.optimize(tx_data).unwrap())
        });
    });
    
    group.bench_function("bundle_builder", |b| {
        b.iter_batched(
            || {
                let mut builder = BundleBuilder::new();
                builder.start_bundle("test_bundle".to_string());
                builder
            },
            |mut builder| {
                for i in 0..10 {
                    let _ = builder.add_transaction(format!("0x{:x}", i), 21000);
                }
                black_box(builder.finalize_bundle().unwrap())
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Benchmark opportunity scanner
fn bench_opportunity_scanner(c: &mut Criterion) {
    let mut group = c.benchmark_group("opportunity_scanner");
    group.measurement_time(Duration::from_secs(5));
    
    group.bench_function("opportunity_scanner", |b| {
        let scanner = OpportunityScanner::new();
        
        b.iter_batched(
            || {
                let mut snapshot = MarketSnapshot::new(1000);
                for i in 0..1000 {
                    let pair = TradingPair::new(i as u32, (i + 1) as u32, (100 + i) as u16, (200 + i) as u16);
                    let price = AlignedPrice::new(1000 + i as u64, 123_456_789, 500);
                    snapshot.add_price(pair, price);
                }
                snapshot
            },
            |snapshot| {
                black_box(scanner.scan(&snapshot).unwrap())
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Comprehensive end-to-end benchmark
fn bench_end_to_end_pipeline(c: &mut Criterion) {
    setup_hot_path();
    
    let mut group = c.benchmark_group("end_to_end_pipeline");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);
    
    group.bench_function("complete_mev_pipeline", |b| {
        b.iter_batched(
            || {
                let mut snapshot = MarketSnapshot::new(1000);
                for i in 0..1000 {
                    let pair = TradingPair::new(i as u32, (i + 1) as u32, (100 + i) as u16, (200 + i) as u16);
                    let price = AlignedPrice::new(1000 + i as u64, 123_456_789, 500);
                    snapshot.add_price(pair, price);
                }
                snapshot
            },
            |snapshot| {
                // Complete MEV pipeline: detection -> execution
                let opportunities = black_box(detect_opportunities(&snapshot).unwrap());
                if let Some(opportunity) = opportunities.first() {
                    let _result = black_box(execute_opportunity(opportunity).unwrap());
                }
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_mev_detection,
    bench_opportunity_execution,
    bench_atomic_operations,
    bench_memory_management,
    bench_simd_operations,
    bench_price_feed_updates,
    bench_gas_optimization,
    bench_opportunity_scanner,
    bench_end_to_end_pipeline,
);

criterion_main!(benches);
