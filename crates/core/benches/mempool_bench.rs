//! Mempool benchmarks for TallyIO core
//!
//! This module provides benchmarks for mempool operations including
//! transaction analysis, filtering, and monitoring.

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
use tallyio_core::mempool::{
    FilterConfig, MempoolAnalyzer, MempoolFilter, MempoolManager, MempoolWatcher, TransactionFilter,
};
use tallyio_core::types::{Gas, Price, Transaction};

/// Benchmark transaction analysis
fn bench_transaction_analysis(c: &mut Criterion) {
    let analyzer = MempoolAnalyzer::new();

    let defi_tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(2),
        Price::from_gwei(50),
        Gas::new(150_000),
        0,
        vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method signature
    );

    c.bench_function("analyze_transaction", |b| {
        b.iter(|| {
            let result = analyzer.analyze_transaction(black_box(&defi_tx));
            black_box(result)
        })
    });
}

/// Benchmark transaction filtering
fn bench_transaction_filtering(c: &mut Criterion) {
    let filter = MempoolFilter::default();

    let tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    c.bench_function("filter_transaction", |b| {
        b.iter(|| {
            let result = filter.filter(black_box(&tx));
            black_box(result)
        })
    });
}

/// Benchmark mempool watcher
fn bench_mempool_watcher(c: &mut Criterion) {
    let mut watcher = MempoolWatcher::new();
    watcher.start().unwrap();

    c.bench_function("watch_for_transactions", |b| {
        b.iter(|| {
            let result = watcher.watch_for_transactions();
            black_box(result)
        })
    });
}

/// Benchmark mempool manager
fn bench_mempool_manager(c: &mut Criterion) {
    let mut manager = MempoolManager::default();
    manager.start().unwrap();

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
            let result = manager.process_transaction(black_box(tx.clone()));
            black_box(result)
        })
    });
}

/// Benchmark different transaction types
fn bench_transaction_types(c: &mut Criterion) {
    let analyzer = MempoolAnalyzer::new();

    let mut group = c.benchmark_group("transaction_types");

    // Simple transfer
    let simple_tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    // DeFi transaction
    let defi_tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(2),
        Price::from_gwei(50),
        Gas::new(150_000),
        0,
        vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00],
    );

    // Contract creation
    let contract_tx = Transaction::new(
        [1u8; 20],
        None, // Contract creation
        Price::new(0),
        Price::from_gwei(20),
        Gas::new(2_000_000),
        0,
        vec![0x60, 0x80, 0x60, 0x40], // Contract bytecode
    );

    group.bench_function("simple_transfer", |b| {
        b.iter(|| {
            let result = analyzer.analyze_transaction(black_box(&simple_tx));
            black_box(result)
        })
    });

    group.bench_function("defi_transaction", |b| {
        b.iter(|| {
            let result = analyzer.analyze_transaction(black_box(&defi_tx));
            black_box(result)
        })
    });

    group.bench_function("contract_creation", |b| {
        b.iter(|| {
            let result = analyzer.analyze_transaction(black_box(&contract_tx));
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark filter configurations
fn bench_filter_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_configurations");

    // Permissive filter
    let permissive_config = FilterConfig {
        min_gas_price_gwei: 1,
        max_gas_price_gwei: 1000,
        min_value_wei: 0,
        enable_defi_filter: false,
        enable_mev_filter: false,
    };
    let permissive_filter = MempoolFilter::new(permissive_config);

    // Strict filter
    let strict_config = FilterConfig {
        min_gas_price_gwei: 20,
        max_gas_price_gwei: 100,
        min_value_wei: 1_000_000_000_000_000_000, // 1 ETH
        enable_defi_filter: true,
        enable_mev_filter: true,
    };
    let strict_filter = MempoolFilter::new(strict_config);

    let tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    group.bench_function("permissive_filter", |b| {
        b.iter(|| {
            let result = permissive_filter.filter(black_box(&tx));
            black_box(result)
        })
    });

    group.bench_function("strict_filter", |b| {
        b.iter(|| {
            let result = strict_filter.filter(black_box(&tx));
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark batch processing
fn bench_batch_processing(c: &mut Criterion) {
    let analyzer = MempoolAnalyzer::new();
    let filter = MempoolFilter::default();

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

    let mut group = c.benchmark_group("batch_processing");

    group.bench_function("analyze_batch", |b| {
        b.iter(|| {
            for tx in &transactions {
                let result = analyzer.analyze_transaction(black_box(tx));
                black_box(result);
            }
        })
    });

    group.bench_function("filter_batch", |b| {
        b.iter(|| {
            for tx in &transactions {
                let result = filter.filter(black_box(tx));
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Benchmark high-frequency operations
fn bench_high_frequency(c: &mut Criterion) {
    let mut manager = MempoolManager::default();
    manager.start().unwrap();

    let mut group = c.benchmark_group("high_frequency");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("rapid_processing", |b| {
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
                let _ = manager.process_transaction(tx);
            }
        })
    });

    group.finish();
}

/// Benchmark statistics collection
fn bench_statistics(c: &mut Criterion) {
    let analyzer = MempoolAnalyzer::new();
    let filter = MempoolFilter::default();
    let manager = MempoolManager::default();

    // Generate some activity
    let tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    for _ in 0..100 {
        let _ = analyzer.analyze_transaction(&tx);
        let _ = filter.filter(&tx);
    }

    let mut group = c.benchmark_group("statistics");

    group.bench_function("analyzer_statistics", |b| {
        b.iter(|| {
            let stats = analyzer.statistics();
            black_box(stats)
        })
    });

    group.bench_function("filter_statistics", |b| {
        b.iter(|| {
            let stats = filter.statistics();
            black_box(stats)
        })
    });

    group.bench_function("manager_statistics", |b| {
        b.iter(|| {
            let stats = manager.statistics();
            black_box(stats)
        })
    });

    group.finish();
}

/// Benchmark MEV opportunity detection
fn bench_mev_detection(c: &mut Criterion) {
    let analyzer = MempoolAnalyzer::new();

    let mut group = c.benchmark_group("mev_detection");

    // Different MEV opportunity types
    let arbitrage_tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(2),
        Price::from_gwei(50),
        Gas::new(150_000),
        0,
        vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // swapExactTokensForTokens
    );

    let sandwich_tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(5),
        Price::from_gwei(80),
        Gas::new(200_000),
        0,
        vec![0x38, 0xed, 0x17, 0x39, 0x00, 0x00], // swapExactETHForTokens
    );

    let liquidation_tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(10),
        Price::from_gwei(100),
        Gas::new(400_000),
        0,
        vec![0x2e, 0x1a, 0x7d, 0x4d, 0x00, 0x00], // liquidateCalculateSeizeTokens
    );

    group.bench_function("arbitrage_detection", |b| {
        b.iter(|| {
            let result = analyzer.analyze_transaction(black_box(&arbitrage_tx));
            black_box(result)
        })
    });

    group.bench_function("sandwich_detection", |b| {
        b.iter(|| {
            let result = analyzer.analyze_transaction(black_box(&sandwich_tx));
            black_box(result)
        })
    });

    group.bench_function("liquidation_detection", |b| {
        b.iter(|| {
            let result = analyzer.analyze_transaction(black_box(&liquidation_tx));
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_transaction_analysis,
    bench_transaction_filtering,
    bench_mempool_watcher,
    bench_mempool_manager,
    bench_transaction_types,
    bench_filter_configurations,
    bench_batch_processing,
    bench_high_frequency,
    bench_statistics,
    bench_mev_detection
);

criterion_main!(benches);
