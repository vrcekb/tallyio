//! # `TallyIO` Strategy Core Performance Benchmarks
//!
//! Ultra-performance benchmarks for the `strategy_core` crate measuring
//! strategy execution latency, coordination overhead, and throughput
//! for MEV operations handling millions of dollars in daily volume.
//!
//! ## Performance Requirements:
//! - Strategy Initialization: <1ms latency
//! - Arbitrage Execution: <10ms end-to-end
//! - Liquidation Detection: <5ms latency
//! - Cross-strategy Coordination: <2ms overhead
//! - Resource Allocation: <100μs latency

#![allow(missing_docs, reason = "Benchmark code doesn't require documentation")]
#![allow(clippy::unwrap_used, reason = "Benchmarks can panic on setup failure")]
#![allow(clippy::str_to_string, reason = "Test data creation uses to_string")]
#![allow(clippy::used_underscore_binding, reason = "Benchmark data can be prefixed")]
#![allow(clippy::let_underscore_must_use, reason = "Setup functions can be ignored")]
#![allow(clippy::let_underscore_untyped, reason = "Setup return types are obvious")]
#![allow(clippy::unit_arg, reason = "black_box accepts unit values")]
#![allow(clippy::semicolon_if_nothing_returned, reason = "Benchmark closures don't need semicolons")]
#![allow(clippy::explicit_iter_loop, reason = "Array iteration is clear")]
#![allow(clippy::cast_sign_loss, reason = "Test data is always positive")]
#![allow(clippy::unseparated_literal_suffix, reason = "Benchmark literals are readable")]
#![allow(clippy::doc_markdown, reason = "Already fixed in doc comments")]
#![allow(clippy::std_instead_of_core, reason = "Benchmarks use std features")]
#![allow(clippy::min_ident_chars, reason = "Single-char identifiers are standard in benchmarks")]
#![allow(clippy::implicit_return, reason = "Implicit returns are cleaner in closures")]
#![allow(clippy::default_numeric_fallback, reason = "Test ranges use default integers")]
#![allow(clippy::as_conversions, reason = "Controlled conversions in benchmark setup")]
#![allow(clippy::shadow_reuse, reason = "Iterator variables are commonly shadowed")]
#![allow(clippy::single_call_fn, reason = "Benchmark functions are called by criterion macros")]

use criterion::{
    criterion_group, criterion_main, Criterion, BenchmarkId, Throughput,
    black_box, BatchSize,
};
use std::time::Duration;

use strategy_core::{
    StrategyConfig, init_strategy_core,
    arbitrage::{
        ArbitrageConfig, ArbitrageCoordinator, DexArbitrageExecutor,
        FlashloanArbitrageExecutor
    },
    liquidation::{
        LiquidationConfig, LiquidationCoordinator
    },
};

use rust_decimal::Decimal;

/// Initialize strategy core for benchmarks
fn setup_strategy_core() {
    let config = StrategyConfig::default();
    let _ = init_strategy_core(&config);
}

/// Benchmark strategy core initialization
fn bench_strategy_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_initialization");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(1000);
    
    group.bench_function("strategy_core_init", |b| {
        b.iter(|| {
            let config = StrategyConfig::default();
            black_box(init_strategy_core(&config).unwrap())
        });
    });
    
    group.bench_function("arbitrage_coordinator_creation", |b| {
        setup_strategy_core();
        b.iter(|| {
            let config = ArbitrageConfig::default();
            black_box(ArbitrageCoordinator::new(config).unwrap())
        });
    });

    group.bench_function("liquidation_coordinator_creation", |b| {
        setup_strategy_core();
        b.iter(|| {
            let config = LiquidationConfig::default();
            black_box(LiquidationCoordinator::new(config).unwrap())
        });
    });
    
    group.finish();
}

/// Benchmark arbitrage strategy performance
fn bench_arbitrage_strategies(c: &mut Criterion) {
    setup_strategy_core();
    
    let mut group = c.benchmark_group("arbitrage_strategies");
    group.measurement_time(Duration::from_secs(5));
    
    group.bench_function("dex_arbitrage_executor_creation", |b| {
        b.iter(|| {
            let min_profit = Decimal::from_str_exact("0.01").unwrap();
            black_box(DexArbitrageExecutor::new(min_profit))
        });
    });

    group.bench_function("flashloan_arbitrage_executor_creation", |b| {
        b.iter(|| {
            black_box(FlashloanArbitrageExecutor::new())
        });
    });
    
    group.finish();
}

/// Benchmark liquidation strategy performance
fn bench_liquidation_strategies(c: &mut Criterion) {
    setup_strategy_core();

    let mut group = c.benchmark_group("liquidation_strategies");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("liquidation_opportunity_creation", |b| {
        b.iter(|| {
            // Create a simple benchmark for liquidation opportunity
            let _opportunity_data = (
                1u64, // position_id
                "AAVE".to_string(), // protocol
                Decimal::from_str_exact("1000.0").unwrap(), // collateral
                Decimal::from_str_exact("50.0").unwrap(), // profit
            );
            black_box(_opportunity_data)
        });
    });

    group.finish();
}

/// Benchmark configuration creation
fn bench_configuration_creation(c: &mut Criterion) {
    setup_strategy_core();

    let mut group = c.benchmark_group("configuration_creation");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("arbitrage_config_creation", |b| {
        b.iter(|| {
            black_box(ArbitrageConfig::default())
        });
    });

    group.bench_function("liquidation_config_creation", |b| {
        b.iter(|| {
            black_box(LiquidationConfig::default())
        });
    });

    group.finish();
}

/// Benchmark coordinator lifecycle operations
fn bench_coordinator_lifecycle(c: &mut Criterion) {
    setup_strategy_core();

    let mut group = c.benchmark_group("coordinator_lifecycle");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);

    group.bench_function("arbitrage_coordinator_start_stop", |b| {
        b.iter_batched(
            || {
                let config = ArbitrageConfig::default();
                ArbitrageCoordinator::new(config).unwrap()
            },
            |coordinator| {
                coordinator.start().unwrap();
                coordinator.stop().unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark multi-strategy coordination
fn bench_multi_strategy_coordination(c: &mut Criterion) {
    setup_strategy_core();

    let mut group = c.benchmark_group("multi_strategy_coordination");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("concurrent_strategy_execution", |b| {
        b.iter(|| {
            // Create multiple strategy components simultaneously
            let min_profit = Decimal::from_str_exact("0.01").unwrap();
            let _dex_executor = black_box(DexArbitrageExecutor::new(min_profit));
            let _flashloan_executor = black_box(FlashloanArbitrageExecutor::new());
        });
    });

    for strategy_count in [5, 10, 20, 50].iter() {
        group.throughput(Throughput::Elements(*strategy_count as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_strategy_creation", strategy_count),
            strategy_count,
            |b, &strategy_count| {
                b.iter(|| {
                    for _ in 0..strategy_count {
                        let min_profit = Decimal::from_str_exact("0.01").unwrap();
                        let _executor = black_box(DexArbitrageExecutor::new(min_profit));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_strategy_initialization,
    bench_arbitrage_strategies,
    bench_liquidation_strategies,
    bench_configuration_creation,
    bench_coordinator_lifecycle,
    bench_multi_strategy_coordination,
);

criterion_main!(benches);
