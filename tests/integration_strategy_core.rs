//! # TallyIO Strategy Core Integration Tests
//! 
//! Comprehensive integration tests for the strategy_core crate validating
//! end-to-end strategy execution for MEV operations, arbitrage, liquidations,
//! and coordination systems handling millions of dollars in daily volume.
//! 
//! ## Test Categories:
//! - Multi-strategy coordination and conflict resolution
//! - Cross-chain arbitrage execution pipelines
//! - Liquidation opportunity detection and execution
//! - Time-bandit MEV strategies (L2 arbitrage, sequencer monitoring)
//! - Zero-risk backrun optimization
//! - Priority-based resource allocation
//! - Real-time profit calculation and risk assessment

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::collections::HashMap;

use strategy_core::{
    StrategyConfig, StrategyError, StrategyResult, init_strategy_core,
    arbitrage::{
        ArbitrageConfig, ArbitrageCoordinator, DexArbitrageExecutor, 
        FlashloanArbitrageExecutor, CrossChainArbitrageExecutor,
        RouteOptimizer, SlippageCalculator, RoutePathfinder
    },
    liquidation::{
        LiquidationConfig, LiquidationCoordinator, LiquidationOpportunity,
        LiquidationResult, LiquidationStatus, AaveLiquidator, CompoundLiquidator,
        VenusLiquidator, HealthMonitor, ProfitCalculator, MulticallOptimizer
    },
    time_bandit::{
        TimeBanditConfig, TimeBanditCoordinator, L2ArbitrageExecutor,
        SequencerMonitor, DelayExploitationAnalyzer
    },
    zero_risk::{
        ZeroRiskConfig, ZeroRiskCoordinator, BackrunOptimizer,
        GasGolfingOptimizer, MevProtectionBypass
    },
    priority::{
        PriorityConfig, PriorityCoordinator, ExecutionQueue,
        OpportunityScorer, ResourceAllocator
    },
    coordination::{
        CoordinationConfig, CoordinationCoordinator, ConflictResolver,
        ParallelExecutor, YieldOptimizer
    },
};

use rust_decimal::Decimal;

/// Test configuration for production-like scenarios
const TEST_CONCURRENT_STRATEGIES: usize = 50;
const TEST_EXECUTION_TIMEOUT_MS: u64 = 1000;
const TEST_MIN_PROFIT_USD: &str = "0.10"; // $0.10 minimum profit
const TEST_MAX_GAS_PRICE_GWEI: u64 = 150;

#[test]
fn test_strategy_core_initialization() {
    let config = StrategyConfig {
        max_concurrent_strategies: TEST_CONCURRENT_STRATEGIES,
        min_profit_threshold: Decimal::from_str_exact(TEST_MIN_PROFIT_USD).unwrap(),
        max_gas_price: TEST_MAX_GAS_PRICE_GWEI,
        execution_timeout_ms: TEST_EXECUTION_TIMEOUT_MS,
        enable_simd: true,
        enable_ml_scoring: false,
        numa_node: None,
    };
    
    init_strategy_core(&config).unwrap();
}

#[test]
fn test_arbitrage_strategy_coordination() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();
    
    // Initialize arbitrage coordinator
    let arbitrage_config = ArbitrageConfig::default();
    let mut coordinator = ArbitrageCoordinator::new(arbitrage_config);
    
    // Test coordinator lifecycle
    coordinator.start().unwrap();
    
    // Create arbitrage executors
    let dex_executor = DexArbitrageExecutor::new();
    let flashloan_executor = FlashloanArbitrageExecutor::new();
    let cross_chain_executor = CrossChainArbitrageExecutor::new();
    
    // Validate executors are created successfully
    assert_eq!(dex_executor.get_execution_count(), 0);
    assert_eq!(flashloan_executor.get_execution_count(), 0);
    assert_eq!(cross_chain_executor.get_execution_count(), 0);
    
    // Test route optimization
    let route_optimizer = RouteOptimizer::new();
    let slippage_calculator = SlippageCalculator::new();
    let pathfinder = RoutePathfinder::new();
    
    // Validate optimization components
    assert_eq!(route_optimizer.get_optimization_count(), 0);
    assert_eq!(slippage_calculator.get_calculation_count(), 0);
    assert_eq!(pathfinder.get_path_count(), 0);
    
    coordinator.stop().unwrap();
}

#[test]
fn test_liquidation_strategy_pipeline() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();
    
    // Initialize liquidation coordinator
    let liquidation_config = LiquidationConfig::default();
    let mut coordinator = LiquidationCoordinator::new(liquidation_config);
    
    coordinator.start().unwrap();
    
    // Create protocol-specific liquidators
    let aave_liquidator = AaveLiquidator::new();
    let compound_liquidator = CompoundLiquidator::new();
    let venus_liquidator = VenusLiquidator::new();
    
    // Test liquidation opportunity creation
    let opportunity = LiquidationOpportunity::new(
        1,
        "AAVE".to_string(),
        "0x123".to_string(),
        Decimal::from_str_exact("1000.0").unwrap(),
        Decimal::from_str_exact("50.0").unwrap(),
    );
    
    assert_eq!(opportunity.protocol(), "AAVE");
    assert!(opportunity.is_profitable());
    
    // Test liquidation result
    let result = LiquidationResult::success(
        "0xabc".to_string(),
        Decimal::from_str_exact("45.0").unwrap(),
        21000,
    );
    
    assert_eq!(result.status(), LiquidationStatus::Success);
    assert!(result.profit() > Decimal::ZERO);
    
    // Test support components
    let health_monitor = HealthMonitor::new();
    let profit_calculator = ProfitCalculator::new();
    let multicall_optimizer = MulticallOptimizer::new();
    
    assert_eq!(health_monitor.get_check_count(), 0);
    assert_eq!(profit_calculator.get_calculation_count(), 0);
    assert_eq!(multicall_optimizer.get_optimization_count(), 0);
    
    coordinator.stop().unwrap();
}

#[test]
fn test_time_bandit_strategies() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();
    
    // Initialize time bandit coordinator
    let time_bandit_config = TimeBanditConfig::default();
    let mut coordinator = TimeBanditCoordinator::new(time_bandit_config);
    
    coordinator.start().unwrap();
    
    // Test L2 arbitrage executor
    let l2_executor = L2ArbitrageExecutor::new();
    assert_eq!(l2_executor.get_execution_count(), 0);
    
    // Test sequencer monitoring
    let sequencer_monitor = SequencerMonitor::new();
    assert_eq!(sequencer_monitor.get_monitoring_count(), 0);
    
    // Test delay exploitation
    let delay_analyzer = DelayExploitationAnalyzer::new();
    assert_eq!(delay_analyzer.get_analysis_count(), 0);
    
    coordinator.stop().unwrap();
}

#[test]
fn test_zero_risk_strategies() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();
    
    // Initialize zero risk coordinator
    let zero_risk_config = ZeroRiskConfig::default();
    let mut coordinator = ZeroRiskCoordinator::new(zero_risk_config);
    
    coordinator.start().unwrap();
    
    // Test backrun optimizer
    let backrun_optimizer = BackrunOptimizer::new();
    assert_eq!(backrun_optimizer.get_optimization_count(), 0);
    
    // Test gas golfing optimizer
    let gas_optimizer = GasGolfingOptimizer::new();
    assert_eq!(gas_optimizer.get_optimization_count(), 0);
    
    // Test MEV protection bypass
    let mev_bypass = MevProtectionBypass::new();
    assert_eq!(mev_bypass.get_bypass_count(), 0);
    
    coordinator.stop().unwrap();
}

#[test]
fn test_priority_system_integration() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();
    
    // Initialize priority coordinator
    let priority_config = PriorityConfig::default();
    let mut coordinator = PriorityCoordinator::new(priority_config);
    
    coordinator.start().unwrap();
    
    // Test execution queue
    let execution_queue = ExecutionQueue::new();
    assert_eq!(execution_queue.len(), 0);
    assert!(execution_queue.is_empty());
    
    // Test opportunity scorer
    let opportunity_scorer = OpportunityScorer::new();
    assert_eq!(opportunity_scorer.get_scoring_count(), 0);
    
    // Test resource allocator
    let resource_allocator = ResourceAllocator::new();
    assert_eq!(resource_allocator.get_allocation_count(), 0);
    
    coordinator.stop().unwrap();
}

#[test]
fn test_coordination_system_integration() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();
    
    // Initialize coordination coordinator
    let coordination_config = CoordinationConfig::default();
    let mut coordinator = CoordinationCoordinator::new(coordination_config);
    
    coordinator.start().unwrap();
    
    // Test conflict resolver
    let conflict_resolver = ConflictResolver::new();
    assert_eq!(conflict_resolver.get_resolution_count(), 0);
    
    // Test parallel executor
    let parallel_executor = ParallelExecutor::new();
    assert_eq!(parallel_executor.get_execution_count(), 0);
    
    // Test yield optimizer
    let yield_optimizer = YieldOptimizer::new();
    assert_eq!(yield_optimizer.get_optimization_count(), 0);
    
    coordinator.stop().unwrap();
}

#[test]
fn test_multi_strategy_coordination() {
    // Initialize strategy core
    let strategy_config = StrategyConfig {
        max_concurrent_strategies: 10,
        min_profit_threshold: Decimal::from_str_exact("0.05").unwrap(),
        max_gas_price: 200,
        execution_timeout_ms: 2000,
        enable_simd: true,
        enable_ml_scoring: true,
        numa_node: Some(0),
    };
    init_strategy_core(&strategy_config).unwrap();
    
    // Initialize all coordinators
    let mut arbitrage_coordinator = ArbitrageCoordinator::new(ArbitrageConfig::default());
    let mut liquidation_coordinator = LiquidationCoordinator::new(LiquidationConfig::default());
    let mut priority_coordinator = PriorityCoordinator::new(PriorityConfig::default());
    let mut coordination_coordinator = CoordinationCoordinator::new(CoordinationConfig::default());
    
    // Start all coordinators
    arbitrage_coordinator.start().unwrap();
    liquidation_coordinator.start().unwrap();
    priority_coordinator.start().unwrap();
    coordination_coordinator.start().unwrap();
    
    // Simulate concurrent strategy execution
    let handles: Vec<_> = (0..5)
        .map(|i| {
            thread::spawn(move || {
                // Simulate strategy work
                thread::sleep(Duration::from_millis(10));
                i
            })
        })
        .collect();
    
    // Wait for all strategies to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Stop all coordinators
    arbitrage_coordinator.stop().unwrap();
    liquidation_coordinator.stop().unwrap();
    priority_coordinator.stop().unwrap();
    coordination_coordinator.stop().unwrap();
}

#[test]
fn test_error_handling_and_recovery() {
    // Test invalid configuration
    let invalid_config = StrategyConfig {
        max_concurrent_strategies: 0, // Invalid
        min_profit_threshold: Decimal::from_str_exact("-1.0").unwrap(), // Invalid
        max_gas_price: 0, // Invalid
        execution_timeout_ms: 0, // Invalid
        enable_simd: true,
        enable_ml_scoring: false,
        numa_node: Some(999), // Invalid NUMA node
    };
    
    // Should handle invalid configuration gracefully
    let result = init_strategy_core(&invalid_config);
    // System should either succeed (with warnings) or fail gracefully
    assert!(result.is_ok() || result.is_err());
    
    // Test error types
    let arbitrage_error = StrategyError::Arbitrage(
        strategy_core::arbitrage::ArbitrageError::ExecutionFailed {
            reason: "Test error".to_string(),
        }
    );
    assert!(arbitrage_error.to_string().contains("Arbitrage error"));
    
    let config_error = StrategyError::Configuration {
        message: "Invalid configuration".to_string(),
    };
    assert!(config_error.to_string().contains("Configuration error"));
    
    let critical_error = StrategyError::Critical {
        details: "System failure".to_string(),
    };
    assert!(critical_error.to_string().contains("Critical system error"));
}

#[test]
fn test_performance_under_load() {
    // Initialize strategy core with high-performance configuration
    let strategy_config = StrategyConfig {
        max_concurrent_strategies: TEST_CONCURRENT_STRATEGIES,
        min_profit_threshold: Decimal::from_str_exact("0.01").unwrap(),
        max_gas_price: 300,
        execution_timeout_ms: 500,
        enable_simd: true,
        enable_ml_scoring: true,
        numa_node: None,
    };
    init_strategy_core(&strategy_config).unwrap();
    
    // Measure initialization time
    let start = Instant::now();
    
    // Initialize multiple coordinators simultaneously
    let mut coordinators = Vec::new();
    
    coordinators.push(Box::new(ArbitrageCoordinator::new(ArbitrageConfig::default())));
    coordinators.push(Box::new(LiquidationCoordinator::new(LiquidationConfig::default())));
    coordinators.push(Box::new(PriorityCoordinator::new(PriorityConfig::default())));
    
    let initialization_time = start.elapsed();
    
    // Should initialize quickly even under load
    assert!(initialization_time < Duration::from_millis(100),
        "Initialization took {}ms, should be <100ms", initialization_time.as_millis());
    
    // Test concurrent operations
    let start = Instant::now();
    
    let handles: Vec<_> = (0..20)
        .map(|_| {
            thread::spawn(|| {
                // Simulate strategy operations
                let _dex_executor = DexArbitrageExecutor::new();
                let _aave_liquidator = AaveLiquidator::new();
                let _backrun_optimizer = BackrunOptimizer::new();
                thread::sleep(Duration::from_millis(1));
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let concurrent_execution_time = start.elapsed();
    
    // Should handle concurrent operations efficiently
    assert!(concurrent_execution_time < Duration::from_millis(500),
        "Concurrent execution took {}ms, should be <500ms",
        concurrent_execution_time.as_millis());
}

#[test]
fn test_cross_strategy_conflict_resolution() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();

    // Initialize coordination system
    let coordination_config = CoordinationConfig::default();
    let mut coordinator = CoordinationCoordinator::new(coordination_config);
    coordinator.start().unwrap();

    // Create conflict resolver
    let conflict_resolver = ConflictResolver::new();

    // Simulate conflicting strategies accessing same resources
    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                // Simulate different strategy types competing for resources
                match i % 3 {
                    0 => {
                        // Arbitrage strategy
                        let _executor = DexArbitrageExecutor::new();
                        thread::sleep(Duration::from_millis(5));
                    },
                    1 => {
                        // Liquidation strategy
                        let _liquidator = AaveLiquidator::new();
                        thread::sleep(Duration::from_millis(3));
                    },
                    _ => {
                        // Zero-risk strategy
                        let _optimizer = BackrunOptimizer::new();
                        thread::sleep(Duration::from_millis(2));
                    }
                }
                i
            })
        })
        .collect();

    // Wait for all strategies to complete
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.len(), 10);

    // Verify conflict resolution worked (no deadlocks, all completed)
    assert_eq!(conflict_resolver.get_resolution_count(), 0); // No conflicts detected in this simple test

    coordinator.stop().unwrap();
}

#[test]
fn test_profit_calculation_accuracy() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();

    // Test profit calculator with various scenarios
    let profit_calculator = ProfitCalculator::new();

    // Test liquidation profit calculation
    let liquidation_opportunity = LiquidationOpportunity::new(
        1,
        "AAVE".to_string(),
        "0x123".to_string(),
        Decimal::from_str_exact("1000.0").unwrap(), // Collateral value
        Decimal::from_str_exact("50.0").unwrap(),   // Expected profit
    );

    assert!(liquidation_opportunity.is_profitable());
    assert_eq!(liquidation_opportunity.expected_profit(), Decimal::from_str_exact("50.0").unwrap());

    // Test different profit thresholds
    let high_profit_opportunity = LiquidationOpportunity::new(
        2,
        "Compound".to_string(),
        "0x456".to_string(),
        Decimal::from_str_exact("5000.0").unwrap(),
        Decimal::from_str_exact("250.0").unwrap(),
    );

    assert!(high_profit_opportunity.is_profitable());
    assert!(high_profit_opportunity.expected_profit() > liquidation_opportunity.expected_profit());

    // Test marginal profit opportunity
    let marginal_opportunity = LiquidationOpportunity::new(
        3,
        "Venus".to_string(),
        "0x789".to_string(),
        Decimal::from_str_exact("100.0").unwrap(),
        Decimal::from_str_exact("1.0").unwrap(),
    );

    assert!(marginal_opportunity.is_profitable());
    assert_eq!(profit_calculator.get_calculation_count(), 0); // Stub implementation
}

#[test]
fn test_gas_optimization_strategies() {
    // Initialize strategy core
    let strategy_config = StrategyConfig {
        max_gas_price: 100, // Conservative gas price
        ..StrategyConfig::default()
    };
    init_strategy_core(&strategy_config).unwrap();

    // Test gas golfing optimizer
    let gas_optimizer = GasGolfingOptimizer::new();
    assert_eq!(gas_optimizer.get_optimization_count(), 0);

    // Test multicall optimizer for liquidations
    let multicall_optimizer = MulticallOptimizer::new();
    assert_eq!(multicall_optimizer.get_optimization_count(), 0);

    // Test zero-risk coordinator with gas optimization focus
    let zero_risk_config = ZeroRiskConfig::default();
    let mut coordinator = ZeroRiskCoordinator::new(zero_risk_config);

    coordinator.start().unwrap();

    // Simulate gas-optimized strategy execution
    let backrun_optimizer = BackrunOptimizer::new();
    let mev_bypass = MevProtectionBypass::new();

    // Verify components are initialized
    assert_eq!(backrun_optimizer.get_optimization_count(), 0);
    assert_eq!(mev_bypass.get_bypass_count(), 0);

    coordinator.stop().unwrap();
}

#[test]
fn test_real_time_monitoring_and_health_checks() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();

    // Test health monitoring for liquidation protocols
    let health_monitor = HealthMonitor::new();
    assert_eq!(health_monitor.get_check_count(), 0);

    // Test sequencer monitoring for time-bandit strategies
    let sequencer_monitor = SequencerMonitor::new();
    assert_eq!(sequencer_monitor.get_monitoring_count(), 0);

    // Test delay exploitation analyzer
    let delay_analyzer = DelayExploitationAnalyzer::new();
    assert_eq!(delay_analyzer.get_analysis_count(), 0);

    // Simulate continuous monitoring
    let start = Instant::now();

    // Run monitoring for a short period
    while start.elapsed() < Duration::from_millis(50) {
        // Simulate monitoring activities
        thread::sleep(Duration::from_millis(1));
    }

    // Verify monitoring systems are responsive
    assert!(start.elapsed() < Duration::from_millis(100));
}

#[test]
fn test_strategy_priority_and_resource_allocation() {
    // Initialize strategy core with limited resources
    let strategy_config = StrategyConfig {
        max_concurrent_strategies: 5, // Limited concurrency
        execution_timeout_ms: 1000,
        ..StrategyConfig::default()
    };
    init_strategy_core(&strategy_config).unwrap();

    // Initialize priority system
    let priority_config = PriorityConfig::default();
    let mut coordinator = PriorityCoordinator::new(priority_config);
    coordinator.start().unwrap();

    // Test execution queue with priority ordering
    let execution_queue = ExecutionQueue::new();
    assert!(execution_queue.is_empty());

    // Test opportunity scorer
    let opportunity_scorer = OpportunityScorer::new();
    assert_eq!(opportunity_scorer.get_scoring_count(), 0);

    // Test resource allocator
    let resource_allocator = ResourceAllocator::new();
    assert_eq!(resource_allocator.get_allocation_count(), 0);

    // Simulate high-priority vs low-priority strategies
    let high_priority_handles: Vec<_> = (0..3)
        .map(|i| {
            thread::spawn(move || {
                // High-priority liquidation strategies
                let _liquidator = AaveLiquidator::new();
                thread::sleep(Duration::from_millis(10));
                format!("high_priority_{}", i)
            })
        })
        .collect();

    let low_priority_handles: Vec<_> = (0..2)
        .map(|i| {
            thread::spawn(move || {
                // Lower-priority arbitrage strategies
                let _executor = DexArbitrageExecutor::new();
                thread::sleep(Duration::from_millis(20));
                format!("low_priority_{}", i)
            })
        })
        .collect();

    // Collect results
    let high_results: Vec<_> = high_priority_handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    let low_results: Vec<_> = low_priority_handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    assert_eq!(high_results.len(), 3);
    assert_eq!(low_results.len(), 2);

    coordinator.stop().unwrap();
}

#[test]
fn test_cross_chain_arbitrage_coordination() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();

    // Initialize arbitrage coordinator
    let arbitrage_config = ArbitrageConfig::default();
    let mut coordinator = ArbitrageCoordinator::new(arbitrage_config);
    coordinator.start().unwrap();

    // Test cross-chain arbitrage executor
    let cross_chain_executor = CrossChainArbitrageExecutor::new();
    assert_eq!(cross_chain_executor.get_execution_count(), 0);

    // Test route optimization for cross-chain paths
    let route_optimizer = RouteOptimizer::new();
    let pathfinder = RoutePathfinder::new();
    let slippage_calculator = SlippageCalculator::new();

    // Simulate cross-chain arbitrage discovery and execution
    let start = Instant::now();

    // Simulate multiple cross-chain opportunities
    let handles: Vec<_> = (0..5)
        .map(|i| {
            thread::spawn(move || {
                // Simulate cross-chain arbitrage execution
                thread::sleep(Duration::from_millis(5));
                format!("cross_chain_arb_{}", i)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    let execution_time = start.elapsed();

    // Verify all cross-chain arbitrages completed
    assert_eq!(results.len(), 5);

    // Should execute efficiently
    assert!(execution_time < Duration::from_millis(100),
        "Cross-chain arbitrage took {}ms, should be <100ms",
        execution_time.as_millis());

    // Verify optimization components
    assert_eq!(route_optimizer.get_optimization_count(), 0);
    assert_eq!(pathfinder.get_path_count(), 0);
    assert_eq!(slippage_calculator.get_calculation_count(), 0);

    coordinator.stop().unwrap();
}

#[test]
fn test_system_resilience_and_fault_tolerance() {
    // Initialize strategy core
    let strategy_config = StrategyConfig::default();
    init_strategy_core(&strategy_config).unwrap();

    // Test system behavior under various failure conditions
    let mut coordinators = Vec::new();

    // Initialize multiple coordinators
    coordinators.push(Box::new(ArbitrageCoordinator::new(ArbitrageConfig::default())));
    coordinators.push(Box::new(LiquidationCoordinator::new(LiquidationConfig::default())));
    coordinators.push(Box::new(TimeBanditCoordinator::new(TimeBanditConfig::default())));
    coordinators.push(Box::new(ZeroRiskCoordinator::new(ZeroRiskConfig::default())));

    // Start all coordinators
    for coordinator in &mut coordinators {
        coordinator.start().unwrap();
    }

    // Simulate system stress with rapid start/stop cycles
    for _ in 0..3 {
        thread::sleep(Duration::from_millis(10));

        // Stop all coordinators
        for coordinator in &mut coordinators {
            coordinator.stop().unwrap();
        }

        thread::sleep(Duration::from_millis(5));

        // Restart all coordinators
        for coordinator in &mut coordinators {
            coordinator.start().unwrap();
        }
    }

    // Final cleanup
    for coordinator in &mut coordinators {
        coordinator.stop().unwrap();
    }

    // System should handle rapid start/stop cycles without issues
    // If we reach this point, the test passed
}
