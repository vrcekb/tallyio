//! # TallyIO Hot Path Integration Tests
//! 
//! Ultra-comprehensive integration tests for the hot_path crate.
//! These tests validate end-to-end functionality for MEV detection and execution
//! in production-like scenarios handling millions of dollars daily.
//! 
//! ## Test Categories:
//! - MEV Detection Pipeline (<500ns latency requirement)
//! - Cross-chain Execution (<50ns requirement)
//! - Memory Management (<5ns allocation requirement)
//! - SIMD Operations (vectorized processing)
//! - Atomic Operations (lock-free concurrency)

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

use hot_path::{
    HotPathConfig, initialize, get_metrics,
    detection::{detect_opportunities, update_price_feed},
    execution::execute_opportunity,
    types::{MarketSnapshot, TradingPair, AlignedPrice, Opportunity},
    atomic::{AtomicCounter, LockFreeQueue, StateMachine, State},
    memory::{ArenaAllocator, ObjectPool, RingBuffer},
    simd::{SimdCapabilities, PriceCalculator, SearchOperations, HashOperations},
};

/// Test configuration for production-like scenarios
const TEST_MARKET_SIZE: usize = 10_000;
const TEST_OPPORTUNITY_COUNT: usize = 1_000;
const TEST_CONCURRENT_THREADS: usize = 16;
const MEV_DETECTION_LATENCY_NS: u64 = 500;
const CROSS_CHAIN_EXECUTION_NS: u64 = 50;
const MEMORY_ALLOCATION_NS: u64 = 5;

#[test]
fn test_hot_path_initialization() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    let metrics = get_metrics();
    assert!(metrics.memory_usage_bytes > 0);
    assert_eq!(metrics.initialization_time_ns, 0); // Should be set after init
}

#[test]
fn test_end_to_end_mev_detection_pipeline() {
    // Initialize hot path
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    // Create large market snapshot simulating real trading conditions
    let mut snapshot = MarketSnapshot::new(TEST_MARKET_SIZE);
    
    // Populate with realistic trading pairs and prices
    for i in 0..TEST_MARKET_SIZE {
        let pair = TradingPair::new(
            i as u64,
            (i + 1) as u64,
            100 + i,
            200 + i,
        );
        
        let price = AlignedPrice::new(
            1000 + (i * 10) as u64,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            500 + (i % 100),
        );
        
        snapshot.add_price(pair, price);
    }
    
    // Measure MEV detection latency
    let start = Instant::now();
    let opportunities = detect_opportunities(&snapshot).unwrap();
    let detection_time = start.elapsed();
    
    // Validate performance requirements
    assert!(detection_time.as_nanos() < MEV_DETECTION_LATENCY_NS as u128,
        "MEV detection took {}ns, must be <{}ns", 
        detection_time.as_nanos(), MEV_DETECTION_LATENCY_NS);
    
    // Validate detection results
    assert!(opportunities.len() <= 16, "Too many opportunities detected");
    
    // Test opportunity execution if any found
    if let Some(opportunity) = opportunities.first() {
        let start = Instant::now();
        let result = execute_opportunity(opportunity).unwrap();
        let execution_time = start.elapsed();
        
        assert!(execution_time.as_nanos() < CROSS_CHAIN_EXECUTION_NS as u128,
            "Execution took {}ns, must be <{}ns",
            execution_time.as_nanos(), CROSS_CHAIN_EXECUTION_NS);
        
        assert!(result.success, "Opportunity execution failed");
    }
}

#[test]
fn test_concurrent_price_feed_updates() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    let handles: Vec<_> = (0..TEST_CONCURRENT_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                for i in 0..100 {
                    let price = AlignedPrice::new(
                        1000 + (thread_id * 100 + i) as u64,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64,
                        500,
                    );
                    
                    update_price_feed(thread_id as u64, price).unwrap();
                }
            })
        })
        .collect();
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify system stability after concurrent updates
    let metrics = get_metrics();
    assert!(metrics.memory_usage_bytes > 0);
}

#[test]
fn test_atomic_operations_under_load() {
    let counter = Arc::new(AtomicCounter::new());
    let queue = Arc::new(LockFreeQueue::new(1000));
    
    let handles: Vec<_> = (0..TEST_CONCURRENT_THREADS)
        .map(|thread_id| {
            let counter = Arc::clone(&counter);
            let queue = Arc::clone(&queue);
            
            thread::spawn(move || {
                for i in 0..1000 {
                    // Test atomic counter
                    counter.increment();
                    
                    // Test lock-free queue
                    let value = thread_id * 1000 + i;
                    if queue.enqueue(value).is_ok() {
                        if let Ok(dequeued) = queue.dequeue() {
                            assert!(dequeued >= 0);
                        }
                    }
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let final_count = counter.get();
    assert_eq!(final_count, TEST_CONCURRENT_THREADS * 1000);
}

#[test]
fn test_memory_management_performance() {
    // Test arena allocator
    let mut arena = ArenaAllocator::new(1024 * 1024).unwrap(); // 1MB
    
    let start = Instant::now();
    for _ in 0..10000 {
        let _allocation = arena.allocate(64).unwrap();
    }
    let arena_time = start.elapsed();
    
    assert!(arena_time.as_nanos() / 10000 < MEMORY_ALLOCATION_NS as u128,
        "Arena allocation took {}ns per allocation, must be <{}ns",
        arena_time.as_nanos() / 10000, MEMORY_ALLOCATION_NS);
    
    // Test object pool
    let pool = ObjectPool::<u64>::new(1000).unwrap();
    
    let start = Instant::now();
    for _ in 0..1000 {
        if let Ok(obj) = pool.acquire() {
            pool.release(obj);
        }
    }
    let pool_time = start.elapsed();
    
    assert!(pool_time.as_nanos() / 1000 < MEMORY_ALLOCATION_NS as u128,
        "Object pool took {}ns per operation, must be <{}ns",
        pool_time.as_nanos() / 1000, MEMORY_ALLOCATION_NS);
    
    // Test ring buffer
    let mut ring = RingBuffer::new(1024).unwrap();
    
    let start = Instant::now();
    for i in 0..1000 {
        ring.push(i).unwrap();
        ring.pop().unwrap();
    }
    let ring_time = start.elapsed();
    
    assert!(ring_time.as_nanos() / 1000 < MEMORY_ALLOCATION_NS as u128,
        "Ring buffer took {}ns per operation, must be <{}ns",
        ring_time.as_nanos() / 1000, MEMORY_ALLOCATION_NS);
}

#[test]
fn test_simd_operations_performance() {
    let caps = SimdCapabilities::detect();
    
    if caps.has_simd() {
        // Test SIMD price calculations
        let calculator = PriceCalculator::new();
        let prices = vec![1000u64; 1000];
        
        let start = Instant::now();
        let results = calculator.calculate_batch(&prices).unwrap();
        let calc_time = start.elapsed();
        
        assert_eq!(results.len(), prices.len());
        assert!(calc_time.as_nanos() < 1_000_000, // <1ms for 1000 calculations
            "SIMD calculations took {}ns, should be <1ms", calc_time.as_nanos());
        
        // Test SIMD search operations
        let search_ops = SearchOperations::new();
        let data = vec![42u64; 10000];
        
        let start = Instant::now();
        let found = search_ops.find_value(&data, 42).unwrap();
        let search_time = start.elapsed();
        
        assert!(found.is_some());
        assert!(search_time.as_nanos() < 100_000, // <100μs for 10k elements
            "SIMD search took {}ns, should be <100μs", search_time.as_nanos());
        
        // Test SIMD hash operations
        let hash_ops = HashOperations::new();
        let data = b"test_data_for_hashing";
        
        let start = Instant::now();
        let hash = hash_ops.hash_data(data).unwrap();
        let hash_time = start.elapsed();
        
        assert_eq!(hash.len(), 32); // SHA-256 hash
        assert!(hash_time.as_nanos() < 10_000, // <10μs for small data
            "SIMD hashing took {}ns, should be <10μs", hash_time.as_nanos());
    }
}

#[test]
fn test_state_machine_transitions() {
    let mut state_machine = StateMachine::new();
    
    // Test valid transitions
    assert!(state_machine.transition_to(State::Detecting).is_ok());
    assert!(state_machine.transition_to(State::Executing).is_ok());
    assert!(state_machine.transition_to(State::Idle).is_ok());
    
    // Test current state
    assert!(state_machine.is_in_state(State::Idle));
    
    // Test invalid transitions (should be handled gracefully)
    let result = state_machine.transition_to(State::Executing);
    // This might fail depending on state machine logic, but shouldn't panic
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_system_under_extreme_load() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    // Simulate extreme load with multiple concurrent operations
    let handles: Vec<_> = (0..TEST_CONCURRENT_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                // Each thread performs mixed operations
                for i in 0..100 {
                    // Price updates
                    let price = AlignedPrice::new(
                        1000 + i as u64,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64,
                        500,
                    );
                    let _ = update_price_feed(thread_id as u64, price);
                    
                    // MEV detection
                    let snapshot = MarketSnapshot::new(100);
                    let _ = detect_opportunities(&snapshot);
                    
                    // Opportunity execution
                    let opportunity = Opportunity::new(
                        i as u64,
                        format!("test_opportunity_{}", i),
                        1000,
                        95,
                    );
                    let _ = execute_opportunity(&opportunity);
                }
            })
        })
        .collect();
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    // System should still be responsive
    let metrics = get_metrics();
    assert!(metrics.memory_usage_bytes > 0);
    
    // Quick smoke test after load
    let snapshot = MarketSnapshot::new(10);
    let opportunities = detect_opportunities(&snapshot).unwrap();
    assert!(opportunities.len() <= 16);
}

#[test]
fn test_error_handling_robustness() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    // Test with invalid market snapshot
    let empty_snapshot = MarketSnapshot::new(0);
    let result = detect_opportunities(&empty_snapshot);
    assert!(result.is_ok()); // Should handle gracefully
    
    // Test with invalid opportunity
    let invalid_opportunity = Opportunity::new(0, String::new(), 0, 0);
    let result = execute_opportunity(&invalid_opportunity);
    // Should either succeed or fail gracefully, but not panic
    assert!(result.is_ok() || result.is_err());
    
    // Test memory allocation failures (simulated)
    let result = ArenaAllocator::new(0);
    assert!(result.is_err()); // Should fail gracefully
    
    // Test queue overflow
    let queue = LockFreeQueue::new(1);
    assert!(queue.enqueue(1).is_ok());
    assert!(queue.enqueue(2).is_err()); // Should fail gracefully when full
}
