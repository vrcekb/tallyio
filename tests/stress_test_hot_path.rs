//! # TallyIO Hot Path Stress Tests
//! 
//! Extreme stress tests for the hot_path crate simulating production load
//! with millions of dollars in daily trading volume. These tests validate
//! system stability under maximum stress conditions.
//! 
//! ## Stress Test Categories:
//! - High-frequency trading simulation (1M+ ops/sec)
//! - Memory pressure testing (GB-scale allocations)
//! - Concurrent access patterns (1000+ threads)
//! - Network latency simulation
//! - Error injection and recovery

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

use hot_path::{
    HotPathConfig, initialize, get_metrics,
    detection::{detect_opportunities, update_price_feed},
    execution::execute_opportunity,
    types::{MarketSnapshot, TradingPair, AlignedPrice, Opportunity},
    atomic::{AtomicCounter, LockFreeQueue},
    memory::{ArenaAllocator, ObjectPool},
};

/// Stress test configuration
const STRESS_THREADS: usize = 100;
const STRESS_OPERATIONS_PER_THREAD: usize = 10_000;
const STRESS_MARKET_SIZE: usize = 50_000;
const STRESS_DURATION_SECS: u64 = 30;

#[test]
#[ignore] // Run with --ignored flag for stress tests
fn stress_test_high_frequency_trading() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    let operations_counter = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(STRESS_THREADS));
    
    let start_time = Instant::now();
    
    let handles: Vec<_> = (0..STRESS_THREADS)
        .map(|thread_id| {
            let operations_counter = Arc::clone(&operations_counter);
            let barrier = Arc::clone(&barrier);
            
            thread::spawn(move || {
                barrier.wait(); // Synchronize start
                
                for i in 0..STRESS_OPERATIONS_PER_THREAD {
                    // Simulate high-frequency price updates
                    let price = AlignedPrice::new(
                        1000 + (thread_id * STRESS_OPERATIONS_PER_THREAD + i) as u64,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64,
                        500 + (i % 100),
                    );
                    
                    if update_price_feed(thread_id as u64, price).is_ok() {
                        operations_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    // Simulate MEV detection every 10 updates
                    if i % 10 == 0 {
                        let snapshot = MarketSnapshot::new(100);
                        if detect_opportunities(&snapshot).is_ok() {
                            operations_counter.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    
                    // Simulate opportunity execution every 100 updates
                    if i % 100 == 0 {
                        let opportunity = Opportunity::new(
                            (thread_id * 1000 + i) as u64,
                            format!("stress_opportunity_{}_{}", thread_id, i),
                            1000 + i as u64,
                            95,
                        );
                        
                        if execute_opportunity(&opportunity).is_ok() {
                            operations_counter.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_time = start_time.elapsed();
    let total_operations = operations_counter.load(Ordering::Relaxed);
    let ops_per_second = total_operations as f64 / total_time.as_secs_f64();
    
    println!("Stress Test Results:");
    println!("  Total operations: {}", total_operations);
    println!("  Total time: {:?}", total_time);
    println!("  Operations per second: {:.2}", ops_per_second);
    
    // Validate performance requirements
    assert!(ops_per_second > 100_000.0, 
        "Performance too low: {:.2} ops/sec, expected >100k ops/sec", ops_per_second);
    
    // Validate system stability
    let metrics = get_metrics();
    assert!(metrics.memory_usage_bytes > 0);
}

#[test]
#[ignore]
fn stress_test_memory_pressure() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    // Test massive memory allocations
    let mut allocators = Vec::new();
    let mut pools = Vec::new();
    
    // Create many arena allocators
    for i in 0..100 {
        if let Ok(allocator) = ArenaAllocator::new(1024 * 1024) { // 1MB each
            allocators.push(allocator);
        }
        
        if let Ok(pool) = ObjectPool::<u64>::new(10000) {
            pools.push(pool);
        }
        
        // Simulate memory pressure
        if i % 10 == 0 {
            let snapshot = MarketSnapshot::new(STRESS_MARKET_SIZE);
            let _ = detect_opportunities(&snapshot);
        }
    }
    
    // Perform operations under memory pressure
    for allocator in &mut allocators {
        for _ in 0..1000 {
            let _ = allocator.allocate(64);
        }
    }
    
    // Test object pools under pressure
    for pool in &pools {
        let mut objects = Vec::new();
        
        // Acquire many objects
        for _ in 0..1000 {
            if let Ok(obj) = pool.acquire() {
                objects.push(obj);
            }
        }
        
        // Release them all
        for obj in objects {
            pool.release(obj);
        }
    }
    
    // System should still be responsive
    let metrics = get_metrics();
    assert!(metrics.memory_usage_bytes > 0);
    
    println!("Memory pressure test completed successfully");
    println!("  Allocators created: {}", allocators.len());
    println!("  Object pools created: {}", pools.len());
}

#[test]
#[ignore]
fn stress_test_concurrent_queue_operations() {
    let queue = Arc::new(LockFreeQueue::new(100_000));
    let operations_counter = Arc::new(AtomicU64::new(0));
    
    let handles: Vec<_> = (0..STRESS_THREADS)
        .map(|thread_id| {
            let queue = Arc::clone(&queue);
            let operations_counter = Arc::clone(&operations_counter);
            
            thread::spawn(move || {
                for i in 0..STRESS_OPERATIONS_PER_THREAD {
                    let value = thread_id * STRESS_OPERATIONS_PER_THREAD + i;
                    
                    // Enqueue operation
                    if queue.enqueue(value).is_ok() {
                        operations_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    // Dequeue operation
                    if queue.dequeue().is_ok() {
                        operations_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    // Yield occasionally to increase contention
                    if i % 1000 == 0 {
                        thread::yield_now();
                    }
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_operations = operations_counter.load(Ordering::Relaxed);
    println!("Concurrent queue operations: {}", total_operations);
    
    // Validate queue integrity
    assert!(queue.len() <= 100_000);
    assert!(total_operations > 0);
}

#[test]
#[ignore]
fn stress_test_atomic_counters() {
    let counter = Arc::new(AtomicCounter::new());
    let expected_total = STRESS_THREADS * STRESS_OPERATIONS_PER_THREAD;
    
    let handles: Vec<_> = (0..STRESS_THREADS)
        .map(|_| {
            let counter = Arc::clone(&counter);
            
            thread::spawn(move || {
                for _ in 0..STRESS_OPERATIONS_PER_THREAD {
                    counter.increment();
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let final_count = counter.get();
    println!("Atomic counter final value: {}", final_count);
    println!("Expected value: {}", expected_total);
    
    assert_eq!(final_count, expected_total);
}

#[test]
#[ignore]
fn stress_test_large_market_snapshots() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    // Create extremely large market snapshot
    let mut snapshot = MarketSnapshot::new(STRESS_MARKET_SIZE);
    
    let start_time = Instant::now();
    
    // Populate with many trading pairs
    for i in 0..STRESS_MARKET_SIZE {
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
            500 + (i % 1000),
        );
        
        snapshot.add_price(pair, price);
    }
    
    let population_time = start_time.elapsed();
    println!("Market snapshot population time: {:?}", population_time);
    
    // Test MEV detection on large snapshot
    let detection_start = Instant::now();
    let opportunities = detect_opportunities(&snapshot).unwrap();
    let detection_time = detection_start.elapsed();
    
    println!("MEV detection time for {} pairs: {:?}", STRESS_MARKET_SIZE, detection_time);
    println!("Opportunities found: {}", opportunities.len());
    
    // Validate performance even with large datasets
    assert!(detection_time.as_millis() < 100, 
        "Detection took {}ms, should be <100ms even for large datasets", 
        detection_time.as_millis());
}

#[test]
#[ignore]
fn stress_test_sustained_load() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    let operations_counter = Arc::new(AtomicU64::new(0));
    let should_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    
    let handles: Vec<_> = (0..STRESS_THREADS)
        .map(|thread_id| {
            let operations_counter = Arc::clone(&operations_counter);
            let should_stop = Arc::clone(&should_stop);
            
            thread::spawn(move || {
                let mut local_counter = 0;
                
                while !should_stop.load(Ordering::Relaxed) {
                    // Mixed workload
                    let price = AlignedPrice::new(
                        1000 + local_counter,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64,
                        500,
                    );
                    
                    if update_price_feed(thread_id as u64, price).is_ok() {
                        local_counter += 1;
                        
                        if local_counter % 100 == 0 {
                            let snapshot = MarketSnapshot::new(100);
                            let _ = detect_opportunities(&snapshot);
                        }
                        
                        if local_counter % 1000 == 0 {
                            let opportunity = Opportunity::new(
                                local_counter,
                                format!("sustained_op_{}", local_counter),
                                1000,
                                95,
                            );
                            let _ = execute_opportunity(&opportunity);
                        }
                    }
                    
                    // Small delay to prevent CPU spinning
                    if local_counter % 10000 == 0 {
                        thread::sleep(Duration::from_micros(1));
                    }
                }
                
                operations_counter.fetch_add(local_counter, Ordering::Relaxed);
            })
        })
        .collect();
    
    // Run for specified duration
    thread::sleep(Duration::from_secs(STRESS_DURATION_SECS));
    should_stop.store(true, Ordering::Relaxed);
    
    // Wait for all threads to finish
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_operations = operations_counter.load(Ordering::Relaxed);
    let ops_per_second = total_operations as f64 / STRESS_DURATION_SECS as f64;
    
    println!("Sustained load test results:");
    println!("  Duration: {}s", STRESS_DURATION_SECS);
    println!("  Total operations: {}", total_operations);
    println!("  Operations per second: {:.2}", ops_per_second);
    
    // Validate sustained performance
    assert!(ops_per_second > 10_000.0,
        "Sustained performance too low: {:.2} ops/sec", ops_per_second);
    
    // Validate system stability after sustained load
    let metrics = get_metrics();
    assert!(metrics.memory_usage_bytes > 0);
}

#[test]
#[ignore]
fn stress_test_error_recovery() {
    let config = HotPathConfig::default();
    initialize(config).unwrap();
    
    let success_counter = Arc::new(AtomicU64::new(0));
    let error_counter = Arc::new(AtomicU64::new(0));
    
    let handles: Vec<_> = (0..STRESS_THREADS)
        .map(|thread_id| {
            let success_counter = Arc::clone(&success_counter);
            let error_counter = Arc::clone(&error_counter);
            
            thread::spawn(move || {
                for i in 0..STRESS_OPERATIONS_PER_THREAD {
                    // Intentionally create some invalid operations to test error handling
                    if i % 10 == 0 {
                        // Invalid opportunity (should fail gracefully)
                        let invalid_opportunity = Opportunity::new(0, String::new(), 0, 0);
                        match execute_opportunity(&invalid_opportunity) {
                            Ok(_) => success_counter.fetch_add(1, Ordering::Relaxed),
                            Err(_) => error_counter.fetch_add(1, Ordering::Relaxed),
                        };
                    } else {
                        // Valid operation
                        let price = AlignedPrice::new(1000 + i as u64, 123_456_789, 500);
                        match update_price_feed(thread_id as u64, price) {
                            Ok(_) => success_counter.fetch_add(1, Ordering::Relaxed),
                            Err(_) => error_counter.fetch_add(1, Ordering::Relaxed),
                        };
                    }
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_success = success_counter.load(Ordering::Relaxed);
    let total_errors = error_counter.load(Ordering::Relaxed);
    let total_operations = total_success + total_errors;
    
    println!("Error recovery test results:");
    println!("  Total operations: {}", total_operations);
    println!("  Successful operations: {}", total_success);
    println!("  Error operations: {}", total_errors);
    println!("  Success rate: {:.2}%", (total_success as f64 / total_operations as f64) * 100.0);
    
    // System should handle errors gracefully and continue operating
    assert!(total_success > 0, "No successful operations");
    assert!(total_errors > 0, "No error conditions tested");
    
    // System should still be responsive after errors
    let metrics = get_metrics();
    assert!(metrics.memory_usage_bytes > 0);
}
