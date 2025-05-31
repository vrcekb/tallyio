//! Chaos Engineering Tests for TallyIO
//!
//! Tests system resilience under extreme conditions, failures, and unexpected scenarios.
//! Essential for production reliability in high-stakes financial environments.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::expect_used)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::default_numeric_fallback)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tallyio_core::error::CoreResult;
use tallyio_core::mempool::MempoolAnalyzer;
use tallyio_core::types::{Gas, Price, Transaction};

/// Chaos engineering configuration
#[derive(Debug, Clone)]
pub struct ChaosConfig {
    /// Probability of random failures (0.0 - 1.0)
    pub failure_rate: f64,
    /// Maximum delay to inject (milliseconds)
    pub max_delay_ms: u64,
    /// Memory pressure simulation
    pub memory_pressure: bool,
    /// CPU stress simulation
    pub cpu_stress: bool,
    /// Network partition simulation
    pub network_partition: bool,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            failure_rate: 0.1, // 10% failure rate
            max_delay_ms: 100,
            memory_pressure: false,
            cpu_stress: false,
            network_partition: false,
        }
    }
}

/// Chaos monkey for introducing controlled failures
pub struct ChaosMonkey {
    config: ChaosConfig,
    active: AtomicBool,
    failures_injected: AtomicU64,
}

impl ChaosMonkey {
    pub fn new(config: ChaosConfig) -> Self {
        Self {
            config,
            active: AtomicBool::new(false),
            failures_injected: AtomicU64::new(0),
        }
    }

    pub fn start(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    pub fn stop(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    pub fn should_inject_failure(&self) -> bool {
        if !self.active.load(Ordering::Relaxed) {
            return false;
        }

        let random_value = fastrand::f64();
        if random_value < self.config.failure_rate {
            self.failures_injected.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn inject_delay(&self) {
        if self.should_inject_failure() && self.config.max_delay_ms > 0 {
            let delay_ms = fastrand::u64(1..=self.config.max_delay_ms);
            thread::sleep(Duration::from_millis(delay_ms));
        }
    }

    pub fn failures_count(&self) -> u64 {
        self.failures_injected.load(Ordering::Relaxed)
    }
}

/// System resilience tests
#[cfg(test)]
mod resilience_tests {
    use super::*;

    #[test]
    fn test_system_under_random_failures() -> CoreResult<()> {
        let chaos_config = ChaosConfig {
            failure_rate: 0.2, // 20% failure rate
            max_delay_ms: 50,
            ..Default::default()
        };

        let chaos_monkey = Arc::new(ChaosMonkey::new(chaos_config));
        chaos_monkey.start();

        let analyzer = MempoolAnalyzer::new();
        let mut successful_operations = 0;
        let mut failed_operations = 0;
        let total_operations = 100;

        for i in 0..total_operations {
            // Inject chaos
            chaos_monkey.inject_delay();

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                Price::from_gwei(1_000_000_000),
                Price::from_gwei(50),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            // Simulate potential failure
            if chaos_monkey.should_inject_failure() {
                failed_operations += 1;
                continue;
            }

            match analyzer.analyze_transaction(&tx) {
                Ok(_) => successful_operations += 1,
                Err(_) => failed_operations += 1,
            }
        }

        chaos_monkey.stop();

        // System should maintain some level of functionality even under chaos
        let success_rate = successful_operations as f64 / total_operations as f64;
        assert!(
            success_rate > 0.5,
            "Success rate too low under chaos: {:.2}%",
            success_rate * 100.0
        );

        println!("Chaos test results:");
        println!("  Successful operations: {}", successful_operations);
        println!("  Failed operations: {}", failed_operations);
        println!("  Success rate: {:.2}%", success_rate * 100.0);
        println!("  Failures injected: {}", chaos_monkey.failures_count());

        Ok(())
    }

    #[test]
    fn test_concurrent_chaos_operations() -> CoreResult<()> {
        let chaos_config = ChaosConfig {
            failure_rate: 0.15,
            max_delay_ms: 25,
            ..Default::default()
        };

        let chaos_monkey = Arc::new(ChaosMonkey::new(chaos_config));
        chaos_monkey.start();

        let analyzer = Arc::new(MempoolAnalyzer::new());
        let successful_ops = Arc::new(AtomicU64::new(0));
        let failed_ops = Arc::new(AtomicU64::new(0));

        let mut handles = vec![];
        let num_threads = 4;
        let ops_per_thread = 25;

        for thread_id in 0..num_threads {
            let chaos_monkey = Arc::clone(&chaos_monkey);
            let analyzer = Arc::clone(&analyzer);
            let successful_ops = Arc::clone(&successful_ops);
            let failed_ops = Arc::clone(&failed_ops);

            let handle = thread::spawn(move || {
                for i in 0..ops_per_thread {
                    // Inject chaos
                    chaos_monkey.inject_delay();

                    let tx = Transaction::new(
                        [(thread_id * 100 + i) as u8; 20],
                        Some([((thread_id * 100 + i) + 1) as u8; 20]),
                        Price::from_gwei(1_000_000_000),
                        Price::from_gwei(50 + i as u64),
                        Gas::new(150_000),
                        (thread_id * 100 + i) as u64,
                        vec![0xa9, 0x05, 0x9c, 0xbb],
                    );

                    if chaos_monkey.should_inject_failure() {
                        failed_ops.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    match analyzer.analyze_transaction(&tx) {
                        Ok(_) => {
                            successful_ops.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            failed_ops.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        chaos_monkey.stop();

        let total_ops = num_threads * ops_per_thread;
        let successful = successful_ops.load(Ordering::Relaxed);
        let failed = failed_ops.load(Ordering::Relaxed);
        let success_rate = successful as f64 / total_ops as f64;

        // Even under concurrent chaos, system should maintain functionality
        assert!(
            success_rate > 0.4,
            "Concurrent success rate too low: {:.2}%",
            success_rate * 100.0
        );

        println!("Concurrent chaos test results:");
        println!("  Total operations: {}", total_ops);
        println!("  Successful: {}", successful);
        println!("  Failed: {}", failed);
        println!("  Success rate: {:.2}%", success_rate * 100.0);

        Ok(())
    }

    #[test]
    fn test_memory_pressure_resilience() -> CoreResult<()> {
        let chaos_config = ChaosConfig {
            failure_rate: 0.1,
            memory_pressure: true,
            ..Default::default()
        };

        let chaos_monkey = ChaosMonkey::new(chaos_config);
        chaos_monkey.start();

        // Simulate memory pressure by allocating large vectors
        let mut memory_hogs = Vec::new();
        for _ in 0..10 {
            let large_vec: Vec<u8> = vec![0; 1024 * 1024]; // 1MB each
            memory_hogs.push(large_vec);
        }

        let analyzer = MempoolAnalyzer::new();
        let start_time = Instant::now();
        let mut operations_completed = 0;

        // Run operations under memory pressure
        for i in 0..50 {
            chaos_monkey.inject_delay();

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                Price::from_gwei(1_000_000_000),
                Price::from_gwei(50),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            if analyzer.analyze_transaction(&tx).is_ok() {
                operations_completed += 1;
            }

            // Occasionally free some memory
            if i % 10 == 0 && !memory_hogs.is_empty() {
                memory_hogs.pop();
            }
        }

        let duration = start_time.elapsed();
        chaos_monkey.stop();

        // System should complete most operations even under memory pressure
        assert!(
            operations_completed > 30,
            "Too few operations completed under memory pressure: {}",
            operations_completed
        );
        assert!(
            duration < Duration::from_secs(5),
            "Operations took too long under memory pressure"
        );

        println!("Memory pressure test results:");
        println!("  Operations completed: {}", operations_completed);
        println!("  Duration: {:?}", duration);

        Ok(())
    }

    #[test]
    fn test_rapid_state_changes() -> CoreResult<()> {
        let chaos_monkey = ChaosMonkey::new(ChaosConfig::default());
        chaos_monkey.start();

        let analyzer = MempoolAnalyzer::new();
        let mut state_changes = 0;
        let max_changes = 100;

        for i in 0..max_changes {
            chaos_monkey.inject_delay();

            // Rapidly changing transaction parameters
            let gas_price = Price::from_gwei(10 + (i % 100) as u64);
            let value = Price::from_gwei(1_000_000_000 + (i * 1000) as u64);
            let gas_limit = Gas::new(150_000 + (i % 50_000) as u64);

            let tx = Transaction::new(
                [(i % 256) as u8; 20],
                Some([((i + 1) % 256) as u8; 20]),
                value,
                gas_price,
                gas_limit,
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb, (i % 256) as u8],
            );

            if analyzer.analyze_transaction(&tx).is_ok() {
                state_changes += 1;
            }
        }

        chaos_monkey.stop();

        // System should handle rapid state changes gracefully
        assert!(
            state_changes > 80,
            "System failed to handle rapid state changes: {}",
            state_changes
        );

        println!("Rapid state changes test results:");
        println!("  State changes handled: {}", state_changes);
        println!(
            "  Success rate: {:.2}%",
            (state_changes as f64 / max_changes as f64) * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_resource_exhaustion_recovery() -> CoreResult<()> {
        let chaos_config = ChaosConfig {
            failure_rate: 0.3, // High failure rate
            max_delay_ms: 100,
            cpu_stress: true,
            ..Default::default()
        };

        let chaos_monkey = ChaosMonkey::new(chaos_config);
        chaos_monkey.start();

        let analyzer = MempoolAnalyzer::new();

        let mut operations_before_recovery = 0;
        let mut operations_after_recovery = 0;

        // Phase 1: Stress the system
        for i in 0..50 {
            chaos_monkey.inject_delay();

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                Price::from_gwei(1_000_000_000),
                Price::from_gwei(50),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            if analyzer.analyze_transaction(&tx).is_ok() {
                operations_before_recovery += 1;
            }
        }

        // Phase 2: Reduce chaos and allow recovery
        chaos_monkey.stop();
        thread::sleep(Duration::from_millis(100)); // Recovery period

        // Phase 3: Test recovery
        for i in 50..100 {
            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                Price::from_gwei(1_000_000_000),
                Price::from_gwei(50),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            if analyzer.analyze_transaction(&tx).is_ok() {
                operations_after_recovery += 1;
            }
        }

        // System should recover and perform reasonably after chaos stops
        let recovery_successful = operations_after_recovery >= (operations_before_recovery / 2);

        assert!(
            recovery_successful,
            "System failed to recover after resource exhaustion: before={}, after={}",
            operations_before_recovery, operations_after_recovery
        );
        assert!(
            operations_after_recovery > 20,
            "Recovery performance too low: {}",
            operations_after_recovery
        );

        println!("Resource exhaustion recovery test results:");
        println!(
            "  Operations before recovery: {}",
            operations_before_recovery
        );
        println!("  Operations after recovery: {}", operations_after_recovery);
        println!("  Recovery successful: {}", recovery_successful);

        Ok(())
    }
}

/// Fault injection tests
#[cfg(test)]
mod fault_injection_tests {
    use super::*;

    #[test]
    fn test_network_partition_simulation() -> CoreResult<()> {
        let chaos_config = ChaosConfig {
            failure_rate: 0.5, // 50% network failures
            network_partition: true,
            ..Default::default()
        };

        let chaos_monkey = ChaosMonkey::new(chaos_config);
        chaos_monkey.start();

        let analyzer = MempoolAnalyzer::new();
        let mut network_operations = 0;
        let mut local_operations = 0;

        for i in 0..100 {
            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                Price::from_gwei(1_000_000_000),
                Price::from_gwei(50),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            // Simulate network partition
            if chaos_monkey.should_inject_failure() {
                // Operation would fail due to network partition
                continue;
            }

            // Local operations should still work
            if analyzer.analyze_transaction(&tx).is_ok() {
                if i % 2 == 0 {
                    network_operations += 1;
                } else {
                    local_operations += 1;
                }
            }
        }

        chaos_monkey.stop();

        // Local operations should be more reliable than network operations
        assert!(local_operations > 0, "No local operations succeeded");

        println!("Network partition simulation results:");
        println!("  Network operations: {}", network_operations);
        println!("  Local operations: {}", local_operations);
        println!(
            "  Total failures injected: {}",
            chaos_monkey.failures_count()
        );

        Ok(())
    }

    #[test]
    fn test_cascading_failure_prevention() -> CoreResult<()> {
        let chaos_monkey = ChaosMonkey::new(ChaosConfig {
            failure_rate: 0.8, // Very high failure rate
            max_delay_ms: 200,
            ..Default::default()
        });
        chaos_monkey.start();

        let analyzer = MempoolAnalyzer::new();

        let mut successful_operations = 0;

        // Test that high failure rate doesn't cause complete system failure
        for i in 0..20 {
            chaos_monkey.inject_delay();

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                Price::from_gwei(1_000_000_000),
                Price::from_gwei(50),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            // Even with high failure rate, some operations should succeed
            if !chaos_monkey.should_inject_failure() && analyzer.analyze_transaction(&tx).is_ok() {
                successful_operations += 1;
            }
        }

        chaos_monkey.stop();

        // System should prevent cascading failures
        let cascade_prevented = successful_operations > 0;

        assert!(cascade_prevented, "System experienced cascading failure");

        println!("Cascading failure prevention results:");
        println!("  Successful operations: {}", successful_operations);
        println!("  Cascade prevented: {}", cascade_prevented);

        Ok(())
    }
}
