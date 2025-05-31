//! Integration tests for `TallyIO`
//!
//! Tests cross-crate functionality and end-to-end workflows.
//! Production-ready tests with comprehensive error handling and <1ms latency requirements.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::default_numeric_fallback)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tallyio_core::engine::{Controllable, Monitorable, TallyEngine, TransactionProcessor};
use tallyio_core::error::{CoreError, CoreResult, CriticalError};
use tallyio_core::mempool::{
    MempoolAnalyzer, MempoolEvent, MempoolFilter, MempoolWatcher, TransactionFilter,
};
use tallyio_core::types::{Gas, Price, Transaction};
use tallyio_core::utils::{affinity, hash, memory, time, validation};

/// Test basic transaction processing integration
///
/// Verifies that basic transaction analysis meets <1ms latency requirements
/// and produces valid analysis results.
#[test]
fn test_engine_integration() -> CoreResult<()> {
    let analyzer = MempoolAnalyzer::new();

    // Test basic transaction processing
    let tx = Transaction::new(
        [0x01; 20],                      // from
        Some([0x02; 20]),                // to
        Price::from_gwei(1_000_000_000), // 1 ETH value
        Price::from_gwei(20),            // 20 gwei gas price
        Gas::new(21_000),                // gas limit
        0,                               // nonce
        vec![],                          // data
    );

    let start = Instant::now();
    let result = analyzer.analyze_transaction(&tx)?;
    let elapsed = start.elapsed();

    // Verify processing meets latency requirements
    assert!(
        elapsed.as_millis() < 1,
        "Processing took {elapsed:?}, must be <1ms"
    );
    assert!(
        result.analysis_time_ns < 1_000_000,
        "Analysis time too high: {}ns",
        result.analysis_time_ns
    );

    Ok(())
}

/// Test MEV opportunity detection integration
///
/// Verifies that DeFi transactions are properly analyzed for MEV opportunities
/// and that the analysis completes within latency requirements.
#[test]
fn test_mev_integration() -> CoreResult<()> {
    let analyzer = MempoolAnalyzer::new();

    // Test MEV opportunity detection with DeFi transaction
    let defi_tx = Transaction::new(
        [0x01; 20], // from
        Some([
            0xde, 0xad, 0xbe, 0xef, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        ]), // DeFi contract
        Price::from_gwei(2_000_000_000), // 2 ETH value
        Price::from_gwei(60), // 60 gwei gas price
        Gas::new(100_000), // gas limit
        0,          // nonce
        vec![0xa9, 0x05, 0x9c, 0xbb], // swapExactTokensForTokens selector
    );

    let start = Instant::now();
    let result = analyzer.analyze_transaction(&defi_tx)?;
    let elapsed = start.elapsed();

    // Verify MEV detection and latency
    assert!(
        elapsed.as_millis() < 1,
        "MEV analysis took {elapsed:?}, must be <1ms"
    );
    assert!(
        result.has_mev_opportunity,
        "Should detect MEV opportunity in DeFi transaction"
    );
    assert!(
        result.analysis_time_ns < 1_000_000,
        "MEV analysis time too high: {}ns",
        result.analysis_time_ns
    );

    Ok(())
}

/// Test concurrent transaction processing
///
/// Verifies that the system can handle concurrent transaction analysis
/// while maintaining latency and correctness requirements.
#[test]
fn test_concurrent_processing() -> CoreResult<()> {
    let analyzer = Arc::new(MempoolAnalyzer::new());
    let mut handles = vec![];
    let thread_count = 5;
    let transactions_per_thread = 10;

    // Spawn multiple threads processing transactions
    for thread_id in 0..thread_count {
        let analyzer_clone = Arc::clone(&analyzer);
        let handle = thread::spawn(move || -> CoreResult<()> {
            for i in 0..transactions_per_thread {
                let tx = Transaction::new(
                    [(thread_id * 100 + i) as u8; 20],
                    Some([((thread_id * 100 + i) + 1) as u8; 20]),
                    Price::from_gwei(1_000_000_000),
                    Price::from_gwei(20 + i as u64),
                    Gas::new(21_000),
                    (thread_id * 100 + i) as u64,
                    vec![],
                );

                let start = Instant::now();
                let result = analyzer_clone.analyze_transaction(&tx)?;
                let elapsed = start.elapsed();

                // Verify latency even under concurrent load
                assert!(
                    elapsed.as_millis() < 1,
                    "Concurrent transaction took {elapsed:?}, must be <1ms"
                );
                assert!(
                    result.analysis_time_ns < 1_000_000,
                    "Concurrent analysis time too high: {}ns",
                    result.analysis_time_ns
                );
            }
            Ok(())
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().map_err(|_| {
            tallyio_core::error::CoreError::mempool("Thread panicked during concurrent test")
        })??;
    }

    Ok(())
}

/// Test error handling integration
///
/// Verifies that error types work correctly across module boundaries
/// and that critical errors are properly propagated.
#[test]
fn test_error_integration() -> CoreResult<()> {
    // Test critical error creation and propagation
    let critical_error = CriticalError::LatencyViolation(1500);
    let core_error = CoreError::Critical(critical_error);

    assert!(core_error.is_critical());
    assert!(!core_error.is_fatal());

    // Test error display
    let error_string = format!("{core_error}");
    assert!(error_string.contains("Latency violation"));

    // Test error conversion
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test file");
    let converted_error: CoreError = io_error.into();
    assert!(matches!(converted_error, CoreError::Io(_)));

    // Test error creation methods
    let config_error = CoreError::config("test configuration error");
    assert!(matches!(config_error, CoreError::Config(_)));

    let mempool_error = CoreError::mempool("test mempool error");
    assert!(matches!(mempool_error, CoreError::Mempool(_)));

    Ok(())
}

/// Test utility functions integration
///
/// Verifies that utility functions work correctly and meet performance requirements.
#[test]
#[allow(clippy::disallowed_methods)]
fn test_utils_integration() -> CoreResult<()> {
    // Test CPU affinity setting
    affinity::set_core_affinity(0)?;

    // Test memory allocation
    let ptr = memory::alloc_aligned::<u64>(8, 64)?;
    assert!(!ptr.is_null());

    // Verify alignment
    let addr = ptr as usize;
    assert_eq!(addr % 64, 0);

    // Clean up memory
    unsafe {
        memory::dealloc_aligned(ptr, 8, 64);
    }

    // Test hash utilities
    let test_data = b"integration test data";
    let hash1 = hash::hash_bytes(test_data);
    let hash2 = hash::hash_bytes(test_data);
    assert_eq!(hash1, hash2);

    let different_data = b"different test data";
    let hash3 = hash::hash_bytes(different_data);
    assert_ne!(hash1, hash3);

    // Test fast hash
    let string_hash1 = hash::fast_hash(&"test string");
    let string_hash2 = hash::fast_hash(&"test string");
    assert_eq!(string_hash1, string_hash2);

    // Test validation utilities
    let valid_addr = [1u8; 20];
    assert!(validation::is_valid_address(valid_addr));

    let zero_addr = [0u8; 20];
    assert!(!validation::is_valid_address(zero_addr));

    // Test gas validation
    validation::validate_gas(1000, 21_000)?;

    let invalid_gas_result = validation::validate_gas(0, 21_000);
    assert!(invalid_gas_result.is_err());

    // Test transaction data validation
    let small_data = vec![0u8; 1000];
    validation::validate_tx_data(&small_data)?;

    let large_data = vec![0u8; 2_000_000];
    let large_data_result = validation::validate_tx_data(&large_data);
    assert!(large_data_result.is_err());

    // Test latency timer
    let timer = time::LatencyTimer::new(Duration::from_millis(100));

    // Add a small delay to ensure measurable time
    std::thread::sleep(Duration::from_millis(1));

    timer.check_timeout()?;

    let elapsed_nanoseconds = timer.elapsed_ns();
    let elapsed_microseconds = timer.elapsed_us();
    assert!(elapsed_nanoseconds > 0);
    assert!(elapsed_microseconds > 0);

    Ok(())
}

/// Test mempool watcher integration
///
/// Verifies that mempool watching functionality works correctly
/// and integrates properly with transaction analysis.
#[test]
fn test_mempool_watcher_integration() -> CoreResult<()> {
    let mut watcher = MempoolWatcher::new();

    // Test watcher lifecycle
    assert!(!watcher.is_running());
    watcher.start()?;
    assert!(watcher.is_running());

    // Test event processing
    let mut tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    // Set hash for the transaction
    tx.hash = Some([1u8; 32]);

    let event = MempoolEvent::TransactionAdded {
        transaction: tx.clone(),
        timestamp: Instant::now(),
    };

    // Test event methods
    assert!(event.transaction().is_some());
    assert!(event.transaction_hash().is_some());

    // Process event
    watcher.process_event(event)?;

    // Check statistics
    let stats = watcher.statistics();
    assert_eq!(stats.events_processed, 1);
    assert_eq!(stats.transactions_seen, 1);
    assert!(stats.is_running);

    // Test replacement event
    let replacement_event = MempoolEvent::TransactionReplaced {
        old_hash: [1u8; 32],
        new_transaction: tx,
        timestamp: Instant::now(),
    };

    watcher.process_event(replacement_event)?;

    let updated_stats = watcher.statistics();
    assert_eq!(updated_stats.events_processed, 2);
    assert_eq!(updated_stats.replacements_detected, 1);

    // Test health check
    let health = watcher.health_check();
    assert!(health.connection_stable);
    assert!(health.score <= 100);

    // Test watching for transactions
    let events = watcher.watch_for_transactions()?;
    assert!(events.is_empty()); // Simulation returns empty

    // Test stop
    watcher.stop()?;
    assert!(!watcher.is_running());

    // Test processing when not running
    let stopped_event = MempoolEvent::TransactionAdded {
        transaction: Transaction::new(
            [3u8; 20],
            Some([4u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            1,
            Vec::with_capacity(0),
        ),
        timestamp: Instant::now(),
    };

    let stopped_result = watcher.process_event(stopped_event);
    assert!(stopped_result.is_err());

    Ok(())
}

/// Test engine comprehensive integration
///
/// Verifies that the TallyEngine works correctly with all components
/// and meets performance requirements.
#[test]
fn test_engine_comprehensive_integration() -> CoreResult<()> {
    let mut engine = TallyEngine::new()?;

    // Test engine lifecycle
    assert_eq!(
        engine.status(),
        tallyio_core::engine::EngineStatus::Initializing
    );
    engine.start()?;
    assert_eq!(engine.status(), tallyio_core::engine::EngineStatus::Running);

    // Test transaction processing
    let tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    // Process transaction
    let result = engine.process_transaction(tx.clone())?;
    assert!(result.is_success());

    // Test batch processing
    let transactions = vec![
        tx,
        Transaction::new(
            [3u8; 20],
            Some([4u8; 20]),
            Price::from_ether(2),
            Price::from_gwei(25),
            Gas::new(50_000),
            1,
            vec![1, 2, 3, 4],
        ),
    ];

    let batch_results = engine.process_batch(transactions)?;
    assert_eq!(batch_results.len(), 2);
    assert!(batch_results
        .iter()
        .all(tallyio_core::ProcessingResult::is_success));

    // Test engine status
    let status = engine.status();
    assert!(matches!(
        status,
        tallyio_core::engine::EngineStatus::Running
    ));

    // Test engine health
    let health = engine.health_check()?;
    assert!(health.score > 0);
    assert!(health.status.is_operational());

    // Test engine stop
    engine.stop()?;
    assert_eq!(engine.status(), tallyio_core::engine::EngineStatus::Stopped);

    Ok(())
}

/// Test mempool filter integration
///
/// Verifies that mempool filtering works correctly and integrates
/// properly with transaction analysis.
#[test]
fn test_mempool_filter_integration() -> CoreResult<()> {
    use tallyio_core::mempool::FilterConfig;

    let config = FilterConfig {
        enable_defi_filter: false, // Disable DeFi filter for test
        enable_mev_filter: false,  // Disable MEV filter for test
        ..FilterConfig::default()
    };
    let filter = MempoolFilter::new(config);

    // Test transaction filtering
    let valid_tx = Transaction::new(
        [1u8; 20],
        Some([2u8; 20]),
        Price::from_ether(1),
        Price::from_gwei(20),
        Gas::new(21_000),
        0,
        Vec::with_capacity(0),
    );

    let invalid_tx = Transaction::new(
        [0u8; 20], // Invalid zero address
        Some([2u8; 20]),
        Price::from_ether(0),    // Zero value
        Price::new(500_000_000), // < 1 gwei (too low)
        Gas::new(10_000),        // Low gas limit
        0,
        Vec::with_capacity(0),
    );

    // Test filtering
    let valid_result = filter.filter(&valid_tx)?;
    let invalid_result = filter.filter(&invalid_tx)?;

    assert!(valid_result.should_process());
    assert!(!invalid_result.should_process());

    // Test filter statistics
    let stats = filter.statistics();
    assert!(stats.total_transactions >= 2);
    assert!(stats.accepted_transactions >= 1);
    assert!(stats.rejected_transactions >= 1);
    assert!(stats.acceptance_rate <= 1.0);

    Ok(())
}
