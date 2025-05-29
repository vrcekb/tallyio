//! `TallyIO` Engine - Ultra-performant transaction processing core

use crate::{CoreError, Metrics, Price, Transaction};
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// High-performance transaction processing engine
///
/// Cache-aligned engine for processing transactions with <1ms latency guarantee.
/// Uses lock-free queues and atomic operations for maximum throughput.
#[repr(C, align(64))]
pub struct TallyEngine {
    // Hot data - cache line aligned
    counter: AtomicU64,
    metrics: Metrics,

    // Processing queues
    incoming_queue: SegQueue<Transaction>,
    processed_queue: SegQueue<ProcessedTransaction>,
}

// Use types from types.rs module
use crate::{MevOpportunity, ProcessedTransaction, ProcessingStatus};

impl TallyEngine {
    /// Create new engine instance
    ///
    /// Initializes a new transaction processing engine with empty queues
    /// and zero metrics. This operation cannot fail under normal conditions.
    ///
    /// # Errors
    /// Currently never returns an error, but uses Result for future extensibility.
    ///
    /// # Returns
    /// New `TallyEngine` instance ready for processing
    #[allow(clippy::unnecessary_wraps)] // API consistency with other constructors
    pub const fn new() -> Result<Self, CoreError> {
        Ok(Self {
            counter: AtomicU64::new(0),
            metrics: Metrics::new(),
            incoming_queue: SegQueue::new(),
            processed_queue: SegQueue::new(),
        })
    }

    /// Submit transaction for processing
    ///
    /// Adds a transaction to the processing queue and immediately processes it.
    /// This provides synchronous processing for testing and simple use cases.
    ///
    /// # Arguments
    /// * `tx` - Transaction to process
    ///
    /// # Returns
    /// `Ok(())` if transaction was submitted successfully
    #[allow(clippy::missing_errors_doc)]
    #[allow(clippy::unnecessary_wraps)]
    pub fn submit_transaction(&self, tx: Transaction) -> Result<(), CoreError> {
        let start = Instant::now();
        let result = self.process_transaction_internal(tx, start);
        self.processed_queue.push(result);
        Ok(())
    }

    /// Process next transaction - <1ms guarantee
    ///
    /// Processes the next transaction from the queue with ultra-low latency.
    /// Returns `None` if no transactions are available for processing.
    ///
    /// # Returns
    /// `Some(ProcessedTransaction)` if a transaction was processed, `None` if queue is empty
    pub fn process_next(&self) -> Option<ProcessedTransaction> {
        let start = Instant::now();

        self.incoming_queue.pop().map(|tx| {
            let result = self.process_transaction_internal(tx, start);
            self.processed_queue.push(result.clone());
            result
        })
    }

    /// Internal transaction processing
    ///
    /// Core processing logic with validation, MEV scanning, and metrics collection.
    /// Designed for <1ms execution time on the critical path.
    fn process_transaction_internal(
        &self,
        tx: Transaction,
        start: Instant,
    ) -> ProcessedTransaction {
        // Validate transaction
        if tx.gas_price.value() == 0 {
            let processing_time = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.metrics.record_transaction(processing_time);
            return ProcessedTransaction {
                transaction: tx,
                processing_time_ns: processing_time,
                mev_opportunity: None,
                status: ProcessingStatus::Rejected("Zero gas price".to_string()),
            };
        }

        // Check for MEV opportunities
        let opportunity_value = if tx.is_defi_related() {
            Self::scan_mev_opportunity(&tx)
        } else {
            None
        };

        let processing_time = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);

        // Record metrics
        self.metrics.record_transaction(processing_time);
        if opportunity_value.is_some() {
            self.metrics.record_opportunity();
        }

        // Ensure <1ms latency
        debug_assert!(
            processing_time < 1_000_000,
            "Processing took {processing_time}ns"
        );

        ProcessedTransaction {
            transaction: tx,
            processing_time_ns: processing_time,
            mev_opportunity: opportunity_value.map(|price| MevOpportunity {
                profit_wei: price,
                gas_cost: Price::new(0), // Simplified for now
                confidence: 80,
            }),
            status: ProcessingStatus::Success,
        }
    }

    /// Scan for MEV opportunities - ultra-fast heuristics
    ///
    /// Performs rapid pattern matching on transaction data to identify
    /// potential MEV opportunities. Uses simplified heuristics for speed.
    ///
    /// # Arguments
    /// * `tx` - Transaction to analyze
    ///
    /// # Returns
    /// `Some(Price)` if opportunity found, `None` otherwise
    #[must_use]
    pub fn scan_mev_opportunity(tx: &Transaction) -> Option<Price> {
        // Quick pattern matching for common MEV opportunities
        if tx.data.len() >= 4 {
            let method_sig = &tx.data[0..4];

            // Common DEX method signatures (simplified)
            match method_sig {
                [0xa9, 0x05, 0x9c, 0xbb] => {
                    // swapExactTokensForTokens
                    // Simplified arbitrage detection
                    if tx.gas_price.value() > 50_000_000_000 {
                        // 50 gwei
                        return Some(Price::new(tx.value.value() / 100)); // 1% opportunity
                    }
                }
                [0x38, 0xed, 0x17, 0x39] => {
                    // swapExactETHForTokens
                    if tx.value.value() > 1_000_000_000_000_000_000 {
                        // > 1 ETH
                        return Some(Price::new(tx.value.value() / 200)); // 0.5% opportunity
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Get next processed transaction
    ///
    /// Retrieves the next processed transaction from the output queue.
    /// Returns `None` if no processed transactions are available.
    ///
    /// # Returns
    /// `Some(ProcessedTransaction)` if available, `None` if queue is empty
    #[must_use]
    pub fn get_processed(&self) -> Option<ProcessedTransaction> {
        self.processed_queue.pop()
    }

    /// Get current metrics snapshot
    ///
    /// Creates a point-in-time snapshot of all engine metrics.
    /// All values are read atomically but may not be perfectly consistent.
    ///
    /// # Returns
    /// Current metrics snapshot
    #[must_use]
    pub fn metrics(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            transactions_processed: self.metrics.transactions_processed.load(Ordering::Relaxed),
            opportunities_found: self.metrics.opportunities_found.load(Ordering::Relaxed),
            errors_encountered: self.metrics.errors_encountered.load(Ordering::Relaxed),
            average_latency_ns: self.metrics.average_latency_ns(),
        }
    }

    /// Get queue sizes for monitoring
    ///
    /// Returns the current sizes of incoming and processed queues.
    /// Useful for monitoring system load and backpressure.
    ///
    /// # Returns
    /// Tuple of (`incoming_queue_size`, `processed_queue_size`)
    #[must_use]
    pub fn queue_sizes(&self) -> (usize, usize) {
        (self.incoming_queue.len(), self.processed_queue.len())
    }
}

/// Metrics snapshot for monitoring
///
/// Point-in-time snapshot of engine performance metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub transactions_processed: u64,
    pub opportunities_found: u64,
    pub errors_encountered: u64,
    pub average_latency_ns: u64,
}

impl Default for TallyEngine {
    fn default() -> Self {
        // Use match instead of expect to comply with zero-panic policy
        #[allow(clippy::option_if_let_else)] // Result, not Option
        match Self::new() {
            Ok(engine) => engine,
            Err(_) => {
                // This should never happen in normal circumstances
                // If it does, it's a programming error
                std::process::abort();
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::missing_errors_doc)]
mod tests {
    use super::*;
    use crate::{Address, Gas};

    #[test]
    fn test_engine_creation() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;
        assert_eq!(engine.queue_sizes(), (0, 0));
        Ok(())
    }

    #[test]
    fn test_zero_gas_price_rejection() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;
        let mut tx = Transaction::default();
        tx.gas_price = Price::new(0); // Zero gas price

        engine.submit_transaction(tx)?;

        // Process and check result
        if let Some(processed) = engine.get_processed() {
            assert!(matches!(processed.status, ProcessingStatus::Rejected(_)));
            if let ProcessingStatus::Rejected(reason) = processed.status {
                assert!(reason.contains("Zero gas price"));
            }
        }
        Ok(())
    }

    #[test]
    fn test_mev_opportunity_detection() -> Result<(), CoreError> {
        // Test swapExactTokensForTokens signature
        let tx = Transaction {
            data: vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // swapExactTokensForTokens + padding
            gas_price: Price::new(60_000_000_000),          // 60 gwei
            value: Price::new(1_000_000_000_000_000_000),   // 1 ETH
            ..Transaction::default()
        };

        let opportunity = TallyEngine::scan_mev_opportunity(&tx);
        assert!(opportunity.is_some(), "Should detect MEV opportunity");

        // Test swapExactETHForTokens signature
        let tx2 = Transaction {
            data: vec![0x38, 0xed, 0x17, 0x39, 0x00, 0x00], // swapExactETHForTokens + padding
            value: Price::new(2_000_000_000_000_000_000),   // 2 ETH
            ..Transaction::default()
        };

        let opportunity2 = TallyEngine::scan_mev_opportunity(&tx2);
        assert!(
            opportunity2.is_some(),
            "Should detect MEV opportunity for ETH swap"
        );

        Ok(())
    }

    #[test]
    fn test_no_mev_opportunity() -> Result<(), CoreError> {
        // Test with low gas price
        let mut tx = Transaction::default();
        tx.data = vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00];
        tx.gas_price = Price::new(10_000_000_000); // 10 gwei (too low)

        let opportunity = TallyEngine::scan_mev_opportunity(&tx);
        assert!(
            opportunity.is_none(),
            "Should not detect MEV opportunity with low gas"
        );

        // Test with unknown method signature
        let mut tx2 = Transaction::default();
        tx2.data = vec![0x12, 0x34, 0x56, 0x78, 0x00, 0x00]; // Unknown signature

        let opportunity2 = TallyEngine::scan_mev_opportunity(&tx2);
        assert!(
            opportunity2.is_none(),
            "Should not detect MEV opportunity for unknown method"
        );

        Ok(())
    }

    #[test]
    fn test_get_processed_transaction() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;

        // Initially empty
        assert!(engine.get_processed().is_none());

        // Submit transaction
        let tx = Transaction::default();
        engine.submit_transaction(tx)?;

        // Should have processed transaction (synchronous processing)
        let processed = engine.get_processed();
        assert!(
            processed.is_some(),
            "Transaction should be processed synchronously"
        );

        Ok(())
    }

    #[test]
    fn test_metrics_snapshot() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;

        let initial_metrics = engine.metrics();
        assert_eq!(initial_metrics.transactions_processed, 0);
        assert_eq!(initial_metrics.opportunities_found, 0);

        // Submit transaction with MEV opportunity
        let mut tx = Transaction::default();
        tx.data = vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00];
        tx.gas_price = Price::new(60_000_000_000);
        tx.value = Price::new(1_000_000_000_000_000_000);

        engine.submit_transaction(tx)?;

        let final_metrics = engine.metrics();
        assert!(final_metrics.transactions_processed > initial_metrics.transactions_processed);

        Ok(())
    }

    #[test]
    fn test_queue_sizes() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;

        let (incoming, processed) = engine.queue_sizes();
        assert_eq!(incoming, 0);
        assert_eq!(processed, 0);

        // Submit transaction
        let tx = Transaction::default();
        engine.submit_transaction(tx)?;

        let (_, processed_after) = engine.queue_sizes();
        assert!(processed_after > processed);

        Ok(())
    }

    #[test]
    fn test_engine_default() {
        let engine = TallyEngine::default();
        assert_eq!(engine.queue_sizes(), (0, 0));
    }

    #[test]
    fn test_defi_related_transaction() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;

        // Create DeFi-related transaction with valid gas price
        let mut tx = Transaction::default();
        tx.to = Some(Address::new([0x1; 20])); // Some address
        tx.data = vec![0xa9, 0x05, 0x9c, 0xbb]; // DeFi method signature
        tx.gas_price = Price::new(20_000_000_000); // 20 gwei - valid gas price

        engine.submit_transaction(tx)?;

        if let Some(processed) = engine.get_processed() {
            // Should have attempted MEV scanning for DeFi transaction
            assert!(matches!(processed.status, ProcessingStatus::Success));
        }

        Ok(())
    }

    #[test]
    fn test_short_data_no_mev() -> Result<(), CoreError> {
        // Test with data too short for method signature
        let mut tx = Transaction::default();
        tx.data = vec![0xa9, 0x05]; // Too short

        let opportunity = TallyEngine::scan_mev_opportunity(&tx);
        assert!(
            opportunity.is_none(),
            "Should not detect MEV opportunity with short data"
        );

        Ok(())
    }

    #[test]
    fn test_low_value_eth_swap() -> Result<(), CoreError> {
        // Test swapExactETHForTokens with low value
        let mut tx = Transaction::default();
        tx.data = vec![0x38, 0xed, 0x17, 0x39, 0x00, 0x00];
        tx.value = Price::new(500_000_000_000_000_000); // 0.5 ETH (too low)

        let opportunity = TallyEngine::scan_mev_opportunity(&tx);
        assert!(
            opportunity.is_none(),
            "Should not detect MEV opportunity with low ETH value"
        );

        Ok(())
    }

    #[test]
    fn test_process_next() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;

        // Initially no transactions to process
        assert!(engine.process_next().is_none());

        // Add transaction to incoming queue
        let tx = Transaction::default();
        engine.incoming_queue.push(tx);

        // Process next should return the processed transaction
        let processed = engine.process_next();
        assert!(processed.is_some());

        if let Some(processed) = processed {
            assert!(matches!(processed.status, ProcessingStatus::Rejected(_))); // Zero gas price
        }

        Ok(())
    }

    #[test]
    fn test_non_defi_transaction() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;

        // Create non-DeFi transaction (no data, simple transfer)
        let tx = Transaction {
            gas_price: Price::new(20_000_000_000), // Valid gas price
            data: vec![],                          // No data = simple transfer
            ..Transaction::default()
        };

        engine.submit_transaction(tx)?;

        if let Some(processed) = engine.get_processed() {
            assert!(matches!(processed.status, ProcessingStatus::Success));
            assert!(processed.mev_opportunity.is_none()); // No MEV for simple transfer
        }

        Ok(())
    }

    #[test]
    fn test_defi_transaction_with_mev() -> Result<(), CoreError> {
        let engine = TallyEngine::new()?;

        // Create DeFi transaction that will trigger MEV opportunity detection
        let tx = Transaction {
            gas_price: Price::new(60_000_000_000), // 60 gwei - high enough for MEV
            gas_limit: Gas::new(100_000),          // High gas limit for DeFi
            value: Price::new(2_000_000_000_000_000_000), // 2 ETH - high value
            data: vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // swapExactTokensForTokens
            ..Transaction::default()
        };

        engine.submit_transaction(tx)?;

        if let Some(processed) = engine.get_processed() {
            assert!(matches!(processed.status, ProcessingStatus::Success));
            assert!(processed.mev_opportunity.is_some()); // Should have MEV opportunity
        }

        // Check that opportunity was recorded in metrics
        let metrics = engine.metrics();
        assert!(metrics.opportunities_found > 0);

        Ok(())
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_debug_assert_coverage() -> Result<(), CoreError> {
        // This test covers the debug_assert! line (120) in debug builds
        let engine = TallyEngine::new()?;

        // Create a transaction that will be processed quickly (under 1ms)
        let tx = Transaction {
            gas_price: Price::new(20_000_000_000), // Valid gas price
            data: vec![],                          // Simple transfer
            ..Transaction::default()
        };

        engine.submit_transaction(tx)?;

        // The debug_assert should pass (not panic) for normal processing times
        if let Some(processed) = engine.get_processed() {
            assert!(matches!(processed.status, ProcessingStatus::Success));
            // Processing time should be well under 1ms (1,000,000 ns)
            assert!(processed.processing_time_ns < 1_000_000);
        }

        Ok(())
    }
}
