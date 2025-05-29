//! `TallyIO` Engine - Ultra-performant transaction processing core

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use crossbeam::queue::SegQueue;
use crate::{CoreError, CriticalError, Transaction, Metrics, Price};

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

/// Processed transaction with results
///
/// Contains the original transaction plus processing metadata and results.
#[derive(Debug, Clone)]
pub struct ProcessedTransaction {
    pub original: Transaction,
    pub processing_time_ns: u64,
    pub opportunity_value: Option<Price>,
    pub status: ProcessingStatus,
}

/// Transaction processing status
///
/// Indicates the outcome of transaction processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessingStatus {
    /// Transaction processed successfully
    Success,
    /// Transaction rejected with reason
    Rejected(String),
    /// Processing error with code
    Error(u16),
}

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
    pub const fn new() -> Result<Self, CoreError> {
        Ok(Self {
            counter: AtomicU64::new(0),
            metrics: Metrics::new(),
            incoming_queue: SegQueue::new(),
            processed_queue: SegQueue::new(),
        })
    }

    /// Submit transaction for processing - lock-free
    ///
    /// Adds a transaction to the processing queue using lock-free operations.
    /// Validates basic transaction parameters before queuing.
    ///
    /// # Arguments
    /// * `tx` - Transaction to process
    ///
    /// # Errors
    /// * `CriticalError::Invalid` - If gas limit is zero
    ///
    /// # Returns
    /// `Ok(())` if transaction was queued successfully
    pub fn submit_transaction(&self, tx: Transaction) -> Result<(), CoreError> {
        if tx.gas_limit.value() == 0 {
            return Err(CoreError::Critical(CriticalError::Invalid(1)));
        }

        self.incoming_queue.push(tx);
        self.counter.fetch_add(1, Ordering::Relaxed);
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
        start: Instant
    ) -> ProcessedTransaction {
        // Validate transaction
        if tx.gas_price.value() == 0 {
            let processing_time = u64::try_from(start.elapsed().as_nanos())
                .unwrap_or(u64::MAX);
            self.metrics.record_transaction(processing_time);
            return ProcessedTransaction {
                original: tx,
                processing_time_ns: processing_time,
                opportunity_value: None,
                status: ProcessingStatus::Rejected("Zero gas price".to_string()),
            };
        }

        // Check for MEV opportunities
        let opportunity_value = if tx.is_defi_related() {
            Self::scan_mev_opportunity(&tx)
        } else {
            None
        };

        let processing_time = u64::try_from(start.elapsed().as_nanos())
            .unwrap_or(u64::MAX);

        // Record metrics
        self.metrics.record_transaction(processing_time);
        if opportunity_value.is_some() {
            self.metrics.record_opportunity();
        }

        // Ensure <1ms latency
        debug_assert!(processing_time < 1_000_000, "Processing took {processing_time}ns");

        ProcessedTransaction {
            original: tx,
            processing_time_ns: processing_time,
            opportunity_value,
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
                [0xa9, 0x05, 0x9c, 0xbb] => { // swapExactTokensForTokens
                    // Simplified arbitrage detection
                    if tx.gas_price.value() > 50_000_000_000 { // 50 gwei
                        return Some(Price::new(tx.value.value() / 100)); // 1% opportunity
                    }
                }
                [0x38, 0xed, 0x17, 0x39] => { // swapExactETHForTokens
                    if tx.value.value() > 1_000_000_000_000_000_000 { // > 1 ETH
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
        // This expect is acceptable in Default implementation
        // as it represents a programming error if engine creation fails
        #[allow(clippy::expect_used)]
        Self::new().expect("Failed to create TallyEngine")
    }
}
