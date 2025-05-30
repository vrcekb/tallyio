//! Processing result types for TallyIO core
//!
//! This module provides types for representing the results of transaction processing
//! and MEV opportunity execution with detailed performance metrics.

use crate::types::{Gas, Opportunity, Price, TransactionHash, TransactionStatus};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

/// Result of transaction processing
///
/// Contains detailed information about the processing outcome including
/// performance metrics and execution details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Unique result ID
    pub id: Uuid,
    /// Transaction hash that was processed
    pub transaction_hash: Option<TransactionHash>,
    /// Final transaction status
    pub status: TransactionStatus,
    /// Processing time in nanoseconds
    pub processing_time_ns: u64,
    /// Queue time in nanoseconds (time spent waiting)
    pub queue_time_ns: u64,
    /// Execution time in nanoseconds (actual processing)
    pub execution_time_ns: u64,
    /// Gas used during execution
    pub gas_used: Gas,
    /// Effective gas price paid
    pub effective_gas_price: Price,
    /// Total transaction cost
    pub total_cost: Price,
    /// MEV opportunities found during processing
    pub mev_opportunities: Vec<Opportunity>,
    /// Error message if processing failed
    pub error_message: Option<String>,
    /// Additional metadata
    pub metadata: ProcessingMetadata,
}

/// Additional metadata for processing results
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Worker thread ID that processed the transaction
    pub worker_id: u32,
    /// CPU core used for processing
    pub cpu_core: Option<u32>,
    /// Memory usage during processing in bytes
    pub memory_usage: u64,
    /// Number of retries attempted
    pub retry_count: u32,
    /// Cache hit/miss information
    pub cache_hits: u32,
    pub cache_misses: u32,
    /// Network latency if applicable
    pub network_latency_ns: Option<u64>,
    /// Database query time if applicable
    pub db_query_time_ns: Option<u64>,
}

impl ProcessingResult {
    /// Create a new processing result
    #[must_use]
    pub fn new(
        transaction_hash: Option<TransactionHash>,
        status: TransactionStatus,
        processing_time: Duration,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            transaction_hash,
            status,
            processing_time_ns: processing_time.as_nanos() as u64,
            queue_time_ns: 0,
            execution_time_ns: processing_time.as_nanos() as u64,
            gas_used: Gas::new(0),
            effective_gas_price: Price::new(0),
            total_cost: Price::new(0),
            mev_opportunities: Vec::with_capacity(0),
            error_message: None,
            metadata: ProcessingMetadata::default(),
        }
    }

    /// Create a successful processing result
    #[must_use]
    pub fn success(
        transaction_hash: TransactionHash,
        processing_time: Duration,
        gas_used: Gas,
        gas_price: Price,
    ) -> Self {
        let total_cost = gas_used.cost_at_price(gas_price);
        Self {
            id: Uuid::new_v4(),
            transaction_hash: Some(transaction_hash),
            status: TransactionStatus::Success,
            processing_time_ns: processing_time.as_nanos() as u64,
            queue_time_ns: 0,
            execution_time_ns: processing_time.as_nanos() as u64,
            gas_used,
            effective_gas_price: gas_price,
            total_cost,
            mev_opportunities: Vec::with_capacity(0),
            error_message: None,
            metadata: ProcessingMetadata::default(),
        }
    }

    /// Create a failed processing result
    #[must_use]
    pub fn failure(
        transaction_hash: Option<TransactionHash>,
        processing_time: Duration,
        error: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            transaction_hash,
            status: TransactionStatus::Failed,
            processing_time_ns: processing_time.as_nanos() as u64,
            queue_time_ns: 0,
            execution_time_ns: processing_time.as_nanos() as u64,
            gas_used: Gas::new(0),
            effective_gas_price: Price::new(0),
            total_cost: Price::new(0),
            mev_opportunities: Vec::with_capacity(0),
            error_message: Some(error),
            metadata: ProcessingMetadata::default(),
        }
    }

    /// Get total processing time as Duration
    #[must_use]
    pub fn processing_time(&self) -> Duration {
        Duration::from_nanos(self.processing_time_ns)
    }

    /// Get queue time as Duration
    #[must_use]
    pub fn queue_time(&self) -> Duration {
        Duration::from_nanos(self.queue_time_ns)
    }

    /// Get execution time as Duration
    #[must_use]
    pub fn execution_time(&self) -> Duration {
        Duration::from_nanos(self.execution_time_ns)
    }

    /// Check if processing was successful
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Check if processing failed
    #[must_use]
    pub fn is_failure(&self) -> bool {
        self.status.is_failed()
    }

    /// Check if processing met latency requirements
    #[must_use]
    pub fn meets_latency_requirement(&self, max_latency: Duration) -> bool {
        self.processing_time() <= max_latency
    }

    /// Get the number of MEV opportunities found
    #[must_use]
    pub fn mev_opportunity_count(&self) -> usize {
        self.mev_opportunities.len()
    }

    /// Get total value of MEV opportunities found
    #[must_use]
    pub fn total_mev_value(&self) -> Price {
        self.mev_opportunities
            .iter()
            .fold(Price::new(0), |acc, opp| acc.add(opp.value()))
    }

    /// Set timing information
    pub fn set_timing(&mut self, queue_time: Duration, execution_time: Duration) {
        self.queue_time_ns = queue_time.as_nanos() as u64;
        self.execution_time_ns = execution_time.as_nanos() as u64;
        self.processing_time_ns = self.queue_time_ns + self.execution_time_ns;
    }

    /// Set gas information
    pub fn set_gas_info(&mut self, gas_used: Gas, gas_price: Price) {
        self.gas_used = gas_used;
        self.effective_gas_price = gas_price;
        self.total_cost = gas_used.cost_at_price(gas_price);
    }

    /// Add MEV opportunity
    pub fn add_mev_opportunity(&mut self, opportunity: Opportunity) {
        self.mev_opportunities.push(opportunity);
    }

    /// Set error message
    pub fn set_error(&mut self, error: String) {
        self.error_message = Some(error);
        if self.status == TransactionStatus::Success {
            self.status = TransactionStatus::Failed;
        }
    }

    /// Set worker metadata
    pub fn set_worker_metadata(&mut self, worker_id: u32, cpu_core: Option<u32>) {
        self.metadata.worker_id = worker_id;
        self.metadata.cpu_core = cpu_core;
    }

    /// Set memory usage
    pub fn set_memory_usage(&mut self, bytes: u64) {
        self.metadata.memory_usage = bytes;
    }

    /// Set retry count
    pub fn set_retry_count(&mut self, count: u32) {
        self.metadata.retry_count = count;
    }

    /// Set cache statistics
    pub fn set_cache_stats(&mut self, hits: u32, misses: u32) {
        self.metadata.cache_hits = hits;
        self.metadata.cache_misses = misses;
    }

    /// Set network latency
    pub fn set_network_latency(&mut self, latency: Duration) {
        self.metadata.network_latency_ns = Some(latency.as_nanos() as u64);
    }

    /// Set database query time
    pub fn set_db_query_time(&mut self, time: Duration) {
        self.metadata.db_query_time_ns = Some(time.as_nanos() as u64);
    }

    /// Calculate cache hit ratio
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.metadata.cache_hits + self.metadata.cache_misses;
        if total == 0 {
            0.0
        } else {
            f64::from(self.metadata.cache_hits) / f64::from(total)
        }
    }

    /// Get processing efficiency score (0-100)
    #[must_use]
    pub fn efficiency_score(&self) -> u8 {
        let mut score = 100u8;

        // Penalize for high latency
        if self.processing_time_ns > 1_000_000 {
            // > 1ms
            score = score.saturating_sub(20);
        }

        // Penalize for retries
        score = score.saturating_sub(self.metadata.retry_count as u8 * 10);

        // Penalize for low cache hit ratio
        let cache_penalty = ((1.0 - self.cache_hit_ratio()) * 20.0) as u8;
        score = score.saturating_sub(cache_penalty);

        // Penalize for failures
        if self.is_failure() {
            score = score.saturating_sub(50);
        }

        score
    }
}

impl Default for ProcessingMetadata {
    fn default() -> Self {
        Self {
            worker_id: 0,
            cpu_core: None,
            memory_usage: 0,
            retry_count: 0,
            cache_hits: 0,
            cache_misses: 0,
            network_latency_ns: None,
            db_query_time_ns: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OpportunityType;

    #[test]
    fn test_processing_result_creation() {
        let tx_hash = [1u8; 32];
        let processing_time = Duration::from_micros(500);

        let result = ProcessingResult::success(
            tx_hash,
            processing_time,
            Gas::new(21000),
            Price::from_gwei(20),
        );

        assert!(result.is_success());
        assert!(!result.is_failure());
        assert_eq!(result.processing_time(), processing_time);
        assert_eq!(result.gas_used.as_units(), 21000);
    }

    #[test]
    fn test_processing_result_failure() {
        let processing_time = Duration::from_millis(2);
        let error = "Transaction reverted".to_string();

        let result = ProcessingResult::failure(None, processing_time, error.clone());

        assert!(!result.is_success());
        assert!(result.is_failure());
        assert_eq!(result.error_message, Some(error));
    }

    #[test]
    fn test_latency_requirements() {
        let result =
            ProcessingResult::new(None, TransactionStatus::Success, Duration::from_micros(800));

        assert!(result.meets_latency_requirement(Duration::from_millis(1)));
        assert!(!result.meets_latency_requirement(Duration::from_micros(500)));
    }

    #[test]
    fn test_mev_opportunities() {
        let mut result =
            ProcessingResult::new(None, TransactionStatus::Success, Duration::from_micros(500));

        let opp1 = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );
        let opp2 = Opportunity::new(
            OpportunityType::Liquidation,
            Price::from_ether(2),
            Gas::new(150_000),
        );

        result.add_mev_opportunity(opp1);
        result.add_mev_opportunity(opp2);

        assert_eq!(result.mev_opportunity_count(), 2);
        assert_eq!(result.total_mev_value().as_ether(), 3);
    }

    #[test]
    fn test_timing_information() {
        let mut result = ProcessingResult::new(
            None,
            TransactionStatus::Success,
            Duration::from_micros(1000),
        );

        let queue_time = Duration::from_micros(200);
        let execution_time = Duration::from_micros(800);

        result.set_timing(queue_time, execution_time);

        assert_eq!(result.queue_time(), queue_time);
        assert_eq!(result.execution_time(), execution_time);
        assert_eq!(result.processing_time(), queue_time + execution_time);
    }

    #[test]
    fn test_cache_statistics() {
        let mut result =
            ProcessingResult::new(None, TransactionStatus::Success, Duration::from_micros(500));

        result.set_cache_stats(80, 20);
        assert_eq!(result.cache_hit_ratio(), 0.8);

        result.set_cache_stats(0, 0);
        assert_eq!(result.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_efficiency_score() {
        let mut result = ProcessingResult::success(
            [1u8; 32],
            Duration::from_micros(500), // Good latency
            Gas::new(21000),
            Price::from_gwei(20),
        );

        result.set_cache_stats(90, 10); // Good cache ratio
        result.set_retry_count(0); // No retries

        let score = result.efficiency_score();
        assert!(score >= 80); // Should be high efficiency

        // Test with poor performance
        result.set_retry_count(3); // Multiple retries
        result.set_cache_stats(10, 90); // Poor cache ratio
        result.processing_time_ns = 2_000_000; // > 1ms

        let poor_score = result.efficiency_score();
        assert!(poor_score < score); // Should be lower
    }

    #[test]
    fn test_metadata_operations() {
        let mut result =
            ProcessingResult::new(None, TransactionStatus::Success, Duration::from_micros(500));

        result.set_worker_metadata(5, Some(2));
        result.set_memory_usage(1024 * 1024); // 1MB
        result.set_network_latency(Duration::from_micros(100));
        result.set_db_query_time(Duration::from_micros(50));

        assert_eq!(result.metadata.worker_id, 5);
        assert_eq!(result.metadata.cpu_core, Some(2));
        assert_eq!(result.metadata.memory_usage, 1024 * 1024);
        assert_eq!(result.metadata.network_latency_ns, Some(100_000));
        assert_eq!(result.metadata.db_query_time_ns, Some(50_000));
    }
}
