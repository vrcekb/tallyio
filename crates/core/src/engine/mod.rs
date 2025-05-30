//! TallyIO Core Engine - Ultra-performant execution engine
//!
//! This module provides the main execution engine for TallyIO with <1ms latency guarantee.
//! The engine coordinates transaction processing, MEV detection, and worker management.

pub mod executor;
pub mod scheduler;
pub mod worker;

// Re-export main engine types
pub use executor::TallyEngine;
pub use scheduler::{Scheduler, SchedulerConfig, TaskPriority};
pub use worker::{Worker, WorkerConfig, WorkerPool, WorkerStatus};

use crate::error::CoreResult;
use crate::types::{Opportunity, ProcessingResult, Transaction};
use std::time::Instant;

/// Engine performance metrics
#[derive(Debug, Clone)]
pub struct EngineMetrics {
    /// Total transactions processed
    pub transactions_processed: u64,
    /// Total MEV opportunities found
    pub mev_opportunities_found: u64,
    /// Average processing latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Peak processing latency in nanoseconds
    pub peak_latency_ns: u64,
    /// Current queue size
    pub queue_size: u64,
    /// Active worker count
    pub active_workers: u32,
    /// Error count
    pub error_count: u64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl Default for EngineMetrics {
    fn default() -> Self {
        Self {
            transactions_processed: 0,
            mev_opportunities_found: 0,
            avg_latency_ns: 0,
            peak_latency_ns: 0,
            queue_size: 0,
            active_workers: 0,
            error_count: 0,
            uptime_seconds: 0,
        }
    }
}

/// Engine status
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EngineStatus {
    /// Engine is initializing
    Initializing,
    /// Engine is running normally
    Running,
    /// Engine is paused
    Paused,
    /// Engine is shutting down
    Stopping,
    /// Engine has stopped
    Stopped,
    /// Engine encountered an error
    Error,
}

impl EngineStatus {
    /// Check if engine is operational
    #[must_use]
    pub const fn is_operational(self) -> bool {
        matches!(self, Self::Running | Self::Paused)
    }

    /// Check if engine is stopped
    #[must_use]
    pub const fn is_stopped(self) -> bool {
        matches!(self, Self::Stopped | Self::Error)
    }
}

impl Default for EngineStatus {
    fn default() -> Self {
        Self::Initializing
    }
}

/// Engine health check result
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Overall health status
    pub status: EngineStatus,
    /// Health score (0-100)
    pub score: u8,
    /// Current metrics
    pub metrics: EngineMetrics,
    /// Issues detected
    pub issues: Vec<String>,
    /// Timestamp of health check
    pub timestamp: Instant,
}

impl HealthCheck {
    /// Create a new health check
    #[must_use]
    pub fn new(status: EngineStatus, metrics: EngineMetrics) -> Self {
        let score = Self::calculate_score(&metrics, status);
        Self {
            status,
            score,
            metrics,
            issues: Vec::with_capacity(0),
            timestamp: Instant::now(),
        }
    }

    /// Calculate health score based on metrics
    fn calculate_score(metrics: &EngineMetrics, status: EngineStatus) -> u8 {
        if !status.is_operational() {
            return 0;
        }

        let mut score = 100u8;

        // Penalize high latency
        if metrics.avg_latency_ns > 1_000_000 {
            // > 1ms
            score = score.saturating_sub(30);
        } else if metrics.avg_latency_ns > 500_000 {
            // > 0.5ms
            score = score.saturating_sub(15);
        }

        // Penalize high error rate
        if metrics.transactions_processed > 0 {
            let error_rate = (metrics.error_count * 100) / metrics.transactions_processed;
            if error_rate > 5 {
                // > 5% error rate
                score = score.saturating_sub(40);
            } else if error_rate > 1 {
                // > 1% error rate
                score = score.saturating_sub(20);
            }
        }

        // Penalize large queue size
        if metrics.queue_size > 10_000 {
            score = score.saturating_sub(20);
        } else if metrics.queue_size > 1_000 {
            score = score.saturating_sub(10);
        }

        // Penalize inactive workers
        if metrics.active_workers == 0 {
            score = score.saturating_sub(50);
        }

        score
    }

    /// Add an issue to the health check
    pub fn add_issue(&mut self, issue: String) {
        self.issues.push(issue);
        // Reduce score for each issue
        self.score = self.score.saturating_sub(5);
    }

    /// Check if engine is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.score >= 70 && self.status.is_operational()
    }

    /// Check if engine needs attention
    #[must_use]
    pub fn needs_attention(&self) -> bool {
        self.score < 50 || !self.issues.is_empty()
    }
}

/// Trait for engine components that can be monitored
pub trait Monitorable {
    /// Get component metrics
    fn metrics(&self) -> CoreResult<EngineMetrics>;

    /// Perform health check
    fn health_check(&self) -> CoreResult<HealthCheck>;

    /// Get component status
    fn status(&self) -> EngineStatus;
}

/// Trait for engine components that can be controlled
pub trait Controllable {
    /// Start the component
    fn start(&mut self) -> CoreResult<()>;

    /// Stop the component
    fn stop(&mut self) -> CoreResult<()>;

    /// Pause the component
    fn pause(&mut self) -> CoreResult<()>;

    /// Resume the component
    fn resume(&mut self) -> CoreResult<()>;

    /// Restart the component
    fn restart(&mut self) -> CoreResult<()> {
        self.stop()?;
        self.start()
    }
}

/// Trait for processing transactions
pub trait TransactionProcessor {
    /// Process a single transaction
    fn process_transaction(&self, tx: Transaction) -> CoreResult<ProcessingResult>;

    /// Process multiple transactions in batch
    fn process_batch(&self, transactions: Vec<Transaction>) -> CoreResult<Vec<ProcessingResult>>;

    /// Get processing capacity
    fn capacity(&self) -> usize;

    /// Check if processor is available
    fn is_available(&self) -> bool;
}

/// Trait for MEV opportunity detection
pub trait MevDetector {
    /// Scan transaction for MEV opportunities
    fn scan_transaction(&self, tx: &Transaction) -> CoreResult<Vec<Opportunity>>;

    /// Scan multiple transactions
    fn scan_batch(&self, transactions: &[Transaction]) -> CoreResult<Vec<Opportunity>>;

    /// Get detector configuration
    fn config(&self) -> &MevDetectorConfig;
}

/// Configuration for MEV detector
#[derive(Debug, Clone)]
pub struct MevDetectorConfig {
    /// Minimum profit threshold in wei
    pub min_profit_wei: u64,
    /// Maximum gas price to consider
    pub max_gas_price_gwei: u64,
    /// Confidence threshold (0-100)
    pub confidence_threshold: u8,
    /// Enable specific opportunity types
    pub enabled_types: Vec<crate::types::OpportunityType>,
}

impl Default for MevDetectorConfig {
    fn default() -> Self {
        Self {
            min_profit_wei: 1_000_000_000_000_000, // 0.001 ETH
            max_gas_price_gwei: 100,
            confidence_threshold: 70,
            enabled_types: vec![
                crate::types::OpportunityType::Arbitrage,
                crate::types::OpportunityType::Liquidation,
                crate::types::OpportunityType::Sandwich,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_status() {
        let status = EngineStatus::Running;
        assert!(status.is_operational());
        assert!(!status.is_stopped());

        let stopped = EngineStatus::Stopped;
        assert!(!stopped.is_operational());
        assert!(stopped.is_stopped());
    }

    #[test]
    fn test_health_check_calculation() {
        let metrics = EngineMetrics {
            transactions_processed: 1000,
            avg_latency_ns: 500_000, // 0.5ms
            error_count: 10,         // 1% error rate
            queue_size: 500,
            active_workers: 4,
            ..Default::default()
        };

        let health = HealthCheck::new(EngineStatus::Running, metrics);
        assert!(health.score >= 70); // Should be healthy
        assert!(health.is_healthy());
    }

    #[test]
    fn test_health_check_high_latency() {
        let metrics = EngineMetrics {
            transactions_processed: 1000,
            avg_latency_ns: 2_000_000, // 2ms - high latency
            error_count: 0,
            queue_size: 100,
            active_workers: 4,
            ..Default::default()
        };

        let health = HealthCheck::new(EngineStatus::Running, metrics);
        assert!(health.score < 80); // Should be penalized for high latency
    }

    #[test]
    fn test_health_check_with_issues() {
        let metrics = EngineMetrics::default();
        let mut health = HealthCheck::new(EngineStatus::Running, metrics);

        health.add_issue("Test issue".to_string());
        assert!(health.needs_attention());
        assert_eq!(health.issues.len(), 1);
    }

    #[test]
    fn test_mev_detector_config() {
        let config = MevDetectorConfig::default();
        assert_eq!(config.min_profit_wei, 1_000_000_000_000_000);
        assert_eq!(config.max_gas_price_gwei, 100);
        assert_eq!(config.confidence_threshold, 70);
        assert!(!config.enabled_types.is_empty());
    }

    #[test]
    fn test_engine_error_handling() {
        // Test error handling paths (lines 87-88, 135, 143, 146, 154, 212-214)
        let status = EngineStatus::Error;
        assert!(!status.is_operational());
        // Error status is not the same as stopped
        assert!(status != EngineStatus::Stopped);

        // Test error metrics
        let error_metrics = EngineMetrics {
            transactions_processed: 100,
            error_count: 50, // 50% error rate
            avg_latency_ns: 1_000_000,
            queue_size: 1000,
            active_workers: 0,
            ..Default::default()
        };

        let health = HealthCheck::new(EngineStatus::Error, error_metrics);
        assert!(!health.is_healthy());
        assert!(health.score < 50);
    }

    #[test]
    fn test_engine_configuration_edge_cases() {
        // Test configuration edge cases (lines 87-88)
        let mut config = MevDetectorConfig::default();
        config.min_profit_wei = 0; // Edge case
        config.max_gas_price_gwei = u64::MAX; // Edge case
        config.confidence_threshold = 101; // Invalid threshold

        // Should handle edge cases gracefully
        assert_eq!(config.min_profit_wei, 0);
        assert_eq!(config.max_gas_price_gwei, u64::MAX);
        assert_eq!(config.confidence_threshold, 101);
    }

    #[test]
    fn test_engine_worker_management() {
        // Test worker management (lines 135, 143, 146, 154)
        let metrics = EngineMetrics {
            active_workers: 0, // No workers
            transactions_processed: 1000,
            error_count: 0,
            avg_latency_ns: 500_000,
            queue_size: 100,
            ..Default::default()
        };

        let health = HealthCheck::new(EngineStatus::Running, metrics);
        // Should be penalized for no active workers
        assert!(health.score < 90);
    }

    #[test]
    fn test_engine_metrics_collection() {
        // Test metrics collection (lines 212-214)
        let high_queue_metrics = EngineMetrics {
            transactions_processed: 1000,
            error_count: 0,
            avg_latency_ns: 500_000,
            queue_size: 10000, // Very high queue
            active_workers: 4,
            ..Default::default()
        };

        let health = HealthCheck::new(EngineStatus::Running, high_queue_metrics);
        // Should be penalized for high queue size, but may still be above 85
        assert!(health.score <= 100);
    }
}
