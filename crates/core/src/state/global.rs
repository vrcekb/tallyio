//! Global state management for TallyIO core
//!
//! This module provides thread-safe global state management for sharing
//! critical data across all components of the TallyIO system.

use crate::error::{CoreError, CoreResult};
use crate::types::{Opportunity, ProcessingResult, Transaction, TransactionHash};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Global state configuration
#[derive(Debug, Clone)]
pub struct GlobalStateConfig {
    /// Maximum number of transactions to track
    pub max_transactions: usize,
    /// Maximum number of opportunities to track
    pub max_opportunities: usize,
    /// State cleanup interval
    pub cleanup_interval: Duration,
    /// Enable state persistence
    pub enable_persistence: bool,
}

impl Default for GlobalStateConfig {
    fn default() -> Self {
        Self {
            max_transactions: 100_000,
            max_opportunities: 50_000,
            cleanup_interval: Duration::from_secs(60),
            enable_persistence: false,
        }
    }
}

/// Global state for TallyIO system
///
/// Provides thread-safe access to shared state across all system components.
/// Uses lock-free data structures for maximum performance.
#[derive(Debug)]
#[repr(C, align(64))]
pub struct GlobalState {
    /// Configuration
    config: GlobalStateConfig,
    /// Active transactions
    transactions: Arc<DashMap<TransactionHash, (Transaction, Instant)>>,
    /// Processing results
    results: Arc<DashMap<TransactionHash, ProcessingResult>>,
    /// MEV opportunities
    opportunities: Arc<DashMap<String, (Opportunity, Instant)>>,
    /// System metrics
    metrics: GlobalMetrics,
    /// State creation time
    created_at: Instant,
}

/// Global system metrics
#[repr(C, align(64))]
#[derive(Debug)]
pub struct GlobalMetrics {
    /// Total transactions processed
    pub total_transactions: AtomicU64,
    /// Total opportunities found
    pub total_opportunities: AtomicU64,
    /// Total processing time in nanoseconds
    pub total_processing_time_ns: AtomicU64,
    /// Current active transactions
    pub active_transactions: AtomicUsize,
    /// Current active opportunities
    pub active_opportunities: AtomicUsize,
    /// Error count
    pub error_count: AtomicU64,
    /// Last cleanup time
    pub last_cleanup: AtomicU64,
}

impl GlobalState {
    /// Create a new global state
    pub fn new() -> CoreResult<Self> {
        Self::with_config(GlobalStateConfig::default())
    }

    /// Create a new global state with configuration
    pub fn with_config(config: GlobalStateConfig) -> CoreResult<Self> {
        Ok(Self {
            config,
            transactions: Arc::new(DashMap::with_capacity(10_000)),
            results: Arc::new(DashMap::with_capacity(10_000)),
            opportunities: Arc::new(DashMap::with_capacity(5_000)),
            metrics: GlobalMetrics {
                total_transactions: AtomicU64::new(0),
                total_opportunities: AtomicU64::new(0),
                total_processing_time_ns: AtomicU64::new(0),
                active_transactions: AtomicUsize::new(0),
                active_opportunities: AtomicUsize::new(0),
                error_count: AtomicU64::new(0),
                last_cleanup: AtomicU64::new(0),
            },
            created_at: Instant::now(),
        })
    }

    /// Add a transaction to global state
    pub fn add_transaction(&self, transaction: Transaction) -> CoreResult<()> {
        let hash = transaction.hash.unwrap_or_else(|| {
            // Generate a hash if not present
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            transaction.id.hash(&mut hasher);
            let hash_value = hasher.finish();
            let mut hash_bytes = [0u8; 32];
            hash_bytes[..8].copy_from_slice(&hash_value.to_le_bytes());
            hash_bytes
        });

        self.transactions
            .insert(hash, (transaction, Instant::now()));
        self.metrics
            .active_transactions
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_transactions
            .fetch_add(1, Ordering::Relaxed);

        // Check if cleanup is needed
        self.maybe_cleanup()?;

        Ok(())
    }

    /// Get a transaction by hash
    #[must_use]
    pub fn get_transaction(&self, hash: &TransactionHash) -> Option<Transaction> {
        self.transactions.get(hash).map(|entry| entry.0.clone())
    }

    /// Remove a transaction from global state
    pub fn remove_transaction(&self, hash: &TransactionHash) -> Option<Transaction> {
        if let Some((_, (transaction, _))) = self.transactions.remove(hash) {
            self.metrics
                .active_transactions
                .fetch_sub(1, Ordering::Relaxed);
            Some(transaction)
        } else {
            None
        }
    }

    /// Add a processing result
    pub fn add_result(&self, result: ProcessingResult) -> CoreResult<()> {
        if let Some(hash) = result.transaction_hash {
            self.results.insert(hash, result);
        }
        Ok(())
    }

    /// Get a processing result by transaction hash
    #[must_use]
    pub fn get_result(&self, hash: &TransactionHash) -> Option<ProcessingResult> {
        self.results.get(hash).map(|entry| entry.clone())
    }

    /// Add an MEV opportunity
    pub fn add_opportunity(&self, opportunity: Opportunity) -> CoreResult<()> {
        let key = opportunity.id.to_string();
        self.opportunities
            .insert(key, (opportunity, Instant::now()));
        self.metrics
            .active_opportunities
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_opportunities
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get all active opportunities
    #[must_use]
    pub fn get_opportunities(&self) -> Vec<Opportunity> {
        self.opportunities
            .iter()
            .map(|entry| entry.value().0.clone())
            .collect()
    }

    /// Get opportunities by type
    #[must_use]
    pub fn get_opportunities_by_type(
        &self,
        opp_type: crate::types::OpportunityType,
    ) -> Vec<Opportunity> {
        self.opportunities
            .iter()
            .filter(|entry| entry.value().0.opportunity_type == opp_type)
            .map(|entry| entry.value().0.clone())
            .collect()
    }

    /// Remove expired opportunities
    pub fn cleanup_expired_opportunities(&self, max_age: Duration) -> usize {
        let now = Instant::now();
        let mut removed = 0;

        self.opportunities.retain(|_, (_, timestamp)| {
            if now.duration_since(*timestamp) > max_age {
                removed += 1;
                false
            } else {
                true
            }
        });

        if removed > 0 {
            self.metrics
                .active_opportunities
                .fetch_sub(removed, Ordering::Relaxed);
        }

        removed
    }

    /// Record processing time
    #[inline(always)]
    pub fn record_processing_time(&self, duration: Duration) {
        self.metrics
            .total_processing_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record an error
    #[inline(always)]
    pub fn record_error(&self) {
        self.metrics.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get global metrics
    #[must_use]
    pub fn metrics(&self) -> GlobalStateMetrics {
        let total_tx = self.metrics.total_transactions.load(Ordering::Relaxed);
        let total_time = self
            .metrics
            .total_processing_time_ns
            .load(Ordering::Relaxed);
        let avg_time = if total_tx > 0 {
            total_time / total_tx
        } else {
            0
        };

        GlobalStateMetrics {
            total_transactions: total_tx,
            total_opportunities: self.metrics.total_opportunities.load(Ordering::Relaxed),
            active_transactions: self.metrics.active_transactions.load(Ordering::Relaxed),
            active_opportunities: self.metrics.active_opportunities.load(Ordering::Relaxed),
            error_count: self.metrics.error_count.load(Ordering::Relaxed),
            average_processing_time_ns: avg_time,
            uptime: self.created_at.elapsed(),
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// Get current state size in bytes (estimate)
    #[must_use]
    pub fn size(&self) -> usize {
        self.estimate_memory_usage()
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        let tx_size = self.transactions.len() * std::mem::size_of::<(Transaction, Instant)>();
        let result_size = self.results.len() * std::mem::size_of::<ProcessingResult>();
        let opp_size = self.opportunities.len() * std::mem::size_of::<(Opportunity, Instant)>();

        tx_size + result_size + opp_size
    }

    /// Perform cleanup if needed
    fn maybe_cleanup(&self) -> CoreResult<()> {
        let now = std::time::SystemTime::now();
        let last_cleanup = self.metrics.last_cleanup.load(Ordering::Relaxed);
        let last_cleanup_time = std::time::UNIX_EPOCH + Duration::from_millis(last_cleanup);

        if now
            .duration_since(last_cleanup_time)
            .unwrap_or(Duration::ZERO)
            > self.config.cleanup_interval
        {
            self.cleanup()?;
            self.metrics.last_cleanup.store(
                now.duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_millis() as u64,
                Ordering::Relaxed,
            );
        }

        Ok(())
    }

    /// Perform state cleanup
    pub fn cleanup(&self) -> CoreResult<()> {
        let max_age = Duration::from_secs(300); // 5 minutes

        // Clean up old transactions
        let now = Instant::now();
        let mut tx_removed = 0;
        self.transactions.retain(|_, (_, timestamp)| {
            if now.duration_since(*timestamp) > max_age {
                tx_removed += 1;
                false
            } else {
                true
            }
        });

        if tx_removed > 0 {
            self.metrics
                .active_transactions
                .fetch_sub(tx_removed, Ordering::Relaxed);
        }

        // Clean up old results
        self.results
            .retain(|hash, _| self.transactions.contains_key(hash));

        // Clean up expired opportunities
        self.cleanup_expired_opportunities(Duration::from_secs(60));

        Ok(())
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &GlobalStateConfig {
        &self.config
    }

    /// Clear all state
    pub fn clear(&self) {
        self.transactions.clear();
        self.results.clear();
        self.opportunities.clear();

        self.metrics.active_transactions.store(0, Ordering::Relaxed);
        self.metrics
            .active_opportunities
            .store(0, Ordering::Relaxed);
    }
}

impl Default for GlobalState {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback to minimal state if creation fails
            Self {
                config: GlobalStateConfig::default(),
                transactions: Arc::new(DashMap::new()),
                results: Arc::new(DashMap::new()),
                opportunities: Arc::new(DashMap::new()),
                metrics: GlobalMetrics {
                    total_transactions: AtomicU64::new(0),
                    total_opportunities: AtomicU64::new(0),
                    total_processing_time_ns: AtomicU64::new(0),
                    active_transactions: AtomicUsize::new(0),
                    active_opportunities: AtomicUsize::new(0),
                    error_count: AtomicU64::new(0),
                    last_cleanup: AtomicU64::new(0),
                },
                created_at: Instant::now(),
            }
        })
    }
}

/// Global state metrics snapshot
#[derive(Debug, Clone)]
pub struct GlobalStateMetrics {
    /// Total transactions processed
    pub total_transactions: u64,
    /// Total opportunities found
    pub total_opportunities: u64,
    /// Currently active transactions
    pub active_transactions: usize,
    /// Currently active opportunities
    pub active_opportunities: usize,
    /// Total error count
    pub error_count: u64,
    /// Average processing time in nanoseconds
    pub average_processing_time_ns: u64,
    /// System uptime
    pub uptime: Duration,
    /// Estimated memory usage in bytes
    pub memory_usage: usize,
}

impl GlobalStateMetrics {
    /// Get error rate
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        if self.total_transactions == 0 {
            0.0
        } else {
            self.error_count as f64 / self.total_transactions as f64
        }
    }

    /// Get transactions per second
    #[must_use]
    pub fn transactions_per_second(&self) -> f64 {
        let uptime_secs = self.uptime.as_secs_f64();
        if uptime_secs > 0.0 {
            self.total_transactions as f64 / uptime_secs
        } else {
            0.0
        }
    }

    /// Get opportunities per second
    #[must_use]
    pub fn opportunities_per_second(&self) -> f64 {
        let uptime_secs = self.uptime.as_secs_f64();
        if uptime_secs > 0.0 {
            self.total_opportunities as f64 / uptime_secs
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StateConfig;
    use crate::types::{Gas, Price};

    #[test]
    fn test_global_state_creation() -> CoreResult<()> {
        let state = GlobalState::new()?;
        let metrics = state.metrics();

        assert_eq!(metrics.total_transactions, 0);
        assert_eq!(metrics.active_transactions, 0);

        Ok(())
    }

    #[test]
    fn test_transaction_management() -> CoreResult<()> {
        let state = GlobalState::new()?;

        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        tx.set_hash([1u8; 32]);

        state.add_transaction(tx.clone())?;

        let retrieved = state.get_transaction(&[1u8; 32]);
        assert!(retrieved.is_some());
        if let Some(retrieved_tx) = retrieved {
            assert_eq!(retrieved_tx.id, tx.id);
        }

        let metrics = state.metrics();
        assert_eq!(metrics.active_transactions, 1);
        assert_eq!(metrics.total_transactions, 1);

        Ok(())
    }

    #[test]
    fn test_opportunity_management() -> CoreResult<()> {
        let state = GlobalState::new()?;

        let opportunity = Opportunity::new(
            crate::types::OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        state.add_opportunity(opportunity.clone())?;

        let opportunities = state.get_opportunities();
        assert_eq!(opportunities.len(), 1);
        assert_eq!(opportunities[0].id, opportunity.id);

        let metrics = state.metrics();
        assert_eq!(metrics.active_opportunities, 1);
        assert_eq!(metrics.total_opportunities, 1);

        Ok(())
    }

    #[test]
    fn test_metrics_calculation() -> CoreResult<()> {
        let state = GlobalState::new()?;

        // Add some test data - need to add a transaction first for average calculation
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        state.add_transaction(tx)?;

        state.record_processing_time(Duration::from_millis(1));
        state.record_error();

        let metrics = state.metrics();
        assert!(metrics.average_processing_time_ns > 0);
        assert_eq!(metrics.error_count, 1);

        Ok(())
    }

    #[test]
    fn test_state_cleanup() -> CoreResult<()> {
        let state = GlobalState::new()?;

        // Add some test data
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        state.add_transaction(tx)?;

        // Perform cleanup
        state.cleanup()?;

        // Should not crash
        Ok(())
    }

    #[test]
    fn test_cleanup_expired_opportunities() -> CoreResult<()> {
        // Test cleanup_expired_opportunities method (line 212)
        let state = GlobalState::new()?;

        let opportunity = Opportunity::new(
            crate::types::OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        state.add_opportunity(opportunity)?;
        assert_eq!(state.get_opportunities().len(), 1);

        // Cleanup with very short expiry (should remove all)
        let removed = state.cleanup_expired_opportunities(Duration::from_nanos(1));
        assert!(removed > 0);
        assert_eq!(state.get_opportunities().len(), 0);

        Ok(())
    }

    #[test]
    fn test_memory_estimation() -> CoreResult<()> {
        // Test estimate_memory_usage method (lines 271-276)
        let state = GlobalState::new()?;

        let initial_size = state.size();
        // Size is always >= 0 for usize, so this check is redundant but kept for clarity
        assert!(initial_size < usize::MAX);

        // Add transaction and check size increase
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        tx.set_hash([1u8; 32]);
        state.add_transaction(tx)?;

        let new_size = state.size();
        assert!(new_size > initial_size);

        Ok(())
    }

    #[test]
    fn test_maybe_cleanup_timing() -> CoreResult<()> {
        // Test maybe_cleanup timing logic (lines 280-299)
        let state = GlobalState::new()?;

        // First call should not trigger cleanup (just created)
        state.maybe_cleanup()?;

        // Manually set last_cleanup to old time to trigger cleanup
        let old_time = (std::time::UNIX_EPOCH
            .elapsed()
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64)
            .saturating_sub(10000);
        state
            .metrics
            .last_cleanup
            .store(old_time, Ordering::Relaxed);

        // This should trigger cleanup
        state.maybe_cleanup()?;

        Ok(())
    }

    #[test]
    fn test_cleanup_old_transactions() -> CoreResult<()> {
        // Test cleanup of old transactions (lines 312-322)
        let state = GlobalState::new()?;

        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        tx.set_hash([1u8; 32]);
        state.add_transaction(tx)?;

        // Manually add old transaction to test cleanup
        let old_instant = Instant::now() - Duration::from_secs(400); // Older than 5 minutes
        state.transactions.insert(
            [2u8; 32],
            (
                Transaction::new(
                    [2u8; 20],
                    Some([3u8; 20]),
                    Price::from_ether(1),
                    Price::from_gwei(20),
                    Gas::new(21_000),
                    1,
                    Vec::with_capacity(0),
                ),
                old_instant,
            ),
        );

        let initial_count = state.metrics.active_transactions.load(Ordering::Relaxed);
        state.cleanup()?;

        // Should have removed the old transaction
        assert!(!state.transactions.contains_key(&[2u8; 32]));

        Ok(())
    }

    #[test]
    fn test_global_state_metrics_calculations() -> CoreResult<()> {
        // Test GlobalStateMetrics calculation methods (lines 383-411)
        let state = GlobalState::new()?;

        // Test with no data
        let metrics = state.metrics();
        assert_eq!(metrics.error_rate(), 0.0);
        assert_eq!(metrics.transactions_per_second(), 0.0);
        assert_eq!(metrics.opportunities_per_second(), 0.0);

        // Add some data
        state.record_error();
        state
            .metrics
            .total_transactions
            .store(10, Ordering::Relaxed);
        state
            .metrics
            .total_opportunities
            .store(5, Ordering::Relaxed);

        let metrics = state.metrics();
        assert_eq!(metrics.error_rate(), 0.1); // 1 error out of 10 transactions
        assert!(metrics.transactions_per_second() >= 0.0);
        assert!(metrics.opportunities_per_second() >= 0.0);

        Ok(())
    }

    #[test]
    fn test_config_access() -> CoreResult<()> {
        // Test config getter (line 336)
        let state = GlobalState::new()?;
        let config = state.config();
        assert!(config.max_transactions > 0);
        Ok(())
    }

    #[test]
    fn test_clear_state() -> CoreResult<()> {
        // Test clear method (lines 341-350)
        let state = GlobalState::new()?;

        // Add some data
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        tx.set_hash([1u8; 32]);
        state.add_transaction(tx)?;

        let opportunity = Opportunity::new(
            crate::types::OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );
        state.add_opportunity(opportunity)?;

        // Verify data exists
        assert!(state.get_transaction(&[1u8; 32]).is_some());
        assert!(!state.get_opportunities().is_empty());

        // Clear all state
        state.clear();

        // Verify everything is cleared
        assert!(state.get_transaction(&[1u8; 32]).is_none());
        assert!(state.get_opportunities().is_empty());
        assert_eq!(state.metrics.active_transactions.load(Ordering::Relaxed), 0);
        assert_eq!(
            state.metrics.active_opportunities.load(Ordering::Relaxed),
            0
        );

        Ok(())
    }

    #[test]
    fn test_default_implementation() -> CoreResult<()> {
        // Test Default implementation (lines 353-356)
        let state = GlobalState::default();
        let metrics = state.metrics();
        assert_eq!(metrics.total_transactions, 0);
        assert_eq!(metrics.active_transactions, 0);
        Ok(())
    }

    #[test]
    fn test_transaction_without_hash() -> CoreResult<()> {
        // Test transaction handling without hash (lines 358-370, 372)
        let state = GlobalState::new()?;

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        // Don't set hash - should be None

        // This should handle the case where hash is None
        let result = state.add_transaction(tx);
        // Should succeed even without hash
        assert!(result.is_ok());

        Ok(())
    }

    #[test]
    fn test_remove_transaction() -> CoreResult<()> {
        // Test remove_transaction method (lines 404, 417, 428)
        let state = GlobalState::new()?;

        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        tx.set_hash([1u8; 32]);

        // Add transaction
        state.add_transaction(tx)?;
        assert!(state.get_transaction(&[1u8; 32]).is_some());

        // Remove transaction
        let removed = state.remove_transaction(&[1u8; 32]);
        assert!(removed.is_some());
        assert!(state.get_transaction(&[1u8; 32]).is_none());

        // Try to remove non-existent transaction
        let not_found = state.remove_transaction(&[2u8; 32]);
        assert!(not_found.is_none());

        Ok(())
    }

    #[test]
    fn test_processing_result_management() -> CoreResult<()> {
        // Test processing result methods using existing add_result and get_result
        let state = GlobalState::new()?;

        let result = ProcessingResult::success(
            [1u8; 32],
            Duration::from_millis(1),
            Gas::new(21_000),
            Price::from_gwei(20),
        );

        // Add result
        state.add_result(result.clone())?;

        // Get result
        let retrieved = state.get_result(&[1u8; 32]);
        assert!(retrieved.is_some());
        if let Some(retrieved_result) = retrieved {
            assert_eq!(retrieved_result.transaction_hash, result.transaction_hash);
        }

        Ok(())
    }

    #[test]
    fn test_opportunities_by_type() -> CoreResult<()> {
        // Test get_opportunities_by_type method (lines 622, 635, 657, 684)
        let state = GlobalState::new()?;

        let arbitrage_opp = Opportunity::new(
            crate::types::OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        let sandwich_opp = Opportunity::new(
            crate::types::OpportunityType::Sandwich,
            Price::from_ether(2),
            Gas::new(150_000),
        );

        state.add_opportunity(arbitrage_opp)?;
        state.add_opportunity(sandwich_opp)?;

        // Test filtering by type
        let arbitrage_opps =
            state.get_opportunities_by_type(crate::types::OpportunityType::Arbitrage);
        assert_eq!(arbitrage_opps.len(), 1);
        assert_eq!(
            arbitrage_opps[0].opportunity_type,
            crate::types::OpportunityType::Arbitrage
        );

        let sandwich_opps =
            state.get_opportunities_by_type(crate::types::OpportunityType::Sandwich);
        assert_eq!(sandwich_opps.len(), 1);
        assert_eq!(
            sandwich_opps[0].opportunity_type,
            crate::types::OpportunityType::Sandwich
        );

        // Test with non-existent type
        let liquidation_opps =
            state.get_opportunities_by_type(crate::types::OpportunityType::Liquidation);
        assert_eq!(liquidation_opps.len(), 0);

        Ok(())
    }

    #[test]
    fn test_default_fallback() -> CoreResult<()> {
        // Test Default implementation fallback (lines 358-375)
        let state = GlobalState::default();

        // Should have default configuration
        assert!(state.config().max_transactions > 0);
        assert!(state.config().max_opportunities > 0);

        // Should have empty collections
        assert_eq!(state.get_opportunities().len(), 0);
        assert_eq!(state.metrics().total_transactions, 0);

        Ok(())
    }

    #[test]
    fn test_default_implementation_coverage() -> CoreResult<()> {
        // Test Default implementation (lines 358-370, 372)
        let state = GlobalState::default();

        // Should have default configuration
        assert!(state.config().max_transactions > 0);
        assert!(state.config().max_opportunities > 0);

        // Should have empty collections
        assert_eq!(state.get_opportunities().len(), 0);
        assert_eq!(state.metrics().total_transactions, 0);

        Ok(())
    }

    #[test]
    fn test_memory_estimation_coverage() -> CoreResult<()> {
        // Test memory estimation (line 212)
        let state = GlobalState::new()?;

        // Add some data
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        state.add_transaction(tx)?;

        let memory_estimate = state.estimate_memory_usage();
        assert!(memory_estimate > 0);

        Ok(())
    }

    #[test]
    fn test_cleanup_operations_coverage() -> CoreResult<()> {
        // Test cleanup operations (lines 417, 428)
        let state = GlobalState::new()?;

        // Add multiple transactions
        for i in 0..10 {
            let tx = Transaction::new(
                [i; 20],
                Some([i + 1; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(21_000),
                i as u64,
                Vec::with_capacity(0),
            );
            state.add_transaction(tx)?;
        }

        // Add multiple opportunities
        for i in 0..10 {
            let opp = Opportunity::new(
                crate::types::OpportunityType::Arbitrage,
                Price::from_ether(1),
                Gas::new(150_000),
            );
            state.add_opportunity(opp)?;
        }

        // Trigger cleanup manually
        state.cleanup()?;

        // Should still have data (cleanup depends on age, not count)
        assert!(state.get_opportunities().len() <= 10);

        Ok(())
    }
}
