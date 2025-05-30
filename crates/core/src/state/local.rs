//! Thread-local state management for TallyIO core
//!
//! This module provides ultra-fast thread-local state for worker threads
//! to minimize synchronization overhead and maximize performance.

use crate::error::{CoreError, CoreResult};
use crate::types::{Opportunity, ProcessingResult, Transaction, TransactionHash};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Thread-local state configuration
#[derive(Debug, Clone)]
pub struct LocalStateConfig {
    /// Maximum number of cached transactions
    pub max_cached_transactions: usize,
    /// Maximum number of cached opportunities
    pub max_cached_opportunities: usize,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Enable statistics collection
    pub enable_statistics: bool,
}

impl Default for LocalStateConfig {
    fn default() -> Self {
        Self {
            max_cached_transactions: 1000,
            max_cached_opportunities: 500,
            cache_ttl: Duration::from_secs(30),
            enable_statistics: true,
        }
    }
}

/// Thread-local state for worker threads
///
/// Provides ultra-fast access to frequently used data without synchronization overhead.
/// Each worker thread maintains its own local state for maximum performance.
#[derive(Debug)]
pub struct LocalState {
    /// Configuration
    config: LocalStateConfig,
    /// Cached transactions
    transactions: RefCell<HashMap<TransactionHash, (Transaction, Instant)>>,
    /// Cached processing results
    results: RefCell<HashMap<TransactionHash, (ProcessingResult, Instant)>>,
    /// Cached opportunities
    opportunities: RefCell<HashMap<String, (Opportunity, Instant)>>,
    /// Local statistics
    stats: RefCell<LocalStateStatistics>,
    /// Creation time
    created_at: Instant,
}

/// Local state statistics
#[derive(Debug, Clone)]
pub struct LocalStateStatistics {
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total operations
    pub total_operations: u64,
    /// Last cleanup time
    pub last_cleanup: Instant,
    /// Memory usage estimate
    pub memory_usage: usize,
}

impl Default for LocalStateStatistics {
    fn default() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            total_operations: 0,
            last_cleanup: Instant::now(),
            memory_usage: 0,
        }
    }
}

impl LocalState {
    /// Create a new thread-local state
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(LocalStateConfig::default())
    }

    /// Create a new thread-local state with configuration
    #[must_use]
    pub fn with_config(config: LocalStateConfig) -> Self {
        Self {
            config,
            transactions: RefCell::new(HashMap::with_capacity(100)),
            results: RefCell::new(HashMap::with_capacity(100)),
            opportunities: RefCell::new(HashMap::with_capacity(50)),
            stats: RefCell::new(LocalStateStatistics::default()),
            created_at: Instant::now(),
        }
    }

    /// Cache a transaction
    pub fn cache_transaction(&self, transaction: Transaction) -> CoreResult<()> {
        let hash = transaction
            .hash
            .ok_or_else(|| CoreError::state("Transaction must have a hash to be cached"))?;

        let mut transactions = self.transactions.borrow_mut();
        transactions.insert(hash, (transaction, Instant::now()));

        // Update statistics
        if self.config.enable_statistics {
            let mut stats = self.stats.borrow_mut();
            stats.total_operations += 1;
            // Don't update memory usage here to avoid borrow conflicts
        }

        // Cleanup if needed
        if transactions.len() > self.config.max_cached_transactions {
            self.cleanup_transactions(&mut transactions);
        }

        Ok(())
    }

    /// Get a cached transaction
    pub fn get_transaction(&self, hash: &TransactionHash) -> Option<Transaction> {
        let transactions = self.transactions.borrow();

        if let Some((transaction, _)) = transactions.get(hash) {
            // Update statistics
            if self.config.enable_statistics {
                let mut stats = self.stats.borrow_mut();
                stats.cache_hits += 1;
                stats.total_operations += 1;
            }
            Some(transaction.clone())
        } else {
            // Update statistics
            if self.config.enable_statistics {
                let mut stats = self.stats.borrow_mut();
                stats.cache_misses += 1;
                stats.total_operations += 1;
            }
            None
        }
    }

    /// Cache a processing result
    pub fn cache_result(&self, result: ProcessingResult) -> CoreResult<()> {
        let hash = result.transaction_hash.ok_or_else(|| {
            CoreError::state("Processing result must have a transaction hash to be cached")
        })?;

        let mut results = self.results.borrow_mut();
        results.insert(hash, (result, Instant::now()));

        // Update statistics
        if self.config.enable_statistics {
            let mut stats = self.stats.borrow_mut();
            stats.total_operations += 1;
            // Don't update memory usage here to avoid borrow conflicts
        }

        Ok(())
    }

    /// Get a cached processing result
    pub fn get_result(&self, hash: &TransactionHash) -> Option<ProcessingResult> {
        let results = self.results.borrow();

        if let Some((result, _)) = results.get(hash) {
            // Update statistics
            if self.config.enable_statistics {
                let mut stats = self.stats.borrow_mut();
                stats.cache_hits += 1;
                stats.total_operations += 1;
            }
            Some(result.clone())
        } else {
            // Update statistics
            if self.config.enable_statistics {
                let mut stats = self.stats.borrow_mut();
                stats.cache_misses += 1;
                stats.total_operations += 1;
            }
            None
        }
    }

    /// Cache an opportunity
    pub fn cache_opportunity(&self, opportunity: Opportunity) -> CoreResult<()> {
        let key = opportunity.id.to_string();

        let mut opportunities = self.opportunities.borrow_mut();
        opportunities.insert(key, (opportunity, Instant::now()));

        // Update statistics
        if self.config.enable_statistics {
            let mut stats = self.stats.borrow_mut();
            stats.total_operations += 1;
            // Don't update memory usage here to avoid borrow conflicts
        }

        // Cleanup if needed
        if opportunities.len() > self.config.max_cached_opportunities {
            self.cleanup_opportunities(&mut opportunities);
        }

        Ok(())
    }

    /// Get cached opportunities
    #[must_use]
    pub fn get_opportunities(&self) -> Vec<Opportunity> {
        let opportunities = self.opportunities.borrow();
        opportunities.values().map(|(opp, _)| opp.clone()).collect()
    }

    /// Get cached opportunities by type
    #[must_use]
    pub fn get_opportunities_by_type(
        &self,
        opp_type: crate::types::OpportunityType,
    ) -> Vec<Opportunity> {
        let opportunities = self.opportunities.borrow();
        opportunities
            .values()
            .filter(|(opp, _)| opp.opportunity_type == opp_type)
            .map(|(opp, _)| opp.clone())
            .collect()
    }

    /// Cleanup expired entries
    pub fn cleanup(&self) -> CoreResult<()> {
        let now = Instant::now();

        // Cleanup transactions
        {
            let mut transactions = self.transactions.borrow_mut();
            self.cleanup_transactions(&mut transactions);
        }

        // Cleanup results
        {
            let mut results = self.results.borrow_mut();
            results
                .retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.config.cache_ttl);
        }

        // Cleanup opportunities
        {
            let mut opportunities = self.opportunities.borrow_mut();
            self.cleanup_opportunities(&mut opportunities);
        }

        // Update statistics
        if self.config.enable_statistics {
            let mut stats = self.stats.borrow_mut();
            stats.last_cleanup = now;
            stats.memory_usage = self.estimate_memory_usage();
        }

        Ok(())
    }

    /// Cleanup transactions
    fn cleanup_transactions(
        &self,
        transactions: &mut HashMap<TransactionHash, (Transaction, Instant)>,
    ) {
        let now = Instant::now();
        transactions
            .retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.config.cache_ttl);

        // If still too many, remove oldest
        if transactions.len() > self.config.max_cached_transactions {
            let mut entries: Vec<_> = transactions.iter().collect();
            entries.sort_by_key(|(_, (_, timestamp))| *timestamp);

            let to_remove = transactions.len() - self.config.max_cached_transactions;
            let keys_to_remove: Vec<_> = entries
                .iter()
                .take(to_remove)
                .map(|(hash, _)| **hash)
                .collect();
            for hash in keys_to_remove {
                transactions.remove(&hash);
            }
        }
    }

    /// Cleanup opportunities
    fn cleanup_opportunities(&self, opportunities: &mut HashMap<String, (Opportunity, Instant)>) {
        let now = Instant::now();
        opportunities
            .retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.config.cache_ttl);

        // If still too many, remove oldest
        if opportunities.len() > self.config.max_cached_opportunities {
            let mut entries: Vec<_> = opportunities.iter().collect();
            entries.sort_by_key(|(_, (_, timestamp))| *timestamp);

            let to_remove = opportunities.len() - self.config.max_cached_opportunities;
            let keys_to_remove: Vec<_> = entries
                .iter()
                .take(to_remove)
                .map(|(key, _)| (*key).clone())
                .collect();
            for key in keys_to_remove {
                opportunities.remove(&key);
            }
        }
    }

    /// Get local state statistics
    #[must_use]
    pub fn statistics(&self) -> LocalStateStatistics {
        if self.config.enable_statistics {
            let mut stats = self.stats.borrow().clone();
            stats.memory_usage = self.estimate_memory_usage();
            stats
        } else {
            LocalStateStatistics::default()
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        let tx_size =
            self.transactions.borrow().len() * std::mem::size_of::<(Transaction, Instant)>();
        let result_size =
            self.results.borrow().len() * std::mem::size_of::<(ProcessingResult, Instant)>();
        let opp_size =
            self.opportunities.borrow().len() * std::mem::size_of::<(Opportunity, Instant)>();

        tx_size + result_size + opp_size
    }

    /// Get cache hit ratio
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f64 {
        if !self.config.enable_statistics {
            return 0.0;
        }

        let stats = self.stats.borrow();
        if stats.total_operations == 0 {
            0.0
        } else {
            stats.cache_hits as f64 / stats.total_operations as f64
        }
    }

    /// Clear all cached data
    pub fn clear(&self) {
        self.transactions.borrow_mut().clear();
        self.results.borrow_mut().clear();
        self.opportunities.borrow_mut().clear();

        if self.config.enable_statistics {
            *self.stats.borrow_mut() = LocalStateStatistics::default();
        }
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &LocalStateConfig {
        &self.config
    }

    /// Get cache sizes
    #[must_use]
    pub fn cache_sizes(&self) -> (usize, usize, usize) {
        (
            self.transactions.borrow().len(),
            self.results.borrow().len(),
            self.opportunities.borrow().len(),
        )
    }

    /// Check if cleanup is needed
    #[must_use]
    pub fn needs_cleanup(&self) -> bool {
        if !self.config.enable_statistics {
            return false;
        }

        let stats = self.stats.borrow();
        let cleanup_interval = Duration::from_secs(60); // Cleanup every minute
        Instant::now().duration_since(stats.last_cleanup) > cleanup_interval
    }
}

impl Default for LocalState {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalStateStatistics {
    /// Get cache miss ratio
    #[must_use]
    pub fn cache_miss_ratio(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            self.cache_misses as f64 / self.total_operations as f64
        }
    }

    /// Get cache hit ratio
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_operations as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Gas, Price};

    #[test]
    fn test_local_state_creation() {
        let state = LocalState::new();
        let (tx_count, result_count, opp_count) = state.cache_sizes();

        assert_eq!(tx_count, 0);
        assert_eq!(result_count, 0);
        assert_eq!(opp_count, 0);
    }

    #[test]
    fn test_transaction_caching() -> CoreResult<()> {
        let state = LocalState::new();

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

        state.cache_transaction(tx.clone())?;

        let retrieved = state.get_transaction(&[1u8; 32]);
        assert!(retrieved.is_some());
        if let Some(retrieved_tx) = retrieved {
            assert_eq!(retrieved_tx.id, tx.id);
        }

        Ok(())
    }

    #[test]
    fn test_opportunity_caching() -> CoreResult<()> {
        let state = LocalState::new();

        let opportunity = Opportunity::new(
            crate::types::OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        state.cache_opportunity(opportunity.clone())?;

        let opportunities = state.get_opportunities();
        assert_eq!(opportunities.len(), 1);
        assert_eq!(opportunities[0].id, opportunity.id);

        Ok(())
    }

    #[test]
    fn test_cache_statistics() -> CoreResult<()> {
        let state = LocalState::new();

        // Cache a transaction
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

        state.cache_transaction(tx)?;

        // Get it (cache hit)
        let _retrieved = state.get_transaction(&[1u8; 32]);

        // Try to get non-existent (cache miss)
        let _missing = state.get_transaction(&[2u8; 32]);

        let stats = state.statistics();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.total_operations, 3); // cache + hit + miss

        Ok(())
    }

    #[test]
    fn test_cache_cleanup() -> CoreResult<()> {
        let state = LocalState::new();

        // Add some test data
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

        state.cache_transaction(tx)?;

        // Perform cleanup
        state.cleanup()?;

        // Should not crash
        Ok(())
    }

    #[test]
    fn test_cache_hit_ratio() -> CoreResult<()> {
        let state = LocalState::new();

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

        state.cache_transaction(tx)?;

        // Multiple hits
        for _ in 0..5 {
            let _retrieved = state.get_transaction(&[1u8; 32]);
        }

        let hit_ratio = state.cache_hit_ratio();
        assert!(hit_ratio > 0.0);

        Ok(())
    }

    #[test]
    fn test_get_nonexistent_items() -> CoreResult<()> {
        let state = LocalState::new();

        // Try to get items that don't exist
        let tx = state.get_transaction(&[99u8; 32]);
        assert!(tx.is_none());

        let result = state.get_result(&[99u8; 32]);
        assert!(result.is_none());

        let opportunities = state.get_opportunities();
        assert!(opportunities.is_empty());

        Ok(())
    }

    #[test]
    fn test_local_state_default() {
        let state = LocalState::default();
        let (tx_count, result_count, opp_count) = state.cache_sizes();

        assert_eq!(tx_count, 0);
        assert_eq!(result_count, 0);
        assert_eq!(opp_count, 0);
    }

    #[test]
    fn test_cache_result_operations() -> CoreResult<()> {
        let state = LocalState::new();

        let result = ProcessingResult::success(
            [1u8; 32],
            std::time::Duration::from_micros(500),
            crate::types::Gas::new(21_000),
            crate::types::Price::from_gwei(20),
        );
        state.cache_result(result.clone())?;

        let retrieved = state.get_result(&[1u8; 32]);
        assert!(retrieved.is_some());
        if let Some(retrieved_result) = retrieved {
            assert_eq!(retrieved_result.id, result.id);
        }

        Ok(())
    }

    #[test]
    fn test_opportunity_types() -> CoreResult<()> {
        let state = LocalState::new();

        let arbitrage = Opportunity::new(
            crate::types::OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        let sandwich = Opportunity::new(
            crate::types::OpportunityType::Sandwich,
            Price::from_ether(2),
            Gas::new(200_000),
        );

        state.cache_opportunity(arbitrage.clone())?;
        state.cache_opportunity(sandwich.clone())?;

        let arbitrage_opps =
            state.get_opportunities_by_type(crate::types::OpportunityType::Arbitrage);
        assert_eq!(arbitrage_opps.len(), 1);
        assert_eq!(arbitrage_opps[0].id, arbitrage.id);

        let sandwich_opps =
            state.get_opportunities_by_type(crate::types::OpportunityType::Sandwich);
        assert_eq!(sandwich_opps.len(), 1);
        assert_eq!(sandwich_opps[0].id, sandwich.id);

        Ok(())
    }

    #[test]
    fn test_cache_clear() -> CoreResult<()> {
        let state = LocalState::new();

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
        state.cache_transaction(tx)?;

        let opportunity = Opportunity::new(
            crate::types::OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );
        state.cache_opportunity(opportunity)?;

        // Clear cache
        state.clear();

        // Verify everything is cleared
        let tx_retrieved = state.get_transaction(&[1u8; 32]);
        assert!(tx_retrieved.is_none());

        let opportunities = state.get_opportunities();
        assert!(opportunities.is_empty());

        Ok(())
    }

    #[test]
    fn test_needs_cleanup() {
        let state = LocalState::new();

        // Initially should not need cleanup
        assert!(!state.needs_cleanup());
    }

    #[test]
    fn test_config_access() {
        let config = LocalStateConfig {
            max_cached_transactions: 500,
            max_cached_opportunities: 250,
            cache_ttl: Duration::from_secs(60),
            enable_statistics: false,
        };

        let state = LocalState::with_config(config.clone());
        let retrieved_config = state.config();

        assert_eq!(retrieved_config.max_cached_transactions, 500);
        assert_eq!(retrieved_config.max_cached_opportunities, 250);
        assert_eq!(retrieved_config.cache_ttl, Duration::from_secs(60));
        assert!(!retrieved_config.enable_statistics);
    }

    #[test]
    fn test_cache_transaction_without_hash() -> CoreResult<()> {
        // Test caching transaction without hash (line 120-121)
        let state = LocalState::new();

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        // Don't set hash - should return error

        let result = state.cache_transaction(tx);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_cache_result_without_hash() -> CoreResult<()> {
        // Test caching result without hash (line 152-153)
        let state = LocalState::new();

        let mut result = ProcessingResult::success(
            [1u8; 32],
            std::time::Duration::from_micros(500),
            crate::types::Gas::new(21_000),
            crate::types::Price::from_gwei(20),
        );
        result.transaction_hash = None; // Remove hash

        let cache_result = state.cache_result(result);
        assert!(cache_result.is_err());

        Ok(())
    }

    #[test]
    fn test_cleanup_transactions_overflow() -> CoreResult<()> {
        // Test cleanup when too many transactions (lines 278-291)
        let mut config = LocalStateConfig::default();
        config.max_cached_transactions = 2; // Very small limit
        let state = LocalState::with_config(config);

        // Add more transactions than the limit
        for i in 0..5 {
            let mut tx = Transaction::new(
                [i as u8; 20],
                Some([2u8; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(21_000),
                0,
                Vec::with_capacity(0),
            );
            tx.set_hash([i as u8; 32]);
            state.cache_transaction(tx)?;
        }

        let (tx_count, _, _) = state.cache_sizes();
        assert!(tx_count <= 2); // Should be limited

        Ok(())
    }

    #[test]
    fn test_cleanup_opportunities_overflow() -> CoreResult<()> {
        // Test cleanup when too many opportunities (lines 301-314)
        let mut config = LocalStateConfig::default();
        config.max_cached_opportunities = 2; // Very small limit
        let state = LocalState::with_config(config);

        // Add more opportunities than the limit
        for _ in 0..5 {
            let opportunity = Opportunity::new(
                crate::types::OpportunityType::Arbitrage,
                Price::from_ether(1),
                Gas::new(100_000),
            );
            state.cache_opportunity(opportunity)?;
        }

        let (_, _, opp_count) = state.cache_sizes();
        assert!(opp_count <= 2); // Should be limited

        Ok(())
    }

    #[test]
    fn test_statistics_disabled() {
        // Test statistics when disabled (line 324-325)
        let mut config = LocalStateConfig::default();
        config.enable_statistics = false;
        let state = LocalState::with_config(config);

        let stats = state.statistics();
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_cache_hit_ratio_disabled() {
        // Test cache hit ratio when statistics disabled (line 344-345)
        let mut config = LocalStateConfig::default();
        config.enable_statistics = false;
        let state = LocalState::with_config(config);

        let ratio = state.cache_hit_ratio();
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_cache_hit_ratio_no_operations() {
        // Test cache hit ratio with no operations (line 349-350)
        let state = LocalState::new();

        let ratio = state.cache_hit_ratio();
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_needs_cleanup_disabled() {
        // Test needs cleanup when statistics disabled (line 386-387)
        let mut config = LocalStateConfig::default();
        config.enable_statistics = false;
        let state = LocalState::with_config(config);

        let needs = state.needs_cleanup();
        assert!(!needs);
    }

    #[test]
    fn test_cache_miss_ratio() {
        // Test cache miss ratio calculation (lines 405-410)
        let stats = LocalStateStatistics {
            cache_hits: 3,
            cache_misses: 7,
            total_operations: 10,
            memory_usage: 1024,
            last_cleanup: Instant::now(),
        };

        let miss_ratio = stats.cache_miss_ratio();
        assert_eq!(miss_ratio, 0.7);

        // Test with zero operations
        let empty_stats = LocalStateStatistics::default();
        let empty_ratio = empty_stats.cache_miss_ratio();
        assert_eq!(empty_ratio, 0.0);
    }

    #[test]
    fn test_cache_hit_ratio_stats() {
        // Test cache hit ratio calculation (lines 415-420)
        let stats = LocalStateStatistics {
            cache_hits: 3,
            cache_misses: 7,
            total_operations: 10,
            memory_usage: 1024,
            last_cleanup: Instant::now(),
        };

        let hit_ratio = stats.cache_hit_ratio();
        assert_eq!(hit_ratio, 0.3);

        // Test with zero operations
        let empty_stats = LocalStateStatistics::default();
        let empty_ratio = empty_stats.cache_hit_ratio();
        assert_eq!(empty_ratio, 0.0);
    }
}
