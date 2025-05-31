//! Mempool monitoring for TallyIO core
//!
//! This module provides ultra-high performance mempool monitoring and transaction analysis
//! for MEV opportunity detection and transaction filtering.

pub mod analyzer;
pub mod filter;
pub mod watcher;

// Re-export main types
pub use analyzer::{MempoolAnalyzer, TransactionAnalysis};
pub use filter::{MempoolFilter, TransactionFilter};
pub use watcher::{MempoolEvent, MempoolWatcher};

use crate::error::CoreResult;
use crate::types::{Transaction, TransactionHash};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Mempool configuration
#[derive(Debug, Clone)]
pub struct MempoolConfig {
    /// Maximum number of transactions to track
    pub max_transactions: usize,
    /// Transaction TTL in mempool
    pub transaction_ttl: Duration,
    /// Enable real-time analysis
    pub enable_realtime_analysis: bool,
    /// Analysis batch size
    pub analysis_batch_size: usize,
    /// Filter configuration
    pub filter_config: FilterConfig,
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_transactions: 100_000,
            transaction_ttl: Duration::from_secs(300), // 5 minutes
            enable_realtime_analysis: true,
            analysis_batch_size: 100,
            filter_config: FilterConfig::default(),
        }
    }
}

/// Filter configuration
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Minimum gas price in gwei
    pub min_gas_price_gwei: u64,
    /// Maximum gas price in gwei
    pub max_gas_price_gwei: u64,
    /// Minimum transaction value in wei
    pub min_value_wei: u64,
    /// Enable DeFi transaction filtering
    pub enable_defi_filter: bool,
    /// Enable MEV opportunity filtering
    pub enable_mev_filter: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_gas_price_gwei: 1,
            max_gas_price_gwei: 1000,
            min_value_wei: 0,
            enable_defi_filter: true,
            enable_mev_filter: true,
        }
    }
}

/// Mempool statistics
#[derive(Debug, Clone)]
pub struct MempoolStatistics {
    /// Total transactions seen
    pub total_transactions: u64,
    /// Current mempool size
    pub current_size: usize,
    /// Transactions per second
    pub transactions_per_second: f64,
    /// Average transaction value
    pub avg_transaction_value: u64,
    /// Average gas price
    pub avg_gas_price: u64,
    /// DeFi transaction count
    pub defi_transactions: u64,
    /// MEV opportunities found
    pub mev_opportunities: u64,
    /// Filtered transactions
    pub filtered_transactions: u64,
}

impl Default for MempoolStatistics {
    fn default() -> Self {
        Self {
            total_transactions: 0,
            current_size: 0,
            transactions_per_second: 0.0,
            avg_transaction_value: 0,
            avg_gas_price: 0,
            defi_transactions: 0,
            mev_opportunities: 0,
            filtered_transactions: 0,
        }
    }
}

/// Mempool manager
///
/// Coordinates mempool watching, filtering, and analysis for ultra-high performance
/// MEV opportunity detection.
#[repr(C, align(64))]
pub struct MempoolManager {
    /// Configuration
    config: MempoolConfig,
    /// Watcher component
    watcher: Option<MempoolWatcher>,
    /// Filter component
    filter: Option<MempoolFilter>,
    /// Analyzer component
    analyzer: Option<MempoolAnalyzer>,
    /// Transaction cache
    transactions: HashMap<TransactionHash, (Transaction, Instant)>,
    /// Statistics counters
    total_transactions: AtomicU64,
    defi_transactions: AtomicU64,
    mev_opportunities: AtomicU64,
    filtered_transactions: AtomicU64,
    /// Start time for TPS calculation
    start_time: Option<Instant>,
}

impl MempoolManager {
    /// Create a new mempool manager
    pub fn new(config: MempoolConfig) -> Self {
        Self {
            config,
            watcher: None,
            filter: None,
            analyzer: None,
            transactions: HashMap::with_capacity(10_000),
            total_transactions: AtomicU64::new(0),
            defi_transactions: AtomicU64::new(0),
            mev_opportunities: AtomicU64::new(0),
            filtered_transactions: AtomicU64::new(0),
            start_time: None,
        }
    }

    /// Start the mempool manager
    pub fn start(&mut self) -> CoreResult<()> {
        self.start_time = Some(Instant::now());

        // Initialize components
        self.watcher = Some(MempoolWatcher::new());
        self.filter = Some(MempoolFilter::new(self.config.filter_config.clone()));

        // Only initialize analyzer if real-time analysis is enabled
        if self.config.enable_realtime_analysis {
            self.analyzer = Some(MempoolAnalyzer::new());
        }

        Ok(())
    }

    /// Stop the mempool manager
    pub fn stop(&mut self) -> CoreResult<()> {
        self.watcher = None;
        self.filter = None;
        self.analyzer = None;
        Ok(())
    }

    /// Process a new transaction
    pub fn process_transaction(&mut self, transaction: Transaction) -> CoreResult<()> {
        let tx_hash = transaction.hash.unwrap_or([0u8; 32]);

        // Apply filters
        if let Some(filter) = &self.filter {
            if !filter.should_process(&transaction)? {
                self.filtered_transactions.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
        }

        // Add to cache
        self.transactions
            .insert(tx_hash, (transaction.clone(), Instant::now()));
        self.total_transactions.fetch_add(1, Ordering::Relaxed);

        // Track DeFi transactions
        if transaction.is_defi_related() {
            self.defi_transactions.fetch_add(1, Ordering::Relaxed);
        }

        // Analyze for MEV opportunities
        if let Some(analyzer) = &self.analyzer {
            let analysis = analyzer.analyze_transaction(&transaction)?;
            if analysis.has_mev_opportunity {
                self.mev_opportunities.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Clean up old transactions
        self.cleanup_old_transactions();

        Ok(())
    }

    /// Get current statistics
    #[must_use]
    pub fn statistics(&self) -> MempoolStatistics {
        let total = self.total_transactions.load(Ordering::Relaxed);
        let tps = if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                total as f64 / elapsed
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calculate averages
        let (avg_value, avg_gas_price) = if self.transactions.is_empty() {
            (0, 0)
        } else {
            let total_value: u64 = self
                .transactions
                .values()
                .map(|(tx, _)| tx.value().as_wei())
                .sum();
            let total_gas_price: u64 = self
                .transactions
                .values()
                .map(|(tx, _)| tx.gas_price().as_wei())
                .sum();

            let count = self.transactions.len() as u64;
            (total_value / count, total_gas_price / count)
        };

        MempoolStatistics {
            total_transactions: total,
            current_size: self.transactions.len(),
            transactions_per_second: tps,
            avg_transaction_value: avg_value,
            avg_gas_price,
            defi_transactions: self.defi_transactions.load(Ordering::Relaxed),
            mev_opportunities: self.mev_opportunities.load(Ordering::Relaxed),
            filtered_transactions: self.filtered_transactions.load(Ordering::Relaxed),
        }
    }

    /// Get transaction by hash
    #[must_use]
    pub fn get_transaction(&self, hash: &TransactionHash) -> Option<&Transaction> {
        self.transactions.get(hash).map(|(tx, _)| tx)
    }

    /// Get all current transactions
    #[must_use]
    pub fn get_all_transactions(&self) -> Vec<&Transaction> {
        self.transactions.values().map(|(tx, _)| tx).collect()
    }

    /// Clean up old transactions
    fn cleanup_old_transactions(&mut self) {
        let now = Instant::now();
        let ttl = self.config.transaction_ttl;

        self.transactions
            .retain(|_, (_, timestamp)| now.duration_since(*timestamp) < ttl);

        // Also enforce max size limit
        if self.transactions.len() > self.config.max_transactions {
            // Remove oldest transactions
            let entries: Vec<_> = self.transactions.iter().map(|(k, v)| (*k, v.1)).collect();
            let mut sorted_entries = entries;
            sorted_entries.sort_by_key(|(_, timestamp)| *timestamp);

            let to_remove = self.transactions.len() - self.config.max_transactions;
            for (hash, _) in sorted_entries.iter().take(to_remove) {
                self.transactions.remove(hash);
            }
        }
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &MempoolConfig {
        &self.config
    }
}

impl Default for MempoolManager {
    fn default() -> Self {
        Self::new(MempoolConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Gas, Price};

    #[test]
    fn test_mempool_manager_creation() {
        let manager = MempoolManager::default();
        assert_eq!(manager.transactions.len(), 0);
    }

    #[test]
    fn test_mempool_manager_start_stop() -> CoreResult<()> {
        let mut manager = MempoolManager::default();

        manager.start()?;
        assert!(manager.watcher.is_some());
        assert!(manager.filter.is_some());
        assert!(manager.analyzer.is_some());

        manager.stop()?;
        assert!(manager.watcher.is_none());
        assert!(manager.filter.is_none());
        assert!(manager.analyzer.is_none());

        Ok(())
    }

    #[test]
    fn test_transaction_processing() -> CoreResult<()> {
        let mut manager = MempoolManager::default();
        manager.start()?;

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );

        manager.process_transaction(tx)?;

        let stats = manager.statistics();
        assert_eq!(stats.total_transactions, 1);
        assert_eq!(stats.current_size, 1);

        Ok(())
    }

    #[test]
    fn test_defi_transaction_tracking() -> CoreResult<()> {
        let mut manager = MempoolManager::default();
        manager.start()?;

        // Create DeFi transaction
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(2),
            Price::from_gwei(50),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method signature
        );

        manager.process_transaction(tx)?;

        let stats = manager.statistics();
        assert_eq!(stats.defi_transactions, 1);

        Ok(())
    }

    #[test]
    fn test_statistics_calculation() -> CoreResult<()> {
        let mut manager = MempoolManager::default();
        manager.start()?;

        // Add multiple DeFi transactions with unique hashes
        for i in 0..5 {
            let mut tx = Transaction::new(
                [i; 20],
                Some([i + 1; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
            );
            // Set unique hash for each transaction
            let mut hash = [0u8; 32];
            hash[0] = i;
            tx.set_hash(hash);
            manager.process_transaction(tx)?;
        }

        let stats = manager.statistics();
        assert_eq!(stats.total_transactions, 5);
        assert_eq!(stats.current_size, 5);
        assert!(stats.transactions_per_second >= 0.0);

        Ok(())
    }

    #[test]
    fn test_mempool_statistics_default() {
        // Test MempoolStatistics default implementation (line 97)
        let stats = MempoolStatistics::default();
        assert_eq!(stats.total_transactions, 0);
        assert_eq!(stats.current_size, 0);
        assert_eq!(stats.transactions_per_second, 0.0);
        assert_eq!(stats.avg_transaction_value, 0);
        assert_eq!(stats.avg_gas_price, 0);
        assert_eq!(stats.defi_transactions, 0);
        assert_eq!(stats.mev_opportunities, 0);
        assert_eq!(stats.filtered_transactions, 0);
    }

    #[test]
    fn test_transaction_filtering() -> CoreResult<()> {
        // Test transaction filtering (lines 179-180)
        let mut config = MempoolConfig::default();
        config.filter_config.min_value_wei = 1_000_000_000_000_000_000; // 1 ETH minimum
        let mut manager = MempoolManager::new(config);
        manager.start()?;

        // Create transaction below minimum value
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(500_000_000_000_000_000), // 0.5 ETH - below minimum
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        manager.process_transaction(tx)?;

        let stats = manager.statistics();
        assert_eq!(stats.total_transactions, 0); // Should be filtered out
        assert_eq!(stats.filtered_transactions, 1);
        Ok(())
    }

    #[test]
    fn test_statistics_no_start_time() -> CoreResult<()> {
        // Test statistics calculation without start time (lines 218, 221)
        let manager = MempoolManager::default();
        // Don't call start() so start_time remains None

        let stats = manager.statistics();
        assert_eq!(stats.transactions_per_second, 0.0);
        Ok(())
    }

    #[test]
    fn test_statistics_empty_transactions() -> CoreResult<()> {
        // Test statistics calculation with empty transactions (lines 226)
        let mut manager = MempoolManager::default();
        manager.start()?;

        let stats = manager.statistics();
        assert_eq!(stats.avg_transaction_value, 0);
        assert_eq!(stats.avg_gas_price, 0);
        Ok(())
    }

    #[test]
    fn test_cleanup_old_transactions_ttl() -> CoreResult<()> {
        // Test cleanup based on TTL (lines 257-258, 263-264)
        let mut config = MempoolConfig::default();
        config.transaction_ttl = std::time::Duration::from_millis(1); // Very short TTL
        config.filter_config.enable_defi_filter = false; // Disable DeFi filter
        config.enable_realtime_analysis = false; // Disable analyzer to avoid latency violations
        let mut manager = MempoolManager::new(config);
        manager.start()?;

        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );
        tx.set_hash([1u8; 32]); // Set unique hash

        manager.process_transaction(tx)?;
        assert_eq!(manager.transactions.len(), 1);

        // Wait for TTL to expire
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Process another transaction to trigger cleanup
        let mut tx2 = Transaction::new(
            [2u8; 20],
            Some([3u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            1,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00],
        );
        tx2.set_hash([2u8; 32]);
        manager.process_transaction(tx2)?;

        // First transaction should be cleaned up
        assert_eq!(manager.transactions.len(), 1);
        Ok(())
    }

    #[test]
    fn test_cleanup_old_transactions_max_size() -> CoreResult<()> {
        // Test cleanup based on max size (lines 267-268, 272-274, 276, 280, 282-284, 286-287)
        let mut config = MempoolConfig::default();
        config.max_transactions = 2; // Very small limit
        config.filter_config.enable_defi_filter = false; // Disable DeFi filter
        config.enable_realtime_analysis = false; // Disable analyzer
        let mut manager = MempoolManager::new(config);
        manager.start()?;

        // Add 3 transactions to exceed max_transactions
        for i in 0..3 {
            let mut tx = Transaction::new(
                [i; 20],
                Some([i + 1; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00],
            );
            let mut hash = [0u8; 32];
            hash[0] = i;
            tx.set_hash(hash);

            // Add small delay between transactions to ensure different timestamps
            if i > 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }

            manager.process_transaction(tx)?;
        }

        // Should only keep max_transactions (2)
        assert_eq!(manager.transactions.len(), 2);
        Ok(())
    }

    #[test]
    fn test_get_transaction() -> CoreResult<()> {
        // Test get_transaction method (lines 261-262)
        let mut manager = MempoolManager::default();
        manager.start()?;

        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00],
        );
        let hash = [1u8; 32];
        tx.set_hash(hash);

        manager.process_transaction(tx)?;

        // Test existing transaction
        let retrieved = manager.get_transaction(&hash);
        assert!(retrieved.is_some());

        // Test non-existing transaction
        let non_existing = manager.get_transaction(&[99u8; 32]);
        assert!(non_existing.is_none());

        Ok(())
    }

    #[test]
    fn test_get_all_transactions() -> CoreResult<()> {
        // Test get_all_transactions method (lines 267-268)
        let mut manager = MempoolManager::default();
        manager.start()?;

        // Add multiple transactions
        for i in 0..3 {
            let mut tx = Transaction::new(
                [i; 20],
                Some([i + 1; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00],
            );
            let mut hash = [0u8; 32];
            hash[0] = i;
            tx.set_hash(hash);
            manager.process_transaction(tx)?;
        }

        let all_transactions = manager.get_all_transactions();
        assert_eq!(all_transactions.len(), 3);

        Ok(())
    }

    #[test]
    fn test_config_getter() -> CoreResult<()> {
        // Test config getter method (lines 295-296)
        let config = MempoolConfig {
            max_transactions: 50_000,
            transaction_ttl: Duration::from_secs(600),
            enable_realtime_analysis: false,
            analysis_batch_size: 200,
            filter_config: FilterConfig::default(),
        };
        let manager = MempoolManager::new(config.clone());

        let retrieved_config = manager.config();
        assert_eq!(retrieved_config.max_transactions, 50_000);
        assert_eq!(retrieved_config.transaction_ttl, Duration::from_secs(600));
        assert!(!retrieved_config.enable_realtime_analysis);
        assert_eq!(retrieved_config.analysis_batch_size, 200);

        Ok(())
    }

    #[test]
    fn test_mempool_config_default() {
        // Test MempoolConfig default implementation (lines 37, 40, 43)
        let config = MempoolConfig::default();
        assert_eq!(config.max_transactions, 100_000);
        assert_eq!(config.transaction_ttl, Duration::from_secs(300));
        assert!(config.enable_realtime_analysis);
        assert_eq!(config.analysis_batch_size, 100);
    }

    #[test]
    fn test_filter_config_default() {
        // Test FilterConfig default implementation (lines 64-65, 68, 70)
        let config = FilterConfig::default();
        assert_eq!(config.min_gas_price_gwei, 1);
        assert_eq!(config.max_gas_price_gwei, 1000);
        assert_eq!(config.min_value_wei, 0);
        assert!(config.enable_defi_filter);
        assert!(config.enable_mev_filter);
    }

    #[test]
    fn test_manager_without_realtime_analysis() -> CoreResult<()> {
        // Test manager with realtime analysis disabled (lines 162-163, 166)
        let mut config = MempoolConfig::default();
        config.enable_realtime_analysis = false;
        let mut manager = MempoolManager::new(config);
        manager.start()?;

        assert!(manager.analyzer.is_none());

        // Process transaction - should work without analyzer
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00],
        );
        tx.set_hash([1u8; 32]);
        manager.process_transaction(tx)?;

        let stats = manager.statistics();
        assert_eq!(stats.total_transactions, 1);
        assert_eq!(stats.mev_opportunities, 0); // No analyzer, so no MEV detection

        Ok(())
    }

    #[test]
    fn test_cleanup_max_transactions() -> CoreResult<()> {
        // Test cleanup based on max transactions (lines 278-280, 282-283)
        let mut config = MempoolConfig::default();
        config.max_transactions = 2; // Very small limit
        config.filter_config.enable_defi_filter = false; // Disable DeFi filter
        config.enable_realtime_analysis = false; // Disable analyzer to avoid latency violations
        let mut manager = MempoolManager::new(config);
        manager.start()?;

        // Add 3 transactions to exceed limit
        for i in 0..3 {
            let mut tx = Transaction::new(
                [i; 20],
                Some([i + 1; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
            );
            let mut hash = [0u8; 32];
            hash[0] = i;
            tx.set_hash(hash);
            manager.process_transaction(tx)?;

            // Small delay to ensure different timestamps
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // Should only have 2 transactions (oldest removed)
        assert_eq!(manager.transactions.len(), 2);
        Ok(())
    }
}
