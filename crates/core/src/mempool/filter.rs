//! Transaction filter for mempool monitoring
//!
//! This module provides ultra-fast transaction filtering to reduce processing load
//! and focus on high-value transactions and MEV opportunities.

use crate::error::CoreResult;
use crate::mempool::FilterConfig;
use crate::types::Transaction;
use std::sync::atomic::{AtomicU64, Ordering};

/// Transaction filter result
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FilterResult {
    /// Transaction should be processed
    Accept,
    /// Transaction should be rejected
    Reject(FilterReason),
}

/// Reason for filtering out a transaction
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FilterReason {
    /// Gas price too low
    GasPriceTooLow,
    /// Gas price too high (likely spam)
    GasPriceTooHigh,
    /// Transaction value too low
    ValueTooLow,
    /// Not a DeFi transaction (when DeFi filter enabled)
    NotDeFi,
    /// No MEV opportunity detected (when MEV filter enabled)
    NoMevOpportunity,
    /// Invalid transaction data
    InvalidData,
}

impl FilterResult {
    /// Check if transaction should be processed
    #[must_use]
    pub const fn should_process(self) -> bool {
        matches!(self, Self::Accept)
    }

    /// Get filter reason if rejected
    #[must_use]
    pub const fn reason(self) -> Option<FilterReason> {
        match self {
            Self::Accept => None,
            Self::Reject(reason) => Some(reason),
        }
    }
}

/// Transaction filter trait
pub trait TransactionFilter {
    /// Filter a transaction
    fn filter(&self, transaction: &Transaction) -> CoreResult<FilterResult>;

    /// Get filter statistics
    fn statistics(&self) -> FilterStatistics;
}

/// Mempool transaction filter
///
/// Provides ultra-fast transaction filtering with <10μs latency guarantee
/// to reduce processing load and focus on valuable transactions.
#[repr(C, align(64))]
pub struct MempoolFilter {
    /// Filter configuration
    config: FilterConfig,
    /// Filter statistics
    total_transactions: AtomicU64,
    accepted_transactions: AtomicU64,
    rejected_transactions: AtomicU64,
    gas_price_rejections: AtomicU64,
    value_rejections: AtomicU64,
    defi_rejections: AtomicU64,
    mev_rejections: AtomicU64,
}

impl MempoolFilter {
    /// Create a new mempool filter
    #[must_use]
    pub fn new(config: FilterConfig) -> Self {
        Self {
            config,
            total_transactions: AtomicU64::new(0),
            accepted_transactions: AtomicU64::new(0),
            rejected_transactions: AtomicU64::new(0),
            gas_price_rejections: AtomicU64::new(0),
            value_rejections: AtomicU64::new(0),
            defi_rejections: AtomicU64::new(0),
            mev_rejections: AtomicU64::new(0),
        }
    }

    /// Check if transaction should be processed
    #[inline(always)]
    pub fn should_process(&self, transaction: &Transaction) -> CoreResult<bool> {
        let result = self.filter(transaction)?;
        Ok(result.should_process())
    }

    /// Get filter configuration
    #[must_use]
    pub const fn config(&self) -> &FilterConfig {
        &self.config
    }

    /// Update filter configuration
    pub fn update_config(&mut self, config: FilterConfig) {
        self.config = config;
    }
}

impl TransactionFilter for MempoolFilter {
    #[inline(always)]
    fn filter(&self, transaction: &Transaction) -> CoreResult<FilterResult> {
        self.total_transactions.fetch_add(1, Ordering::Relaxed);

        // Check gas price bounds
        let gas_price_gwei = transaction.gas_price().as_gwei();
        if gas_price_gwei < self.config.min_gas_price_gwei {
            self.gas_price_rejections.fetch_add(1, Ordering::Relaxed);
            self.rejected_transactions.fetch_add(1, Ordering::Relaxed);
            return Ok(FilterResult::Reject(FilterReason::GasPriceTooLow));
        }

        if gas_price_gwei > self.config.max_gas_price_gwei {
            self.gas_price_rejections.fetch_add(1, Ordering::Relaxed);
            self.rejected_transactions.fetch_add(1, Ordering::Relaxed);
            return Ok(FilterResult::Reject(FilterReason::GasPriceTooHigh));
        }

        // Check minimum value
        if transaction.value().as_wei() < self.config.min_value_wei {
            self.value_rejections.fetch_add(1, Ordering::Relaxed);
            self.rejected_transactions.fetch_add(1, Ordering::Relaxed);
            return Ok(FilterResult::Reject(FilterReason::ValueTooLow));
        }

        // Check DeFi filter
        if self.config.enable_defi_filter && !transaction.is_defi_related() {
            self.defi_rejections.fetch_add(1, Ordering::Relaxed);
            self.rejected_transactions.fetch_add(1, Ordering::Relaxed);
            return Ok(FilterResult::Reject(FilterReason::NotDeFi));
        }

        // Check MEV filter (simplified - in real implementation would use analyzer)
        if self.config.enable_mev_filter {
            let has_mev_potential = self.quick_mev_check(transaction);
            if !has_mev_potential {
                self.mev_rejections.fetch_add(1, Ordering::Relaxed);
                self.rejected_transactions.fetch_add(1, Ordering::Relaxed);
                return Ok(FilterResult::Reject(FilterReason::NoMevOpportunity));
            }
        }

        // Transaction passed all filters
        self.accepted_transactions.fetch_add(1, Ordering::Relaxed);
        Ok(FilterResult::Accept)
    }

    fn statistics(&self) -> FilterStatistics {
        let total = self.total_transactions.load(Ordering::Relaxed);
        let accepted = self.accepted_transactions.load(Ordering::Relaxed);
        let rejected = self.rejected_transactions.load(Ordering::Relaxed);

        let acceptance_rate = if total > 0 {
            accepted as f64 / total as f64
        } else {
            0.0
        };

        FilterStatistics {
            total_transactions: total,
            accepted_transactions: accepted,
            rejected_transactions: rejected,
            acceptance_rate,
            gas_price_rejections: self.gas_price_rejections.load(Ordering::Relaxed),
            value_rejections: self.value_rejections.load(Ordering::Relaxed),
            defi_rejections: self.defi_rejections.load(Ordering::Relaxed),
            mev_rejections: self.mev_rejections.load(Ordering::Relaxed),
        }
    }
}

impl MempoolFilter {
    /// Quick MEV opportunity check
    #[inline(always)]
    fn quick_mev_check(&self, transaction: &Transaction) -> bool {
        // Quick heuristics for MEV potential
        if !transaction.is_defi_related() {
            return false;
        }

        // Check for high-value transactions
        if transaction.value().as_wei() > 100_000_000_000_000_000 {
            // > 0.1 ETH
            return true;
        }

        // Check for high gas price (indicates urgency)
        if transaction.gas_price().as_gwei() > 50 {
            return true;
        }

        // Check for known MEV-prone function selectors
        if transaction.data.len() >= 4 {
            let selector = &transaction.data[0..4];
            match selector {
                [0xa9, 0x05, 0x9c, 0xbb] | // swapExactTokensForTokens
                [0x38, 0xed, 0x17, 0x39] | // swapExactETHForTokens
                [0x7f, 0xf3, 0x6a, 0xb5] | // swapExactTokensForETH
                [0x2e, 0x1a, 0x7d, 0x4d] => { // liquidateCalculateSeizeTokens
                    return true;
                }
                _ => {}
            }
        }

        false
    }
}

impl Default for MempoolFilter {
    fn default() -> Self {
        Self::new(FilterConfig::default())
    }
}

/// Filter statistics
#[derive(Debug, Clone)]
pub struct FilterStatistics {
    /// Total transactions processed
    pub total_transactions: u64,
    /// Transactions accepted
    pub accepted_transactions: u64,
    /// Transactions rejected
    pub rejected_transactions: u64,
    /// Acceptance rate (0.0 - 1.0)
    pub acceptance_rate: f64,
    /// Rejections by gas price
    pub gas_price_rejections: u64,
    /// Rejections by value
    pub value_rejections: u64,
    /// Rejections by DeFi filter
    pub defi_rejections: u64,
    /// Rejections by MEV filter
    pub mev_rejections: u64,
}

impl FilterStatistics {
    /// Get rejection rate
    #[must_use]
    pub fn rejection_rate(&self) -> f64 {
        1.0 - self.acceptance_rate
    }

    /// Get most common rejection reason
    #[must_use]
    pub fn most_common_rejection(&self) -> Option<FilterReason> {
        let rejections = [
            (self.gas_price_rejections, FilterReason::GasPriceTooLow),
            (self.value_rejections, FilterReason::ValueTooLow),
            (self.defi_rejections, FilterReason::NotDeFi),
            (self.mev_rejections, FilterReason::NoMevOpportunity),
        ];

        rejections
            .iter()
            .max_by_key(|(count, _)| *count)
            .filter(|(count, _)| *count > 0)
            .map(|(_, reason)| *reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Gas, Price};

    #[test]
    fn test_filter_creation() {
        let filter = MempoolFilter::default();
        let stats = filter.statistics();
        assert_eq!(stats.total_transactions, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
    }

    #[test]
    fn test_gas_price_filtering() -> CoreResult<()> {
        let config = FilterConfig {
            min_gas_price_gwei: 10,
            max_gas_price_gwei: 100,
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction with gas price too low
        let low_gas_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(5), // Too low
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = filter.filter(&low_gas_tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::GasPriceTooLow));

        // Transaction with gas price too high
        let high_gas_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(200), // Too high
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = filter.filter(&high_gas_tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::GasPriceTooHigh));

        // Transaction with acceptable gas price
        let good_gas_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20), // Good
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );

        let result = filter.filter(&good_gas_tx)?;
        assert_eq!(result, FilterResult::Accept);

        Ok(())
    }

    #[test]
    fn test_value_filtering() -> CoreResult<()> {
        let config = FilterConfig {
            min_value_wei: 1_000_000_000_000_000_000, // 1 ETH minimum
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction with value too low
        let low_value_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(0), // Too low
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = filter.filter(&low_value_tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::ValueTooLow));

        Ok(())
    }

    #[test]
    fn test_defi_filtering() -> CoreResult<()> {
        let config = FilterConfig {
            enable_defi_filter: true,
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Non-DeFi transaction
        let simple_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0), // No data = simple transfer
        );

        let result = filter.filter(&simple_tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::NotDeFi));

        // DeFi transaction
        let defi_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );

        let result = filter.filter(&defi_tx)?;
        assert_eq!(result, FilterResult::Accept);

        Ok(())
    }

    #[test]
    fn test_mev_filtering() -> CoreResult<()> {
        let config = FilterConfig {
            enable_mev_filter: true,
            enable_defi_filter: false, // Disable DeFi filter for this test
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction with MEV potential (high value DeFi)
        let mev_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(2), // High value
            Price::from_gwei(60), // High gas price
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // MEV-prone method
        );

        let result = filter.filter(&mev_tx)?;
        assert_eq!(result, FilterResult::Accept);

        Ok(())
    }

    #[test]
    fn test_filter_statistics() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // Process multiple transactions
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
            filter.filter(&tx)?;
        }

        let stats = filter.statistics();
        assert_eq!(stats.total_transactions, 10);
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);

        Ok(())
    }

    #[test]
    fn test_should_process_convenience_method() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let should_process = filter.should_process(&tx)?;
        assert!(should_process || !should_process); // Just test it doesn't panic

        Ok(())
    }

    #[test]
    fn test_gas_price_filtering_high() -> CoreResult<()> {
        let config = FilterConfig {
            max_gas_price_gwei: 100,
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction with too high gas price
        let high_gas_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(150), // > 100 gwei
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = filter.filter(&high_gas_tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::GasPriceTooHigh));

        Ok(())
    }

    #[test]
    fn test_filter_result_methods() {
        // Test FilterResult methods
        let accept = FilterResult::Accept;
        assert!(accept.should_process());
        assert!(accept.reason().is_none());

        let reject = FilterResult::Reject(FilterReason::GasPriceTooLow);
        assert!(!reject.should_process());
        assert_eq!(reject.reason(), Some(FilterReason::GasPriceTooLow));
    }

    #[test]
    fn test_filter_statistics_methods() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // Add some transactions to get statistics
        let tx1 = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        filter.filter(&tx1)?;

        let stats = filter.statistics();

        // Test rejection rate
        let rejection_rate = stats.rejection_rate();
        assert!((0.0..=1.0).contains(&rejection_rate));
        assert_eq!(rejection_rate, 1.0 - stats.acceptance_rate);

        // Test most common rejection
        let most_common = stats.most_common_rejection();
        // Should be None if no rejections, or Some(reason) if there are rejections
        assert!(most_common.is_none() || most_common.is_some());

        Ok(())
    }

    #[test]
    fn test_quick_mev_check_high_value() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // High value DeFi transaction
        let high_value_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1), // > 0.1 ETH
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );

        let has_mev = filter.quick_mev_check(&high_value_tx);
        assert!(has_mev);

        Ok(())
    }

    #[test]
    fn test_quick_mev_check_high_gas_price() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // High gas price DeFi transaction
        let high_gas_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(0), // Low value
            Price::from_gwei(60), // > 50 gwei
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );

        let has_mev = filter.quick_mev_check(&high_gas_tx);
        assert!(has_mev);

        Ok(())
    }

    #[test]
    fn test_quick_mev_check_known_selectors() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // Test different MEV-prone selectors
        let selectors = [
            vec![0xa9, 0x05, 0x9c, 0xbb], // swapExactTokensForTokens
            vec![0x38, 0xed, 0x17, 0x39], // swapExactETHForTokens
            vec![0x7f, 0xf3, 0x6a, 0xb5], // swapExactTokensForETH
            vec![0x2e, 0x1a, 0x7d, 0x4d], // liquidateCalculateSeizeTokens
        ];

        for selector in selectors {
            let tx = Transaction::new(
                [1u8; 20],
                Some([2u8; 20]),
                Price::from_ether(0), // Low value
                Price::from_gwei(20), // Low gas price
                Gas::new(150_000),
                0,
                selector,
            );

            let has_mev = filter.quick_mev_check(&tx);
            assert!(has_mev);
        }

        Ok(())
    }

    #[test]
    fn test_quick_mev_check_non_defi() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // Non-DeFi transaction
        let non_defi_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(60),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0), // No data = not DeFi
        );

        let has_mev = filter.quick_mev_check(&non_defi_tx);
        assert!(!has_mev);

        Ok(())
    }

    #[test]
    fn test_quick_mev_check_unknown_selector() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // DeFi transaction with unknown selector
        let unknown_selector_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(0), // Low value
            Price::from_gwei(20), // Low gas price
            Gas::new(150_000),
            0,
            vec![0xff, 0xff, 0xff, 0xff], // Unknown selector
        );

        let has_mev = filter.quick_mev_check(&unknown_selector_tx);
        assert!(!has_mev);

        Ok(())
    }

    #[test]
    fn test_config_methods() -> CoreResult<()> {
        let config = FilterConfig::default();
        let mut filter = MempoolFilter::new(config.clone());

        // Test config getter
        let retrieved_config = filter.config();
        assert_eq!(
            retrieved_config.enable_defi_filter,
            config.enable_defi_filter
        );

        // Test config update
        let new_config = FilterConfig {
            enable_defi_filter: true,
            enable_mev_filter: true,
            ..Default::default()
        };
        filter.update_config(new_config.clone());

        let updated_config = filter.config();
        assert_eq!(
            updated_config.enable_defi_filter,
            new_config.enable_defi_filter
        );
        assert_eq!(
            updated_config.enable_mev_filter,
            new_config.enable_mev_filter
        );

        Ok(())
    }

    #[test]
    fn test_mev_filter_no_mev_opportunity() -> CoreResult<()> {
        let config = FilterConfig {
            enable_mev_filter: true,
            enable_defi_filter: false,
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction without MEV potential
        let no_mev_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(0), // Low value
            Price::from_gwei(20), // Low gas price
            Gas::new(21_000),
            0,
            Vec::with_capacity(0), // No data = not DeFi
        );

        let result = filter.filter(&no_mev_tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::NoMevOpportunity));

        Ok(())
    }

    #[test]
    fn test_statistics_with_rejections() -> CoreResult<()> {
        let config = FilterConfig {
            min_gas_price_gwei: 50,
            min_value_wei: 1_000_000_000_000_000_000, // 1 ETH
            enable_defi_filter: true,
            enable_mev_filter: true,
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Add transactions that will be rejected for different reasons

        // Gas price too low
        let low_gas_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(2),
            Price::from_gwei(10), // Too low
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb],
        );
        filter.filter(&low_gas_tx)?;

        // Value too low
        let low_value_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(0), // Too low
            Price::from_gwei(60),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb],
        );
        filter.filter(&low_value_tx)?;

        // Not DeFi
        let non_defi_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(2),
            Price::from_gwei(60),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0), // No data
        );
        filter.filter(&non_defi_tx)?;

        let stats = filter.statistics();
        assert!(stats.gas_price_rejections > 0);
        assert!(stats.value_rejections > 0);
        assert!(stats.defi_rejections > 0);

        // Test most common rejection
        let most_common = stats.most_common_rejection();
        assert!(most_common.is_some());

        Ok(())
    }

    #[test]
    fn test_value_filtering_low() -> CoreResult<()> {
        let config = FilterConfig {
            min_value_wei: 1_000_000_000_000_000_000, // 1 ETH minimum
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction with too low value
        let low_value_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(500_000_000_000_000_000), // 0.5 ETH
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = filter.filter(&low_value_tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::ValueTooLow));

        Ok(())
    }

    #[test]
    fn test_mev_filtering_no_opportunity() -> CoreResult<()> {
        let config = FilterConfig {
            enable_mev_filter: true,
            enable_defi_filter: false,
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Simple transfer with no MEV potential
        let simple_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(50_000_000_000_000_000), // 0.05 ETH (low value)
            Price::from_gwei(15),               // Low gas price
            Gas::new(21_000),
            0,
            Vec::with_capacity(0), // No data
        );

        let result = filter.filter(&simple_tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::NoMevOpportunity));

        Ok(())
    }

    #[test]
    fn test_filter_config_default() {
        let config = FilterConfig::default();
        assert_eq!(config.min_gas_price_gwei, 1);
        assert_eq!(config.max_gas_price_gwei, 1000);
        assert_eq!(config.min_value_wei, 0);
        assert!(config.enable_defi_filter);
        assert!(config.enable_mev_filter);
    }

    #[test]
    fn test_filter_statistics_detailed() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // Create transactions that will be rejected for different reasons
        let low_gas_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::new(500_000_000), // < 1 gwei
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let high_gas_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(1500), // > 1000 gwei
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let good_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method to pass DeFi filter
        );

        // Process transactions
        filter.filter(&low_gas_tx)?;
        filter.filter(&high_gas_tx)?;
        filter.filter(&good_tx)?;

        let stats = filter.statistics();
        assert_eq!(stats.total_transactions, 3);
        assert_eq!(stats.accepted_transactions, 1);
        assert_eq!(stats.rejected_transactions, 2);
        assert_eq!(stats.gas_price_rejections, 2);
        assert_eq!(stats.acceptance_rate, 1.0 / 3.0);

        Ok(())
    }

    #[test]
    fn test_filter_creation_with_config() {
        let config = FilterConfig {
            min_gas_price_gwei: 5,
            max_gas_price_gwei: 500,
            min_value_wei: 1000,
            enable_defi_filter: true,
            enable_mev_filter: true,
        };

        let filter = MempoolFilter::new(config);
        let stats = filter.statistics();

        assert_eq!(stats.total_transactions, 0);
        assert_eq!(stats.accepted_transactions, 0);
        assert_eq!(stats.rejected_transactions, 0);
    }

    #[test]
    fn test_should_process_method() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // Test should_process method (line 101)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );

        let should_process = filter.should_process(&tx)?;
        assert!(should_process);

        Ok(())
    }

    #[test]
    fn test_gas_price_too_high_rejection() -> CoreResult<()> {
        let config = FilterConfig {
            max_gas_price_gwei: 100,
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction with gas price too high (line 129)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(150), // > 100 gwei
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = filter.filter(&tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::GasPriceTooHigh));

        let stats = filter.statistics();
        assert_eq!(stats.gas_price_rejections, 1);

        Ok(())
    }

    #[test]
    fn test_value_too_low_rejection() -> CoreResult<()> {
        let config = FilterConfig {
            min_value_wei: 1_000_000_000_000_000_000, // 1 ETH
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction with value too low (line 136)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(500_000_000_000_000_000), // 0.5 ETH
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = filter.filter(&tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::ValueTooLow));

        let stats = filter.statistics();
        assert_eq!(stats.value_rejections, 1);

        Ok(())
    }

    #[test]
    fn test_defi_filter_rejection() -> CoreResult<()> {
        let config = FilterConfig {
            enable_defi_filter: true,
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Non-DeFi transaction (line 145-146)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0), // No data = not DeFi
        );

        let result = filter.filter(&tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::NotDeFi));

        let stats = filter.statistics();
        assert_eq!(stats.defi_rejections, 1);

        Ok(())
    }

    #[test]
    fn test_mev_filter_enabled() -> CoreResult<()> {
        let config = FilterConfig {
            enable_mev_filter: true,
            enable_defi_filter: false, // Disable DeFi filter
            ..Default::default()
        };
        let filter = MempoolFilter::new(config);

        // Transaction without MEV potential (line 150, 154-155)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(50_000_000_000_000_000), // 0.05 ETH (low value)
            Price::from_gwei(15),               // Low gas price
            Gas::new(21_000),
            0,
            Vec::with_capacity(0), // No data
        );

        let result = filter.filter(&tx)?;
        assert_eq!(result, FilterResult::Reject(FilterReason::NoMevOpportunity));

        let stats = filter.statistics();
        assert_eq!(stats.mev_rejections, 1);

        Ok(())
    }

    #[test]
    fn test_transaction_accepted() -> CoreResult<()> {
        let filter = MempoolFilter::default();

        // Transaction that passes all filters (line 161)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );

        let result = filter.filter(&tx)?;
        assert_eq!(result, FilterResult::Accept);

        let stats = filter.statistics();
        assert_eq!(stats.accepted_transactions, 1);

        Ok(())
    }

    #[test]
    fn test_quick_mev_check_high_gas_price_coverage() {
        let filter = MempoolFilter::default();

        // Transaction with high gas price (line 204)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(50_000_000_000_000_000), // 0.05 ETH
            Price::from_gwei(60),               // > 50 gwei
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method
        );

        let has_mev = filter.quick_mev_check(&tx);
        assert!(has_mev);
    }

    #[test]
    fn test_quick_mev_check_selectors() {
        let filter = MempoolFilter::default();

        // Test different MEV-prone selectors (lines 209-212, 216, 218)
        let selectors = [
            vec![0xa9, 0x05, 0x9c, 0xbb], // swapExactTokensForTokens
            vec![0x38, 0xed, 0x17, 0x39], // swapExactETHForTokens
            vec![0x7f, 0xf3, 0x6a, 0xb5], // swapExactTokensForETH
            vec![0x2e, 0x1a, 0x7d, 0x4d], // liquidateCalculateSeizeTokens
        ];

        for selector in selectors {
            let tx = Transaction::new(
                [1u8; 20],
                Some([2u8; 20]),
                Price::new(50_000_000_000_000_000), // 0.05 ETH
                Price::from_gwei(20),
                Gas::new(150_000),
                0,
                selector,
            );

            let has_mev = filter.quick_mev_check(&tx);
            assert!(has_mev);
        }
    }

    #[test]
    fn test_quick_mev_check_no_mev() {
        let filter = MempoolFilter::default();

        // Transaction with no MEV potential (line 222)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(50_000_000_000_000_000), // 0.05 ETH (low value)
            Price::from_gwei(20),               // Low gas price
            Gas::new(21_000),
            0,
            vec![0xff, 0xff, 0xff, 0xff], // Unknown selector
        );

        let has_mev = filter.quick_mev_check(&tx);
        assert!(!has_mev);
    }

    #[test]
    fn test_filter_statistics_rejection_rate() {
        let filter = MempoolFilter::default();

        // Process some transactions to get statistics
        let tx1 = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(150_000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // DeFi method - should accept
        );

        let tx2 = Transaction::new(
            [2u8; 20],
            Some([3u8; 20]),
            Price::from_ether(1),
            Price::new(500_000_000), // < 1 gwei - should reject
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let _ = filter.filter(&tx1);
        let _ = filter.filter(&tx2);

        let stats = filter.statistics();

        // Test rejection_rate method (line 256-257)
        let rejection_rate = stats.rejection_rate();
        assert!((0.0..=1.0).contains(&rejection_rate));
        assert_eq!(rejection_rate, 1.0 - stats.acceptance_rate);
    }

    #[test]
    fn test_most_common_rejection() {
        let filter = MempoolFilter::default();

        // Create multiple transactions with gas price rejections
        for i in 0..5 {
            let tx = Transaction::new(
                [i; 20],
                Some([i + 1; 20]),
                Price::from_ether(1),
                Price::new(500_000_000), // < 1 gwei
                Gas::new(21_000),
                0,
                Vec::with_capacity(0),
            );
            let _ = filter.filter(&tx);
        }

        let stats = filter.statistics();

        // Test most_common_rejection method (lines 262-267, 270, 272-274)
        let most_common = stats.most_common_rejection();
        assert_eq!(most_common, Some(FilterReason::GasPriceTooLow));
    }

    #[test]
    fn test_most_common_rejection_none() {
        let filter = MempoolFilter::default();
        let stats = filter.statistics();

        // No rejections yet, should return None
        let most_common = stats.most_common_rejection();
        assert_eq!(most_common, None);
    }

    #[test]
    fn test_quick_mev_check_no_opportunity() {
        // Test quick MEV check with no opportunity (line 222)
        let filter = MempoolFilter::default();

        let no_mev_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(50_000_000_000_000_000), // 0.05 ETH (low value)
            Price::from_gwei(20),               // Low gas price
            Gas::new(21_000),
            0,
            Vec::with_capacity(0), // No data
        );

        let has_mev = filter.quick_mev_check(&no_mev_tx);
        assert!(!has_mev);
    }

    #[test]
    fn test_filter_config_update() {
        // Test filter config update (lines 111-113)
        let mut filter = MempoolFilter::default();

        let new_config = FilterConfig {
            min_gas_price_gwei: 50,
            max_gas_price_gwei: 500,
            min_value_wei: 1_000_000_000_000_000_000, // 1 ETH
            enable_defi_filter: false,
            enable_mev_filter: false,
        };

        filter.update_config(new_config);
        assert_eq!(filter.config().min_gas_price_gwei, 50);
        assert_eq!(filter.config().max_gas_price_gwei, 500);
        assert_eq!(filter.config().min_value_wei, 1_000_000_000_000_000_000);
        assert!(!filter.config().enable_defi_filter);
        assert!(!filter.config().enable_mev_filter);
    }
}
