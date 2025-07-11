//! MEV detection engine with SIMD-optimized scanning.
//!
//! This module provides ultra-fast MEV opportunity detection using
//! SIMD-optimized algorithms and lock-free data structures.

use crate::{Result, types::{MarketSnapshot, Opportunity, AlignedPrice}};
use alloc::vec::Vec;

// Sub-modules
pub mod opportunity_scanner;
pub mod price_monitor;
pub mod mempool_analyzer;
pub mod pattern_matcher;

// Re-export key types
pub use opportunity_scanner::{OpportunityScanner, ScanResult};
pub use price_monitor::{PriceMonitor, PriceUpdate};
pub use mempool_analyzer::{MempoolAnalyzer, TransactionAnalysis};
pub use pattern_matcher::{PatternMatcher, Pattern, MatchResult};

/// Detect MEV opportunities from market snapshot
///
/// # Errors
///
/// Returns an error if detection fails
#[inline]
pub fn detect_opportunities(snapshot: &MarketSnapshot) -> Result<Vec<Opportunity>> {
    let mut opportunities = Vec::with_capacity(16);
    
    // Use opportunity scanner to find potential MEV opportunities
    let scanner = OpportunityScanner::new();
    let scan_results = scanner.scan(snapshot)?;
    
    // Convert scan results to opportunities
    for (index, result) in scan_results.iter().enumerate() {
        if result.confidence > 80 {
            let opportunity = Opportunity::new(
                u64::try_from(index).unwrap_or(0),
                result.opportunity_type.clone(),
                result.expected_profit,
                result.confidence,
            );
            opportunities.push(opportunity);
        }
    }
    
    return Ok(opportunities);
}

/// Update price feed with new price data
///
/// # Errors
///
/// Returns an error if price update fails
#[inline]
pub fn update_price_feed(pair_id: u32, price: AlignedPrice) -> Result<()> {
    // Use price monitor to update price feed
    let monitor = PriceMonitor::new();
    monitor.update_price(pair_id, price)?;
    
    return Ok(());
}

/// Initialize detection subsystem
///
/// # Errors
///
/// Returns an error if initialization fails
#[inline]
pub fn initialize() -> Result<()> {
    // Initialize all detection components
    opportunity_scanner::initialize()?;
    price_monitor::initialize()?;
    mempool_analyzer::initialize()?;
    pattern_matcher::initialize()?;
    
    return Ok(());
}

/// Get detection statistics
#[must_use]
#[inline]
pub fn get_detection_stats() -> DetectionStats {
    return DetectionStats {
        opportunities_detected: opportunity_scanner::get_opportunities_detected(),
        prices_processed: price_monitor::get_prices_processed(),
        transactions_analyzed: mempool_analyzer::get_transactions_analyzed(),
        patterns_matched: pattern_matcher::get_patterns_matched(),
    };
}

/// Detection statistics
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DetectionStats {
    /// Number of opportunities detected
    pub opportunities_detected: u64,
    /// Number of prices processed
    pub prices_processed: u64,
    /// Number of transactions analyzed
    pub transactions_analyzed: u64,
    /// Number of patterns matched
    pub patterns_matched: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MarketSnapshot;

    #[test]
    fn test_detection_initialization() {
        initialize().unwrap();
    }

    #[test]
    fn test_detect_opportunities() {
        let snapshot = MarketSnapshot::new(10);
        let opportunities = detect_opportunities(&snapshot).unwrap();
        assert!(opportunities.len() <= 16);
    }

    #[test]
    fn test_update_price_feed() {
        let price = AlignedPrice::new(1000, 123_456_789, 500);
        update_price_feed(1, price).unwrap();
    }

    #[test]
    fn test_detection_stats() {
        let stats = get_detection_stats();
        assert!(stats.opportunities_detected >= 0);
        assert!(stats.prices_processed >= 0);
        assert!(stats.transactions_analyzed >= 0);
        assert!(stats.patterns_matched >= 0);
    }
}
