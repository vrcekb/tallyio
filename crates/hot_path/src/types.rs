//! Zero-cost abstractions and cache-aligned data structures for ultra-high performance.
//!
//! This module provides performance-critical data types optimized for AMD EPYC 9454P
//! with nanosecond-level precision requirements.

use alloc::{string::String, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};

/// Cache-aligned price data for SIMD operations
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AlignedPrice {
    /// Price value in fixed-point representation
    pub value: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Volume at this price level
    pub volume: u64,
    /// Padding to ensure cache line alignment
    pub padding: [u8; 40],
}

impl AlignedPrice {
    /// Create a new aligned price
    #[must_use]
    #[inline]
    pub const fn new(value: u64, timestamp_ns: u64, volume: u64) -> Self {
        return Self {
            value,
            timestamp_ns,
            volume,
            padding: [0; 40],
        };
    }

    /// Get the price value
    #[must_use]
    #[inline]
    pub const fn value(&self) -> u64 {
        return self.value;
    }

    /// Get the timestamp
    #[must_use]
    #[inline]
    pub const fn timestamp(&self) -> u64 {
        return self.timestamp_ns;
    }

    /// Get the volume
    #[must_use]
    #[inline]
    pub const fn volume(&self) -> u64 {
        return self.volume;
    }
}

impl Default for AlignedPrice {
    #[inline]
    fn default() -> Self {
        return Self::new(0, 0, 0);
    }
}

/// Trading pair identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TradingPair {
    /// Base token identifier
    pub base: u32,
    /// Quote token identifier  
    pub quote: u32,
    /// Exchange identifier
    pub exchange: u16,
    /// Pool identifier
    pub pool: u16,
}

impl TradingPair {
    /// Create a new trading pair
    #[must_use]
    #[inline]
    pub const fn new(base: u32, quote: u32, exchange: u16, pool: u16) -> Self {
        return Self { base, quote, exchange, pool };
    }
}

/// Market snapshot for MEV detection
#[repr(C, align(64))]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MarketSnapshot {
    /// Trading pairs in this snapshot
    pub pairs: Vec<TradingPair>,
    /// Price data aligned for SIMD operations
    pub prices: Vec<AlignedPrice>,
    /// Snapshot timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Block number when snapshot was taken
    pub block_number: u64,
}

impl MarketSnapshot {
    /// Create a new market snapshot
    #[must_use]
    #[inline]
    pub fn new(capacity: usize) -> Self {
        return Self {
            pairs: Vec::with_capacity(capacity),
            prices: Vec::with_capacity(capacity),
            timestamp_ns: get_timestamp_ns(),
            block_number: 0,
        };
    }

    /// Add a price update to the snapshot
    #[inline]
    pub fn add_price(&mut self, pair: TradingPair, price: AlignedPrice) {
        self.pairs.push(pair);
        self.prices.push(price);
    }

    /// Get the number of price updates
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        return self.prices.len();
    }

    /// Check if the snapshot is empty
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        return self.prices.is_empty();
    }
}

/// Execution parameters for MEV opportunities
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExecutionParams {
    /// Gas limit for the transaction
    pub gas_limit: u64,
    /// Gas price in wei
    pub gas_price: u64,
    /// Maximum slippage tolerance (basis points)
    pub max_slippage_bps: u16,
    /// Deadline for execution (timestamp)
    pub deadline: u64,
    /// Priority fee for EIP-1559
    pub priority_fee: u64,
}

impl ExecutionParams {
    /// Create new execution parameters
    #[must_use]
    #[inline]
    pub const fn new(gas_limit: u64, gas_price: u64, max_slippage_bps: u16, deadline: u64, priority_fee: u64) -> Self {
        return Self {
            gas_limit,
            gas_price,
            max_slippage_bps,
            deadline,
            priority_fee,
        };
    }
}

impl Default for ExecutionParams {
    #[inline]
    fn default() -> Self {
        return Self::new(
            300_000,  // 300k gas limit
            20_000_000_000, // 20 gwei
            50,       // 0.5% slippage
            get_timestamp_ns() + 300_000_000_000, // 5 minutes from now
            1_000_000_000, // 1 gwei priority fee
        );
    }
}

/// MEV opportunity detected by the system
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Opportunity {
    /// Unique identifier for this opportunity
    pub id: u64,
    /// Type of MEV opportunity
    pub opportunity_type: String,
    /// Expected profit in wei
    pub expected_profit: u64,
    /// Confidence score (0-100)
    pub confidence: u8,
    /// Execution parameters
    pub execution_params: ExecutionParams,
    /// Detection timestamp
    pub detected_at: u64,
    /// Expiration timestamp
    pub expires_at: u64,
}

impl Opportunity {
    /// Create a new MEV opportunity
    #[must_use]
    #[inline]
    pub fn new(id: u64, opportunity_type: String, expected_profit: u64, confidence: u8) -> Self {
        let now = get_timestamp_ns();
        return Self {
            id,
            opportunity_type,
            expected_profit,
            confidence,
            execution_params: ExecutionParams::default(),
            detected_at: now,
            expires_at: now + 10_000_000_000, // 10 seconds expiry
        };
    }

    /// Check if the opportunity has expired
    #[must_use]
    #[inline]
    pub fn is_expired(&self) -> bool {
        return get_timestamp_ns() > self.expires_at;
    }

    /// Get the remaining time until expiry in nanoseconds
    #[must_use]
    #[inline]
    pub fn time_to_expiry(&self) -> u64 {
        let now = get_timestamp_ns();
        return self.expires_at.saturating_sub(now);
    }
}

// Global timestamp counter for consistent timing
static TIMESTAMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Get current timestamp in nanoseconds
#[must_use]
#[inline]
pub fn get_timestamp_ns() -> u64 {
    // Stub implementation - in production this would use high-resolution timer
    return TIMESTAMP_COUNTER.fetch_add(1, Ordering::Relaxed);
}

// Configuration types are defined in lib.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_price_creation() {
        let price = AlignedPrice::new(1000, 123_456_789, 500);
        assert_eq!(price.value(), 1000);
        assert_eq!(price.timestamp(), 123_456_789);
        assert_eq!(price.volume(), 500);
    }

    #[test]
    fn test_trading_pair_creation() {
        let pair = TradingPair::new(1, 2, 100, 200);
        assert_eq!(pair.base, 1);
        assert_eq!(pair.quote, 2);
        assert_eq!(pair.exchange, 100);
        assert_eq!(pair.pool, 200);
    }

    #[test]
    fn test_market_snapshot() {
        let mut snapshot = MarketSnapshot::new(10);
        assert!(snapshot.is_empty());
        
        let pair = TradingPair::new(1, 2, 100, 200);
        let price = AlignedPrice::new(1000, 123_456_789, 500);
        snapshot.add_price(pair, price);
        
        assert_eq!(snapshot.len(), 1);
        assert!(!snapshot.is_empty());
    }

    #[test]
    fn test_opportunity_expiry() {
        let opp = Opportunity::new(1, "arbitrage".into(), 1000, 95);
        assert!(!opp.is_expired());
        assert!(opp.time_to_expiry() > 0);
    }

    #[test]
    fn test_timestamp_generation() {
        let ts1 = get_timestamp_ns();
        let ts2 = get_timestamp_ns();
        assert!(ts2 > ts1);
    }
}
