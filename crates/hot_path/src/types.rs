//! Zero-cost abstractions and cache-aligned types for maximum performance.
//!
//! This module provides fundamental types optimized for the hot path with
//! cache-line alignment and atomic-safe operations.

use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use core::mem::{align_of, size_of};

extern crate alloc;

/// Cache line size for optimal memory alignment (64 bytes for `x86_64`)
pub const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of supported trading pairs
pub const MAX_TRADING_PAIRS: usize = 4096;

/// Maximum number of supported blockchain networks
pub const MAX_CHAINS: usize = 16;

/// Cache-aligned price data structure
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct AlignedPrice {
    /// Price value in wei (18 decimal places)
    pub price: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Chain identifier
    pub chain_id: u8,
    /// DEX identifier
    pub dex_id: u8,
    /// Padding to ensure 64-byte alignment
    pub padding: [u8; 46],
}

impl Default for AlignedPrice {
    #[inline]
    fn default() -> Self {
        Self {
            price: 0,
            timestamp_ns: 0,
            chain_id: 0,
            dex_id: 0,
            padding: [0; 46],
        }
    }
}

/// Trading pair identifier
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TradingPair {
    /// Unique pair identifier
    pub pair_id: u32,
    /// Base price
    pub base_price: AlignedPrice,
    /// Quote price
    pub quote_price: AlignedPrice,
    /// Padding for cache alignment
    pub padding: [u8; 48],
}

impl TradingPair {
    /// Get the unique identifier for this trading pair
    #[must_use]
    #[inline]
    pub const fn id(&self) -> u32 {
        self.pair_id
    }

    /// Create a new trading pair
    #[must_use]
    #[inline]
    pub const fn new(pair_id: u32, base_price: AlignedPrice, quote_price: AlignedPrice) -> Self {
        Self {
            pair_id,
            base_price,
            quote_price,
            padding: [0; 48],
        }
    }
}

/// MEV opportunity data structure
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Opportunity {
    /// Trading pair identifier
    pub pair_id: u32,
    /// Estimated profit in basis points
    pub profit_estimate: u64,
    /// Timestamp when opportunity was detected
    pub timestamp_ns: u64,
    /// Execution parameters
    pub execution_params: ExecutionParams,
    /// Padding for cache alignment
    pub padding: [u8; 20],
}

impl Opportunity {
    /// Get the age of this opportunity in nanoseconds
    #[must_use]
    #[inline]
    pub fn age_ns(&self) -> u64 {
        get_timestamp_ns().saturating_sub(self.timestamp_ns)
    }

    /// Check if this opportunity is profitable
    #[must_use]
    #[inline]
    pub const fn is_profitable(&self) -> bool {
        self.profit_estimate > 0
    }

    /// Create a new opportunity
    #[must_use]
    #[inline]
    pub const fn new(
        pair_id: u32,
        profit_estimate: u64,
        execution_params: ExecutionParams,
    ) -> Self {
        Self {
            pair_id,
            profit_estimate,
            timestamp_ns: 0, // Will be set by get_timestamp_ns() in real implementation
            execution_params,
            padding: [0; 20],
        }
    }
}

impl Default for Opportunity {
    #[inline]
    fn default() -> Self {
        Self {
            pair_id: 0,
            profit_estimate: 0,
            timestamp_ns: 0,
            execution_params: ExecutionParams::default(),
            padding: [0; 20],
        }
    }
}

/// Market data snapshot
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MarketSnapshot {
    /// Trading pair identifier
    pub pair_id: u32,
    /// Current price in wei
    pub price: u64,
    /// Available liquidity in wei
    pub liquidity: u64,
    /// 24h volume in wei
    pub volume_24h: u64,
    /// Timestamp of this snapshot
    pub timestamp_ns: u64,
    /// Chain ID
    pub chain_id: u8,
    /// DEX ID
    pub dex_id: u8,
    /// Padding for alignment
    pub padding: [u8; 18],
}

impl Default for MarketSnapshot {
    #[inline]
    fn default() -> Self {
        Self {
            pair_id: 0,
            price: 0,
            liquidity: 0,
            volume_24h: 0,
            timestamp_ns: 0,
            chain_id: 0,
            dex_id: 0,
            padding: [0; 18],
        }
    }
}

impl MarketSnapshot {
    /// Create a new market snapshot
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            pair_id: 0,
            price: 0,
            liquidity: 0,
            volume_24h: 0,
            timestamp_ns: 0,
            chain_id: 0,
            dex_id: 0,
            padding: [0; 18],
        }
    }
}

/// Transaction execution parameters
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExecutionParams {
    /// Gas limit
    pub gas_limit: u64,
    /// Gas price in wei
    pub gas_price: u64,
    /// Maximum slippage in basis points
    pub max_slippage_bp: u32,
    /// Transaction deadline in nanoseconds
    pub deadline_ns: u64,
    /// Padding for cache alignment
    pub padding: [u8; 32],
}

impl ExecutionParams {
    /// Check if execution parameters are still valid
    #[must_use]
    #[inline]
    pub fn is_valid(&self) -> bool {
        get_timestamp_ns() < self.deadline_ns
    }

    /// Create new execution parameters
    #[must_use]
    #[inline]
    pub const fn new(
        gas_limit: u64,
        gas_price: u64,
        max_slippage_bp: u32,
        deadline_ns: u64,
    ) -> Self {
        Self {
            gas_limit,
            gas_price,
            max_slippage_bp,
            deadline_ns,
            padding: [0; 32],
        }
    }
}

impl Default for ExecutionParams {
    #[inline]
    fn default() -> Self {
        Self {
            gas_limit: 500_000,
            gas_price: 20_000_000_000, // 20 gwei
            max_slippage_bp: 50, // 0.5%
            deadline_ns: 0,
            padding: [0; 32],
        }
    }
}

/// Get current timestamp in nanoseconds
#[must_use]
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "Nanosecond precision is sufficient")]
#[expect(clippy::as_conversions, reason = "Safe conversion for timestamp")]
pub fn get_timestamp_ns() -> u64 {
    instant::Instant::now().elapsed().as_nanos() as u64
}

/// Atomic ordering for performance-critical operations
pub const ATOMIC_ORDERING: Ordering = Ordering::Relaxed;

/// Atomic-safe counter type
pub type AtomicCounter = AtomicU64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_price_size() {
        assert_eq!(size_of::<AlignedPrice>(), CACHE_LINE_SIZE);
        assert_eq!(align_of::<AlignedPrice>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn opportunity_size() {
        assert_eq!(align_of::<Opportunity>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn trading_pair_id() {
        let base_price = AlignedPrice::default();
        let quote_price = AlignedPrice::default();
        let pair = TradingPair::new(1, base_price, quote_price);
        let id1 = pair.id();
        let id2 = pair.id();
        assert_eq!(id1, id2); // Should be deterministic
        assert_eq!(id1, 1);
    }

    #[test]
    fn opportunity_profitability() {
        let execution_params = ExecutionParams::default();
        let profitable = Opportunity::new(
            1,
            100, // 100 basis points profit
            execution_params,
        );

        assert!(profitable.is_profitable());
        assert_eq!(profitable.profit_estimate, 100);
    }

    #[test]
    fn execution_params_validity() {
        let future_deadline = get_timestamp_ns() + 1_000_000_000; // 1 second from now
        let params = ExecutionParams::new(
            21_000,
            20_000_000_000,
            50,
            future_deadline,
        );

        assert!(params.is_valid());
    }
}