//! Core types for `TallyIO` - Cache-optimized, zero-copy where possible

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Transaction ID - optimized for hashing and comparison
///
/// A 32-byte unique identifier for transactions, using UUID v4 for the first 16 bytes.
/// Designed for fast hashing and comparison operations in hot paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TxId(pub [u8; 32]);

impl TxId {
    /// Create new transaction ID
    ///
    /// Generates a new unique transaction ID using UUID v4 for randomness.
    /// The remaining 16 bytes are zero-filled for deterministic behavior.
    ///
    /// # Returns
    /// A new unique `TxId`
    #[must_use]
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(uuid.as_bytes());
        Self(bytes)
    }

    /// Get as bytes slice
    ///
    /// Returns a reference to the underlying 32-byte array.
    /// Useful for serialization and hashing operations.
    ///
    /// # Returns
    /// Reference to the 32-byte array
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Default for TxId {
    fn default() -> Self {
        Self::new()
    }
}

/// Price in wei/gwei - 64-bit for performance
///
/// Represents cryptocurrency amounts in the smallest unit (wei for Ethereum).
/// Uses u64 for optimal performance in arithmetic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Price(pub u64);

impl Price {
    /// Zero price constant
    pub const ZERO: Self = Self(0);

    /// Maximum price constant
    pub const MAX: Self = Self(u64::MAX);

    /// Create new price
    ///
    /// # Arguments
    /// * `value` - Price value in wei
    ///
    /// # Returns
    /// New `Price` instance
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get raw value
    ///
    /// Returns the underlying u64 value in wei.
    ///
    /// # Returns
    /// Price value in wei
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.0
    }

    /// Add prices with overflow check
    ///
    /// Safely adds two prices, returning `None` if overflow would occur.
    ///
    /// # Arguments
    /// * `other` - Price to add
    ///
    /// # Returns
    /// `Some(Price)` if no overflow, `None` otherwise
    #[must_use]
    pub fn checked_add(&self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }

    /// Multiply by scalar with overflow check
    ///
    /// Safely multiplies price by a scalar, returning `None` if overflow would occur.
    ///
    /// # Arguments
    /// * `scalar` - Value to multiply by
    ///
    /// # Returns
    /// `Some(Price)` if no overflow, `None` otherwise
    #[must_use]
    pub fn checked_mul(&self, scalar: u64) -> Option<Self> {
        self.0.checked_mul(scalar).map(Self)
    }
}

/// Gas amount - optimized for arithmetic operations
///
/// Represents gas amounts for Ethereum transactions.
/// Uses u64 for optimal performance in gas calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Gas(pub u64);

impl Gas {
    /// Zero gas constant
    pub const ZERO: Self = Self(0);

    /// Standard transfer gas constant (21,000 gas)
    pub const TRANSFER: Self = Self(21_000);

    /// Create new gas amount
    ///
    /// # Arguments
    /// * `value` - Gas amount
    ///
    /// # Returns
    /// New `Gas` instance
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get raw value
    ///
    /// Returns the underlying u64 gas amount.
    ///
    /// # Returns
    /// Gas amount as u64
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.0
    }
}

/// Memory-aligned transaction for cache efficiency
///
/// Transaction structure optimized for cache performance with 64-byte alignment.
/// Fields are ordered by access frequency: hot data first, cold data last.
#[repr(C, align(64))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    // Hot data first - frequently accessed
    pub id: TxId,
    pub gas_price: Price,
    pub gas_limit: Gas,
    pub value: Price,

    // Warm data - occasionally accessed
    pub from: [u8; 20],
    pub to: Option<[u8; 20]>,
    pub nonce: u64,

    // Cold data - rarely accessed
    pub data: Vec<u8>,
    pub timestamp: DateTime<Utc>,
}

impl Transaction {
    /// Create new transaction
    ///
    /// # Arguments
    /// * `from` - Sender address (20 bytes)
    /// * `to` - Recipient address (20 bytes, None for contract creation)
    /// * `value` - Transaction value in wei
    /// * `gas_price` - Gas price in wei
    /// * `gas_limit` - Maximum gas to consume
    /// * `nonce` - Transaction nonce
    /// * `data` - Transaction data/input
    ///
    /// # Returns
    /// New `Transaction` instance with generated ID and current timestamp
    #[must_use]
    pub fn new(
        from: [u8; 20],
        to: Option<[u8; 20]>,
        value: Price,
        gas_price: Price,
        gas_limit: Gas,
        nonce: u64,
        data: Vec<u8>,
    ) -> Self {
        Self {
            id: TxId::new(),
            gas_price,
            gas_limit,
            value,
            from,
            to,
            nonce,
            data,
            timestamp: Utc::now(),
        }
    }

    /// Check if transaction is `DeFi` related (for MEV scanning)
    ///
    /// Uses heuristics to quickly identify `DeFi` transactions:
    /// - Has transaction data (not a simple transfer)
    /// - Gas limit exceeds simple transfer requirements
    ///
    /// # Returns
    /// `true` if likely a `DeFi` transaction, `false` otherwise
    #[must_use]
    pub const fn is_defi_related(&self) -> bool {
        // Quick heuristic: has data and reasonable gas limit
        !self.data.is_empty() && self.gas_limit.value() > Gas::TRANSFER.value()
    }

    /// Calculate transaction cost
    ///
    /// Computes the maximum cost of this transaction (`gas_price` * `gas_limit`).
    /// Returns `None` if the multiplication would overflow.
    ///
    /// # Returns
    /// `Some(Price)` with total cost, or `None` if overflow
    #[must_use]
    pub fn cost(&self) -> Option<Price> {
        self.gas_price.checked_mul(self.gas_limit.value())
    }
}

/// Performance metrics - lock-free counters
///
/// Cache-aligned structure for collecting performance metrics without locks.
/// All operations use atomic instructions for thread-safe access.
#[repr(C, align(64))]
pub struct Metrics {
    pub transactions_processed: AtomicU64,
    pub opportunities_found: AtomicU64,
    pub errors_encountered: AtomicU64,
    pub total_latency_ns: AtomicU64,
}

impl Metrics {
    /// Create new metrics instance
    ///
    /// Initializes all counters to zero using atomic operations.
    ///
    /// # Returns
    /// New `Metrics` instance with zero counters
    #[must_use]
    pub const fn new() -> Self {
        Self {
            transactions_processed: AtomicU64::new(0),
            opportunities_found: AtomicU64::new(0),
            errors_encountered: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
        }
    }

    /// Record transaction processing
    ///
    /// Atomically increments the transaction counter and adds latency.
    /// This is a hot path operation designed for minimal overhead.
    ///
    /// # Arguments
    /// * `latency_ns` - Processing latency in nanoseconds
    pub fn record_transaction(&self, latency_ns: u64) {
        self.transactions_processed.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
    }

    /// Record MEV opportunity found
    ///
    /// Atomically increments the opportunities counter.
    /// Called when MEV scanning detects a profitable opportunity.
    pub fn record_opportunity(&self) {
        self.opportunities_found.fetch_add(1, Ordering::Relaxed);
    }

    /// Record error occurrence
    ///
    /// Atomically increments the error counter.
    /// Used for monitoring system health and error rates.
    pub fn record_error(&self) {
        self.errors_encountered.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average latency in nanoseconds
    ///
    /// Calculates the average processing latency across all transactions.
    /// Returns 0 if no transactions have been processed.
    ///
    /// # Returns
    /// Average latency in nanoseconds, or 0 if no data
    #[must_use]
    pub fn average_latency_ns(&self) -> u64 {
        let total = self.total_latency_ns.load(Ordering::Relaxed);
        let count = self.transactions_processed.load(Ordering::Relaxed);
        if count > 0 { total / count } else { 0 }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
