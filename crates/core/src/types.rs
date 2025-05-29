//! Core types for `TallyIO` - Cache-optimized, zero-copy where possible

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

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

impl std::fmt::Display for TxId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display first 16 bytes as hex
        for byte in &self.0[..16] {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
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
    pub const fn value(self) -> u64 {
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
    pub fn checked_add(self, other: Self) -> Option<Self> {
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
    pub fn checked_mul(self, scalar: u64) -> Option<Self> {
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

    /// Standard contract call gas limit
    pub const CONTRACT_CALL: Self = Self(50_000);

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
    pub const fn value(self) -> u64 {
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
    pub to: Option<Address>,
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
            to: to.map(Address::new),
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

impl Default for Transaction {
    fn default() -> Self {
        Self {
            id: TxId::new(),
            gas_price: Price::new(0),
            gas_limit: Gas::TRANSFER, // Default to transfer gas
            value: Price::new(0),
            from: [0; 20],
            to: None,
            nonce: 0,
            data: Vec::new(),
            timestamp: Utc::now(),
        }
    }
}

/// Ethereum address type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address([u8; 20]);

impl Address {
    /// Create new address
    #[must_use]
    pub const fn new(bytes: [u8; 20]) -> Self {
        Self(bytes)
    }

    /// Zero address constant
    #[must_use]
    pub const fn zero() -> Self {
        Self([0; 20])
    }

    /// Get address bytes
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 20] {
        &self.0
    }
}

impl From<[u8; 20]> for Address {
    fn from(bytes: [u8; 20]) -> Self {
        Self::new(bytes)
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x")?;
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// MEV opportunity data
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MevOpportunity {
    pub profit_wei: Price,
    pub gas_cost: Price,
    pub confidence: u8,
}

/// Transaction processing status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessingStatus {
    /// Transaction processed successfully
    Success,
    /// Transaction rejected with reason
    Rejected(String),
    /// Processing failed with error message
    Failed(String),
}

/// Processed transaction with results
#[derive(Debug, Clone)]
pub struct ProcessedTransaction {
    pub transaction: Transaction,
    pub status: ProcessingStatus,
    pub processing_time_ns: u64,
    pub mev_opportunity: Option<MevOpportunity>,
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
        self.total_latency_ns
            .fetch_add(latency_ns, Ordering::Relaxed);
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
        if count > 0 {
            total / count
        } else {
            0
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::missing_errors_doc)]
mod tests {
    use super::*;
    // All types are now in this module

    #[test]
    fn test_price_creation() {
        let price = Price::new(1_000_000_000_000_000_000);
        assert_eq!(price.value(), 1_000_000_000_000_000_000);
    }

    #[test]
    fn test_price_checked_mul() {
        let price = Price::new(1_000_000_000);
        let gas = Gas::new(21_000);

        let result = price.checked_mul(gas.value());
        assert!(result.is_some());
        if let Some(result) = result {
            assert_eq!(result.value(), 21_000_000_000_000);
        }
    }

    #[test]
    fn test_price_checked_mul_overflow() {
        let price = Price::new(u64::MAX);
        let gas = Gas::new(2);

        let result = price.checked_mul(gas.value());
        assert!(result.is_none());
    }

    #[test]
    fn test_gas_constants() {
        assert_eq!(Gas::TRANSFER.value(), 21_000);
        assert_eq!(Gas::CONTRACT_CALL.value(), 50_000);
    }

    #[test]
    fn test_address_creation() {
        let addr = Address::new([0x42; 20]);
        assert_eq!(addr.as_bytes(), &[0x42; 20]);
    }

    #[test]
    fn test_address_zero() {
        let zero = Address::zero();
        assert_eq!(zero.as_bytes(), &[0; 20]);
    }

    #[test]
    fn test_txid_generation() {
        let id1 = TxId::new();
        let id2 = TxId::new();
        assert_ne!(id1, id2); // Should be unique
    }

    #[test]
    fn test_transaction_creation() {
        let tx = Transaction::new(
            [1; 20],
            Some([2; 20]),
            Price::new(1_000_000_000_000_000_000),
            Price::new(20_000_000_000),
            Gas::new(21_000),
            0,
            vec![],
        );

        assert_eq!(tx.from, [1; 20]);
        assert_eq!(tx.to, Some(Address::new([2; 20])));
        assert_eq!(tx.value.value(), 1_000_000_000_000_000_000);
        assert_eq!(tx.gas_price.value(), 20_000_000_000);
        assert_eq!(tx.gas_limit.value(), 21_000);
        assert_eq!(tx.nonce, 0);
        assert!(tx.data.is_empty());
    }

    #[test]
    fn test_transaction_default() {
        let tx = Transaction::default();
        assert_eq!(tx.from, [0; 20]);
        assert_eq!(tx.to, None);
        assert_eq!(tx.value.value(), 0);
        assert_eq!(tx.gas_price.value(), 0);
        assert_eq!(tx.gas_limit.value(), 21_000); // Default gas limit
        assert_eq!(tx.nonce, 0);
        assert!(tx.data.is_empty());
    }

    #[test]
    fn test_transaction_is_defi_related() {
        // Simple transfer - not DeFi
        let simple_tx = Transaction::default();
        assert!(!simple_tx.is_defi_related());

        // Transaction with data and higher gas - DeFi
        let mut defi_tx = Transaction::default();
        defi_tx.data = vec![0xa9, 0x05, 0x9c, 0xbb]; // Some method call
        defi_tx.gas_limit = Gas::new(100_000); // Higher than transfer
        assert!(defi_tx.is_defi_related());

        // Transaction with data but low gas - not DeFi
        let mut low_gas_tx = Transaction::default();
        low_gas_tx.data = vec![0x01, 0x02];
        low_gas_tx.gas_limit = Gas::new(21_000); // Transfer gas
        assert!(!low_gas_tx.is_defi_related());
    }

    #[test]
    fn test_transaction_cost() {
        let tx = Transaction::new(
            [1; 20],
            Some([2; 20]),
            Price::new(1_000_000_000_000_000_000),
            Price::new(20_000_000_000), // 20 gwei
            Gas::new(21_000),
            0,
            vec![],
        );

        let cost = tx.cost();
        assert!(cost.is_some());
        if let Some(cost) = cost {
            assert_eq!(cost.value(), 420_000_000_000_000); // 20 gwei * 21k gas
        }
    }

    #[test]
    fn test_transaction_cost_overflow() {
        let mut tx = Transaction::default();
        tx.gas_price = Price::new(u64::MAX);
        tx.gas_limit = Gas::new(2);

        let cost = tx.cost();
        assert!(cost.is_none()); // Should overflow
    }

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new();
        assert_eq!(metrics.transactions_processed.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.opportunities_found.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.errors_encountered.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.total_latency_ns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_default() {
        let metrics = Metrics::default();
        assert_eq!(metrics.transactions_processed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_record_transaction() {
        let metrics = Metrics::new();

        metrics.record_transaction(1_000_000); // 1ms
        assert_eq!(metrics.transactions_processed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_latency_ns.load(Ordering::Relaxed), 1_000_000);

        metrics.record_transaction(2_000_000); // 2ms
        assert_eq!(metrics.transactions_processed.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.total_latency_ns.load(Ordering::Relaxed), 3_000_000);
    }

    #[test]
    fn test_metrics_record_opportunity() {
        let metrics = Metrics::new();

        metrics.record_opportunity();
        assert_eq!(metrics.opportunities_found.load(Ordering::Relaxed), 1);

        metrics.record_opportunity();
        assert_eq!(metrics.opportunities_found.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_metrics_record_error() {
        let metrics = Metrics::new();

        metrics.record_error();
        assert_eq!(metrics.errors_encountered.load(Ordering::Relaxed), 1);

        metrics.record_error();
        assert_eq!(metrics.errors_encountered.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_metrics_average_latency() {
        let metrics = Metrics::new();

        // No transactions - should return 0
        assert_eq!(metrics.average_latency_ns(), 0);

        // Add some transactions
        metrics.record_transaction(1_000_000); // 1ms
        metrics.record_transaction(3_000_000); // 3ms

        // Average should be 2ms
        assert_eq!(metrics.average_latency_ns(), 2_000_000);
    }

    #[test]
    fn test_processing_status_variants() {
        let success = ProcessingStatus::Success;
        let rejected = ProcessingStatus::Rejected("test reason".to_string());
        let failed = ProcessingStatus::Failed("test error".to_string());

        assert!(matches!(success, ProcessingStatus::Success));
        assert!(matches!(rejected, ProcessingStatus::Rejected(_)));
        assert!(matches!(failed, ProcessingStatus::Failed(_)));
    }

    #[test]
    fn test_mev_opportunity() {
        let opportunity = MevOpportunity {
            profit_wei: Price::new(1_000_000_000_000_000_000), // 1 ETH
            gas_cost: Price::new(420_000_000_000_000),         // 0.00042 ETH
            confidence: 95,
        };

        assert_eq!(opportunity.profit_wei.value(), 1_000_000_000_000_000_000);
        assert_eq!(opportunity.gas_cost.value(), 420_000_000_000_000);
        assert_eq!(opportunity.confidence, 95);
    }

    #[test]
    fn test_processed_transaction() {
        let tx = Transaction::default();
        let processed = ProcessedTransaction {
            transaction: tx,
            status: ProcessingStatus::Success,
            processing_time_ns: 500_000, // 0.5ms
            mev_opportunity: None,
        };

        assert!(matches!(processed.status, ProcessingStatus::Success));
        assert_eq!(processed.processing_time_ns, 500_000);
        assert!(processed.mev_opportunity.is_none());
    }

    #[test]
    fn test_txid_display() {
        let id = TxId::new();
        let display_str = format!("{id}");
        assert!(!display_str.is_empty());
        assert!(display_str.len() >= 8); // At least 8 hex chars
    }

    #[test]
    fn test_txid_equality() {
        let id1 = TxId::new();
        let id2 = TxId::new();

        // Should be different
        assert_ne!(id1, id2);

        // Should be equal to itself
        assert_eq!(id1, id1);
    }

    #[test]
    fn test_address_display() {
        let addr = Address::new([0x42; 20]);
        let display_str = format!("{addr}");
        assert!(!display_str.is_empty());
        assert!(display_str.starts_with("0x"));
    }

    #[test]
    fn test_address_from_bytes() {
        let bytes = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
        ];
        let addr = Address::from(bytes);
        assert_eq!(addr.as_bytes(), &bytes);
    }

    #[test]
    fn test_price_zero() {
        let price = Price::ZERO;
        assert_eq!(price.value(), 0);
    }

    #[test]
    fn test_gas_zero() {
        let gas = Gas::ZERO;
        assert_eq!(gas.value(), 0);
    }

    #[test]
    fn test_transaction_with_empty_data() {
        let mut tx = Transaction::default();
        tx.data = vec![];
        assert!(!tx.is_defi_related()); // Empty data should not be DeFi
    }

    #[test]
    fn test_transaction_with_high_gas_no_data() {
        let mut tx = Transaction::default();
        tx.gas_limit = Gas::new(100_000); // High gas
        tx.data = vec![]; // But no data
        assert!(!tx.is_defi_related()); // Should not be DeFi without data
    }

    #[test]
    fn test_mev_opportunity_with_values() {
        let opportunity = MevOpportunity {
            profit_wei: Price::new(500_000_000_000_000_000), // 0.5 ETH
            gas_cost: Price::new(210_000_000_000_000),       // 0.00021 ETH
            confidence: 75,
        };

        // Test that we can access all fields
        assert!(opportunity.profit_wei.value() > opportunity.gas_cost.value());
        assert!(opportunity.confidence > 0 && opportunity.confidence <= 100);
    }

    #[test]
    fn test_processed_transaction_with_mev() {
        let tx = Transaction::default();
        let mev_opp = MevOpportunity {
            profit_wei: Price::new(1_000_000_000_000_000_000),
            gas_cost: Price::new(420_000_000_000_000),
            confidence: 90,
        };

        let processed = ProcessedTransaction {
            transaction: tx,
            status: ProcessingStatus::Success,
            processing_time_ns: 750_000, // 0.75ms
            mev_opportunity: Some(mev_opp),
        };

        assert!(processed.mev_opportunity.is_some());
        if let Some(opp) = processed.mev_opportunity {
            assert_eq!(opp.confidence, 90);
        }
    }

    #[test]
    fn test_processing_status_failed() {
        let status = ProcessingStatus::Failed("Network error".to_string());
        assert!(matches!(status, ProcessingStatus::Failed(_)));
        if let ProcessingStatus::Failed(msg) = status {
            assert_eq!(msg, "Network error");
        }
    }

    #[test]
    fn test_txid_as_bytes() {
        let id = TxId::new();
        let bytes = id.as_bytes();
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_price_checked_add() {
        let price1 = Price::new(1_000_000_000);
        let price2 = Price::new(500_000_000);

        let result = price1.checked_add(price2);
        assert!(result.is_some());
        if let Some(result) = result {
            assert_eq!(result.value(), 1_500_000_000);
        }
    }

    #[test]
    fn test_price_checked_add_overflow() {
        let price1 = Price::new(u64::MAX);
        let price2 = Price::new(1);

        let result = price1.checked_add(price2);
        assert!(result.is_none());
    }

    #[test]
    fn test_txid_default() {
        // Test Default implementation for TxId (lines 45-46)
        let id1 = TxId::default();
        let id2 = TxId::default();

        // Should create different IDs each time
        assert_ne!(id1, id2);

        // Should have 32 bytes
        assert_eq!(id1.as_bytes().len(), 32);
    }
}
