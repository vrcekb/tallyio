//! Transaction types for TallyIO core
//!
//! This module provides ultra-performant transaction types with zero-cost abstractions
//! and cache-optimized memory layout for high-frequency trading operations.

use crate::types::{Address, ChainId, Nonce, Priority, Timestamp};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Transaction hash type
pub type TransactionHash = [u8; 32];

/// Price type for gas prices and transaction values
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Price(u64);

impl Price {
    /// Create a new price from wei
    #[must_use]
    pub const fn new(wei: u64) -> Self {
        Self(wei)
    }

    /// Get the price in wei
    #[must_use]
    pub const fn as_wei(self) -> u64 {
        self.0
    }

    /// Create price from gwei
    #[must_use]
    pub const fn from_gwei(gwei: u64) -> Self {
        Self(gwei * 1_000_000_000)
    }

    /// Get the price in gwei
    #[must_use]
    pub const fn as_gwei(self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Create price from ether
    #[must_use]
    pub const fn from_ether(ether: u64) -> Self {
        Self(ether * 1_000_000_000_000_000_000)
    }

    /// Get the price in ether
    #[must_use]
    pub const fn as_ether(self) -> u64 {
        self.0 / 1_000_000_000_000_000_000
    }

    /// Check if price is zero
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Add two prices
    #[must_use]
    pub const fn add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Subtract two prices
    #[must_use]
    pub const fn sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Multiply price by scalar
    #[must_use]
    pub const fn mul(self, scalar: u64) -> Self {
        Self(self.0.saturating_mul(scalar))
    }

    /// Divide price by scalar
    #[must_use]
    pub const fn div(self, scalar: u64) -> Self {
        if scalar == 0 {
            Self(0)
        } else {
            Self(self.0 / scalar)
        }
    }
}

impl Default for Price {
    fn default() -> Self {
        Self(0)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} wei", self.0)
    }
}

impl From<u64> for Price {
    fn from(wei: u64) -> Self {
        Self(wei)
    }
}

impl From<Price> for u64 {
    fn from(price: Price) -> Self {
        price.0
    }
}

/// Gas type for gas limits and usage
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Gas(u64);

impl Gas {
    /// Create a new gas amount
    #[must_use]
    pub const fn new(units: u64) -> Self {
        Self(units)
    }

    /// Get the gas amount in units
    #[must_use]
    pub const fn as_units(self) -> u64 {
        self.0
    }

    /// Check if gas is zero
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Add two gas amounts
    #[must_use]
    pub const fn add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Subtract two gas amounts
    #[must_use]
    pub const fn sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Calculate gas cost at given price
    #[must_use]
    pub const fn cost_at_price(self, price: Price) -> Price {
        Price::new(self.0.saturating_mul(price.as_wei()))
    }

    /// Standard gas limit for simple transfers
    pub const TRANSFER: Self = Self(21_000);
    /// Standard gas limit for ERC-20 transfers
    pub const ERC20_TRANSFER: Self = Self(65_000);
    /// Standard gas limit for Uniswap swaps
    pub const UNISWAP_SWAP: Self = Self(150_000);
}

impl Default for Gas {
    fn default() -> Self {
        Self(0)
    }
}

impl fmt::Display for Gas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} gas", self.0)
    }
}

impl From<u64> for Gas {
    fn from(units: u64) -> Self {
        Self(units)
    }
}

impl From<Gas> for u64 {
    fn from(gas: Gas) -> Self {
        gas.0
    }
}

/// Transaction status
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// Transaction is pending in mempool
    Pending,
    /// Transaction is being processed
    Processing,
    /// Transaction was successfully executed
    Success,
    /// Transaction failed during execution
    Failed,
    /// Transaction was reverted
    Reverted,
    /// Transaction was dropped from mempool
    Dropped,
}

impl TransactionStatus {
    /// Check if transaction is final (not pending or processing)
    #[must_use]
    pub const fn is_final(self) -> bool {
        matches!(
            self,
            Self::Success | Self::Failed | Self::Reverted | Self::Dropped
        )
    }

    /// Check if transaction was successful
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }

    /// Check if transaction failed
    #[must_use]
    pub const fn is_failed(self) -> bool {
        matches!(self, Self::Failed | Self::Reverted)
    }
}

impl Default for TransactionStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl fmt::Display for TransactionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Processing => write!(f, "processing"),
            Self::Success => write!(f, "success"),
            Self::Failed => write!(f, "failed"),
            Self::Reverted => write!(f, "reverted"),
            Self::Dropped => write!(f, "dropped"),
        }
    }
}

/// Transaction type optimized for high-frequency processing
///
/// Memory layout is optimized for cache efficiency with hot fields first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct Transaction {
    // Hot fields - accessed frequently
    /// Transaction ID for internal tracking
    pub id: Uuid,
    /// Transaction hash
    pub hash: Option<TransactionHash>,
    /// Current status
    pub status: TransactionStatus,
    /// Processing priority
    pub priority: Priority,
    /// Timestamp when transaction was created
    pub timestamp: Timestamp,

    // Transaction data
    /// Sender address
    pub from: Address,
    /// Recipient address (None for contract creation)
    pub to: Option<Address>,
    /// Transaction value in wei
    pub value: Price,
    /// Gas price in wei
    pub gas_price: Price,
    /// Gas limit
    pub gas_limit: Gas,
    /// Transaction nonce
    pub nonce: Nonce,
    /// Transaction data/input
    pub data: Vec<u8>,

    // Optional fields
    /// Chain ID
    pub chain_id: Option<ChainId>,
    /// Maximum fee per gas (EIP-1559)
    pub max_fee_per_gas: Option<Price>,
    /// Maximum priority fee per gas (EIP-1559)
    pub max_priority_fee_per_gas: Option<Price>,
}

impl Transaction {
    /// Create a new transaction
    #[must_use]
    pub fn new(
        from: Address,
        to: Option<Address>,
        value: Price,
        gas_price: Price,
        gas_limit: Gas,
        nonce: Nonce,
        data: Vec<u8>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            hash: None,
            status: TransactionStatus::Pending,
            priority: Priority::Normal,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            from,
            to,
            value,
            gas_price,
            gas_limit,
            nonce,
            data,
            chain_id: None,
            max_fee_per_gas: None,
            max_priority_fee_per_gas: None,
        }
    }

    /// Get transaction value
    #[must_use]
    pub const fn value(&self) -> Price {
        self.value
    }

    /// Get gas price
    #[must_use]
    pub const fn gas_price(&self) -> Price {
        self.gas_price
    }

    /// Get gas limit
    #[must_use]
    pub const fn gas_limit(&self) -> Gas {
        self.gas_limit
    }

    /// Calculate maximum transaction cost
    #[must_use]
    pub fn max_cost(&self) -> Price {
        self.value.add(self.gas_limit.cost_at_price(self.gas_price))
    }

    /// Check if transaction has data (is contract interaction)
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.data.is_empty()
    }

    /// Check if transaction is contract creation
    #[must_use]
    pub const fn is_contract_creation(&self) -> bool {
        self.to.is_none()
    }

    /// Check if transaction is DeFi-related based on data
    #[must_use]
    pub fn is_defi_related(&self) -> bool {
        if self.data.len() < 4 {
            return false;
        }

        // Check for common DeFi function selectors
        let selector = &self.data[0..4];
        matches!(
            selector,
            // Uniswap V2/V3 selectors
            [0xa9, 0x05, 0x9c, 0xbb] | // swapExactTokensForTokens
            [0x38, 0xed, 0x17, 0x39] | // swapExactTokensForETH
            [0x7f, 0xf3, 0x6a, 0xb5] | // swapExactETHForTokens
            [0x41, 0x44, 0x1d, 0x67] | // exactInputSingle (V3)
            // Curve selectors
            [0x3d, 0xf0, 0x21, 0x24] | // exchange
            [0x2e, 0x1a, 0x7d, 0x4d] | // exchange_underlying (different signature)
            // 1inch selectors
            [0x7c, 0x02, 0x5e, 0x60] | // swap
            [0x2e, 0x95, 0xb6, 0xc8] // unoswap
        )
    }

    /// Set transaction hash
    pub fn set_hash(&mut self, hash: TransactionHash) {
        self.hash = Some(hash);
    }

    /// Set transaction status
    pub fn set_status(&mut self, status: TransactionStatus) {
        self.status = status;
    }

    /// Set transaction priority
    pub fn set_priority(&mut self, priority: Priority) {
        self.priority = priority;
    }

    /// Set chain ID
    pub fn set_chain_id(&mut self, chain_id: ChainId) {
        self.chain_id = Some(chain_id);
    }

    /// Set EIP-1559 fees
    pub fn set_eip1559_fees(&mut self, max_fee: Price, max_priority_fee: Price) {
        self.max_fee_per_gas = Some(max_fee);
        self.max_priority_fee_per_gas = Some(max_priority_fee);
    }

    /// Check if transaction uses EIP-1559
    #[must_use]
    pub fn is_eip1559(&self) -> bool {
        self.max_fee_per_gas.is_some() && self.max_priority_fee_per_gas.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_operations() {
        let price1 = Price::new(1000);
        let price2 = Price::new(500);

        assert_eq!(price1.add(price2), Price::new(1500));
        assert_eq!(price1.sub(price2), Price::new(500));
        assert_eq!(price1.mul(2), Price::new(2000));
        assert_eq!(price1.div(2), Price::new(500));
    }

    #[test]
    fn test_price_conversions() {
        let price = Price::from_gwei(20);
        assert_eq!(price.as_gwei(), 20);
        assert_eq!(price.as_wei(), 20_000_000_000);

        let eth_price = Price::from_ether(1);
        assert_eq!(eth_price.as_ether(), 1);
        assert_eq!(eth_price.as_wei(), 1_000_000_000_000_000_000);
    }

    #[test]
    fn test_gas_operations() {
        let gas1 = Gas::new(21000);
        let gas2 = Gas::new(10000);
        let price = Price::from_gwei(20);

        assert_eq!(gas1.add(gas2), Gas::new(31000));
        assert_eq!(gas1.sub(gas2), Gas::new(11000));

        let cost = gas1.cost_at_price(price);
        assert_eq!(cost.as_wei(), 21000 * 20_000_000_000);
    }

    #[test]
    fn test_transaction_creation() {
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21000),
            0,
            Vec::with_capacity(0),
        );

        assert_eq!(tx.value().as_ether(), 1);
        assert_eq!(tx.gas_price().as_gwei(), 20);
        assert_eq!(tx.gas_limit().as_units(), 21000);
        assert!(!tx.is_contract_creation());
        assert!(!tx.has_data());
    }

    #[test]
    fn test_defi_detection() {
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(0),
            Price::from_gwei(20),
            Gas::new(150000),
            0,
            vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // swapExactTokensForTokens
        );

        assert!(tx.is_defi_related());

        tx.data = vec![0x12, 0x34, 0x56, 0x78]; // Non-DeFi selector
        assert!(!tx.is_defi_related());
    }

    #[test]
    fn test_transaction_status() {
        let status = TransactionStatus::Pending;
        assert!(!status.is_final());
        assert!(!status.is_success());
        assert!(!status.is_failed());

        let success = TransactionStatus::Success;
        assert!(success.is_final());
        assert!(success.is_success());
        assert!(!success.is_failed());

        let failed = TransactionStatus::Failed;
        assert!(failed.is_final());
        assert!(!failed.is_success());
        assert!(failed.is_failed());
    }

    #[test]
    fn test_eip1559_transaction() {
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21000),
            0,
            Vec::with_capacity(0),
        );

        assert!(!tx.is_eip1559());

        tx.set_eip1559_fees(Price::from_gwei(30), Price::from_gwei(2));
        assert!(tx.is_eip1559());
    }

    #[test]
    fn test_price_edge_cases() {
        // Test zero price
        let zero_price = Price::new(0);
        assert!(zero_price.is_zero());
        assert_eq!(zero_price.as_wei(), 0);
        assert_eq!(zero_price.as_gwei(), 0);
        assert_eq!(zero_price.as_ether(), 0);

        // Test division by zero
        let price = Price::new(1000);
        let result = price.div(0);
        assert_eq!(result.as_wei(), 0);

        // Test saturating operations
        let max_price = Price::new(u64::MAX);
        let overflow_add = max_price.add(Price::new(1));
        assert_eq!(overflow_add.as_wei(), u64::MAX);

        let overflow_mul = max_price.mul(2);
        assert_eq!(overflow_mul.as_wei(), u64::MAX);

        // Test underflow
        let small_price = Price::new(100);
        let large_price = Price::new(1000);
        let underflow = small_price.sub(large_price);
        assert_eq!(underflow.as_wei(), 0);
    }

    #[test]
    fn test_gas_edge_cases() {
        // Test zero gas
        let zero_gas = Gas::new(0);
        assert!(zero_gas.is_zero());
        assert_eq!(zero_gas.as_units(), 0);

        // Test gas constants
        assert_eq!(Gas::TRANSFER.as_units(), 21_000);
        assert_eq!(Gas::ERC20_TRANSFER.as_units(), 65_000);
        assert_eq!(Gas::UNISWAP_SWAP.as_units(), 150_000);

        // Test saturating operations
        let max_gas = Gas::new(u64::MAX);
        let overflow_add = max_gas.add(Gas::new(1));
        assert_eq!(overflow_add.as_units(), u64::MAX);

        // Test underflow
        let small_gas = Gas::new(100);
        let large_gas = Gas::new(1000);
        let underflow = small_gas.sub(large_gas);
        assert_eq!(underflow.as_units(), 0);

        // Test cost calculation with high values
        let high_gas = Gas::new(1_000_000);
        let high_price = Price::new(1_000_000_000);
        let cost = high_gas.cost_at_price(high_price);
        assert_eq!(cost.as_wei(), 1_000_000 * 1_000_000_000);
    }

    #[test]
    fn test_transaction_status_display() {
        assert_eq!(format!("{}", TransactionStatus::Pending), "pending");
        assert_eq!(format!("{}", TransactionStatus::Processing), "processing");
        assert_eq!(format!("{}", TransactionStatus::Success), "success");
        assert_eq!(format!("{}", TransactionStatus::Failed), "failed");
        assert_eq!(format!("{}", TransactionStatus::Reverted), "reverted");
        assert_eq!(format!("{}", TransactionStatus::Dropped), "dropped");
    }

    #[test]
    fn test_price_display() {
        let price = Price::new(1000);
        assert_eq!(format!("{}", price), "1000 wei");
    }

    #[test]
    fn test_gas_display() {
        let gas = Gas::new(21000);
        assert_eq!(format!("{}", gas), "21000 gas");
    }

    #[test]
    fn test_price_from_conversions() {
        let price_from_u64: Price = 1000u64.into();
        assert_eq!(price_from_u64.as_wei(), 1000);

        let u64_from_price: u64 = price_from_u64.into();
        assert_eq!(u64_from_price, 1000);
    }

    #[test]
    fn test_gas_from_conversions() {
        let gas_from_u64: Gas = 21000u64.into();
        assert_eq!(gas_from_u64.as_units(), 21000);

        let u64_from_gas: u64 = gas_from_u64.into();
        assert_eq!(u64_from_gas, 21000);
    }

    #[test]
    fn test_transaction_setters() {
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21000),
            0,
            Vec::with_capacity(0),
        );

        // Test set_hash
        let hash = [0x42u8; 32];
        tx.set_hash(hash);
        assert_eq!(tx.hash, Some(hash));

        // Test set_status
        tx.set_status(TransactionStatus::Processing);
        assert_eq!(tx.status, TransactionStatus::Processing);

        // Test set_priority
        tx.set_priority(Priority::High);
        assert_eq!(tx.priority, Priority::High);

        // Test set_chain_id
        tx.set_chain_id(ChainId::new(1));
        assert_eq!(tx.chain_id, Some(ChainId::new(1)));
    }

    #[test]
    fn test_transaction_contract_creation() {
        let tx = Transaction::new(
            [1u8; 20],
            None, // No recipient = contract creation
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(2_000_000),
            0,
            vec![0x60, 0x80, 0x60, 0x40], // Contract bytecode
        );

        assert!(tx.is_contract_creation());
        assert!(tx.has_data());
    }

    #[test]
    fn test_transaction_max_cost() {
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1), // 1 ETH value
            Price::from_gwei(20), // 20 gwei gas price
            Gas::new(21000),      // 21k gas limit
            0,
            Vec::with_capacity(0),
        );

        let max_cost = tx.max_cost();
        let expected_gas_cost = Gas::new(21000).cost_at_price(Price::from_gwei(20));
        let expected_total = Price::from_ether(1).add(expected_gas_cost);
        assert_eq!(max_cost, expected_total);
    }

    #[test]
    fn test_defi_selectors_comprehensive() {
        let test_cases = vec![
            // Uniswap V2/V3 selectors
            ([0xa9, 0x05, 0x9c, 0xbb], true), // swapExactTokensForTokens
            ([0x38, 0xed, 0x17, 0x39], true), // swapExactTokensForETH
            ([0x7f, 0xf3, 0x6a, 0xb5], true), // swapExactETHForTokens
            ([0x41, 0x44, 0x1d, 0x67], true), // exactInputSingle (V3)
            // Curve selectors
            ([0x3d, 0xf0, 0x21, 0x24], true), // exchange
            ([0x2e, 0x1a, 0x7d, 0x4d], true), // exchange_underlying
            // 1inch selectors
            ([0x7c, 0x02, 0x5e, 0x60], true), // swap
            ([0x2e, 0x95, 0xb6, 0xc8], true), // unoswap
            // Non-DeFi selectors
            ([0xa9, 0x05, 0x9c, 0xbc], false), // Similar but different
            ([0x12, 0x34, 0x56, 0x78], false), // Random selector
        ];

        for (selector, expected) in test_cases {
            let mut data = selector.to_vec();
            data.extend_from_slice(&[0x00, 0x00]); // Add some padding

            let tx = Transaction::new(
                [1u8; 20],
                Some([2u8; 20]),
                Price::new(0),
                Price::from_gwei(20),
                Gas::new(150000),
                0,
                data,
            );

            assert_eq!(
                tx.is_defi_related(),
                expected,
                "Failed for selector {:?}",
                selector
            );
        }
    }

    #[test]
    fn test_defi_detection_insufficient_data() {
        // Test with less than 4 bytes of data
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(0),
            Price::from_gwei(20),
            Gas::new(150000),
            0,
            vec![0xa9, 0x05, 0x9c], // Only 3 bytes
        );

        assert!(!tx.is_defi_related());

        // Test with empty data
        let tx_empty = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::new(0),
            Price::from_gwei(20),
            Gas::new(21000),
            0,
            Vec::new(),
        );

        assert!(!tx_empty.is_defi_related());
    }

    #[test]
    fn test_defaults() {
        assert_eq!(Price::default(), Price::new(0));
        assert_eq!(Gas::default(), Gas::new(0));
        assert_eq!(TransactionStatus::default(), TransactionStatus::Pending);
    }

    #[test]
    fn test_eip1559_partial_fees() {
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21000),
            0,
            Vec::with_capacity(0),
        );

        // Set only max_fee_per_gas
        tx.max_fee_per_gas = Some(Price::from_gwei(30));
        assert!(!tx.is_eip1559()); // Should be false because max_priority_fee_per_gas is None

        // Set only max_priority_fee_per_gas
        tx.max_fee_per_gas = None;
        tx.max_priority_fee_per_gas = Some(Price::from_gwei(2));
        assert!(!tx.is_eip1559()); // Should be false because max_fee_per_gas is None

        // Set both
        tx.max_fee_per_gas = Some(Price::from_gwei(30));
        tx.max_priority_fee_per_gas = Some(Price::from_gwei(2));
        assert!(tx.is_eip1559()); // Should be true because both are set
    }
}
