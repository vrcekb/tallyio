//! Utility functions and helpers for chain core operations
//!
//! This module provides common utility functions used across the chain core
//! system for performance optimization and code reuse.

use crate::types::{ChainId, TokenAddress};
use rust_decimal::Decimal;
use std::time::{SystemTime, UNIX_EPOCH};

/// Time utilities
pub mod time {
    use super::*;
    
    /// Get current Unix timestamp in seconds
    #[must_use]
    pub fn now_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    }
    
    /// Get current Unix timestamp in milliseconds
    #[must_use]
    pub fn now_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }
    
    /// Get current Unix timestamp in nanoseconds
    #[must_use]
    pub fn now_timestamp_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
    }
    
    /// Check if timestamp is expired
    #[must_use]
    pub fn is_expired(timestamp: u64, current: Option<u64>) -> bool {
        let current = current.unwrap_or_else(now_timestamp);
        timestamp < current
    }
    
    /// Get time until deadline in seconds
    #[must_use]
    pub fn time_until_deadline(deadline: u64, current: Option<u64>) -> i64 {
        let current = current.unwrap_or_else(now_timestamp);
        i64::try_from(deadline).unwrap_or(i64::MAX) - i64::try_from(current).unwrap_or(i64::MAX)
    }
}

/// Math utilities for financial calculations
pub mod math {
    use super::*;
    
    /// Calculate percentage of a value
    #[must_use]
    pub fn percentage(value: Decimal, percent: Decimal) -> Decimal {
        value * percent / Decimal::new(100, 0)
    }
    
    /// Calculate percentage difference between two values
    #[must_use]
    pub fn percentage_diff(value1: Decimal, value2: Decimal) -> Decimal {
        if value1.is_zero() {
            return Decimal::ZERO;
        }
        ((value2 - value1) / value1).abs() * Decimal::new(100, 0)
    }
    
    /// Calculate slippage percentage
    #[must_use]
    pub fn slippage_percentage(expected: Decimal, actual: Decimal) -> Decimal {
        if expected.is_zero() {
            return Decimal::ZERO;
        }
        ((expected - actual) / expected).abs() * Decimal::new(100, 0)
    }
    
    /// Apply slippage tolerance to a price
    #[must_use]
    pub fn apply_slippage(price: Decimal, slippage_percent: Decimal, is_buy: bool) -> Decimal {
        let slippage_amount = percentage(price, slippage_percent);
        if is_buy {
            price + slippage_amount // Buy: accept higher price
        } else {
            price - slippage_amount // Sell: accept lower price
        }
    }
    
    /// Calculate compound interest
    #[must_use]
    pub fn compound_interest(principal: Decimal, rate: Decimal, periods: u32) -> Decimal {
        let rate_plus_one = Decimal::ONE + rate;
        let mut result = principal;
        for _ in 0..periods {
            result *= rate_plus_one;
        }
        result
    }
    
    /// Calculate annual percentage yield (APY) from APR
    #[must_use]
    pub fn apr_to_apy(apr: Decimal, compounds_per_year: u32) -> Decimal {
        if compounds_per_year == 0 {
            return apr;
        }
        
        let rate_per_period = apr / Decimal::new(compounds_per_year.into(), 0);
        compound_interest(Decimal::ONE, rate_per_period, compounds_per_year) - Decimal::ONE
    }
}

/// Gas calculation utilities
pub mod gas {
    use super::*;
    
    /// Calculate total gas cost in wei
    #[must_use]
    pub const fn calculate_gas_cost(gas_limit: u64, gas_price: u64) -> u64 {
        gas_limit.saturating_mul(gas_price)
    }
    
    /// Calculate EIP-1559 max fee per gas
    #[must_use]
    pub fn calculate_max_fee(base_fee: u64, priority_fee: u64, buffer_percent: u32) -> u64 {
        let total_fee = base_fee.saturating_add(priority_fee);
        let buffer = total_fee.saturating_mul(u64::from(buffer_percent)) / 100;
        total_fee.saturating_add(buffer)
    }
    
    /// Estimate gas limit with buffer
    #[must_use]
    pub fn estimate_gas_with_buffer(base_estimate: u64, buffer_percent: u32) -> u64 {
        let buffer = base_estimate.saturating_mul(u64::from(buffer_percent)) / 100;
        base_estimate.saturating_add(buffer)
    }
    
    /// Convert gas cost from wei to native token units
    #[must_use]
    pub fn wei_to_native(wei: u64) -> Decimal {
        Decimal::new(i64::try_from(wei).unwrap_or(i64::MAX), 18) // 18 decimals for ETH-like tokens
    }
    
    /// Convert native token units to wei
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails or amount is too large
    pub fn native_to_wei(amount: Decimal) -> Result<u64, crate::ChainCoreError> {
        use rust_decimal::prelude::ToPrimitive;

        let wei_amount = amount * Decimal::new(10_i64.pow(18), 0);
        wei_amount.to_u64().ok_or_else(|| {
            crate::ChainCoreError::Internal("Amount too large for u64".to_string())
        })
    }
}

/// Address utilities
pub mod address {
    use super::*;
    
    /// Check if address is valid EVM address
    #[must_use]
    pub fn is_valid_evm_address(address: &str) -> bool {
        let address = address.strip_prefix("0x").unwrap_or(address);
        address.len() == 40 && address.chars().all(|c| c.is_ascii_hexdigit())
    }
    
    /// Normalize address to lowercase
    #[must_use]
    pub fn normalize_address(address: &str) -> String {
        let address = address.strip_prefix("0x").unwrap_or(address);
        format!("0x{}", address.to_lowercase())
    }
    
    /// Get checksum address (EIP-55)
    ///
    /// # Errors
    ///
    /// Returns error if address format is invalid
    pub fn to_checksum_address(address: &str) -> Result<String, crate::ChainCoreError> {
        use sha3::{Digest, Keccak256};
        
        let address = address.strip_prefix("0x").unwrap_or(address).to_lowercase();
        if address.len() != 40 {
            return Err(crate::ChainCoreError::Configuration(
                "Invalid address length".to_string()
            ));
        }
        
        let hash = Keccak256::digest(address.as_bytes());
        let hash_hex = hex::encode(hash);
        
        let mut checksum = String::with_capacity(42);
        checksum.push_str("0x");
        
        for (i, c) in address.chars().enumerate() {
            if c.is_ascii_digit() {
                checksum.push(c);
            } else {
                let hash_char = hash_hex.chars().nth(i).unwrap_or('0');
                if hash_char >= '8' {
                    checksum.push(c.to_ascii_uppercase());
                } else {
                    checksum.push(c);
                }
            }
        }
        
        Ok(checksum)
    }
    
    /// Generate deterministic address from chain and index
    #[must_use]
    pub fn generate_deterministic_address(chain_id: ChainId, index: u64) -> TokenAddress {
        use sha3::{Digest, Keccak256};
        
        let mut hasher = Keccak256::new();
        hasher.update(chain_id.as_u64().to_be_bytes());
        hasher.update(index.to_be_bytes());
        hasher.update(b"TallyIO_Deterministic_Address");
        
        let hash = hasher.finalize();
        let mut address = [0_u8; 20];
        if let Some(slice) = hash.get(12..32) {
            address.copy_from_slice(slice); // Take last 20 bytes
        }
        
        TokenAddress(address)
    }
}

/// Encoding utilities
pub mod encoding {
    use super::*;
    
    /// Encode function selector from signature
    ///
    /// # Errors
    ///
    /// Returns error if signature format is invalid
    pub fn function_selector(signature: &str) -> Result<[u8; 4], crate::ChainCoreError> {
        use sha3::{Digest, Keccak256};
        
        let hash = Keccak256::digest(signature.as_bytes());
        let mut selector = [0_u8; 4];
        if let Some(slice) = hash.get(0..4) {
            selector.copy_from_slice(slice);
        }
        Ok(selector)
    }
    
    /// Encode uint256 for ABI
    #[must_use]
    pub fn encode_uint256(value: u64) -> [u8; 32] {
        let mut encoded = [0_u8; 32];
        encoded[24..32].copy_from_slice(&value.to_be_bytes());
        encoded
    }
    
    /// Encode address for ABI
    #[must_use]
    pub fn encode_address(address: &TokenAddress) -> [u8; 32] {
        let mut encoded = [0_u8; 32];
        encoded[12..32].copy_from_slice(&address.0);
        encoded
    }
    
    /// Decode uint256 from ABI
    ///
    /// # Errors
    ///
    /// Returns error if data format is invalid
    pub fn decode_uint256(data: &[u8]) -> Result<u64, crate::ChainCoreError> {
        if data.len() < 32 {
            return Err(crate::ChainCoreError::Internal(
                "Insufficient data for uint256".to_string()
            ));
        }
        
        let mut bytes = [0_u8; 8];
        if let Some(slice) = data.get(24..32) {
            bytes.copy_from_slice(slice);
        }
        Ok(u64::from_be_bytes(bytes))
    }
    
    /// Decode address from ABI
    ///
    /// # Errors
    ///
    /// Returns error if data format is invalid
    pub fn decode_address(data: &[u8]) -> Result<TokenAddress, crate::ChainCoreError> {
        if data.len() < 32 {
            return Err(crate::ChainCoreError::Internal(
                "Insufficient data for address".to_string()
            ));
        }
        
        let mut address = [0_u8; 20];
        if let Some(slice) = data.get(12..32) {
            address.copy_from_slice(slice);
        }
        Ok(TokenAddress(address))
    }
}

/// Performance utilities
pub mod perf {
    use std::time::Instant;
    
    /// Simple performance timer
    pub struct Timer {
        start: Instant,
        name: String,
    }
    
    impl Timer {
        /// Start new timer
        #[must_use]
        pub fn new<T: Into<String>>(name: T) -> Self {
            Self {
                start: Instant::now(),
                name: name.into(),
            }
        }
        
        /// Get elapsed time in nanoseconds
        #[must_use]
        pub fn elapsed_ns(&self) -> u64 {
            u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }
        
        /// Get elapsed time in microseconds
        #[must_use]
        pub fn elapsed_us(&self) -> u64 {
            u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX)
        }
        
        /// Get elapsed time in milliseconds
        #[must_use]
        pub fn elapsed_ms(&self) -> u64 {
            u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX)
        }
    }
    
    impl Drop for Timer {
        fn drop(&mut self) {
            let elapsed = self.elapsed_us();
            tracing::debug!("Timer '{}' elapsed: {}μs", self.name, elapsed);
        }
    }
    
    /// Measure execution time of a closure
    pub fn measure<F, R>(name: &str, f: F) -> (R, u64)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        tracing::debug!("Operation '{}' took: {}ns", name, elapsed);
        (result, elapsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_utilities() {
        let now = time::now_timestamp();
        assert!(now > 0);
        
        let future = now + 3600; // 1 hour in future
        assert!(!time::is_expired(future, Some(now)));
        assert!(time::is_expired(now - 1, Some(now)));
        
        assert_eq!(time::time_until_deadline(future, Some(now)), 3600);
    }
    
    #[test]
    fn test_math_utilities() {
        let value = Decimal::new(1000, 0);
        let percent = Decimal::new(10, 0);
        
        assert_eq!(math::percentage(value, percent), Decimal::new(100, 0));
        
        let diff = math::percentage_diff(Decimal::new(100, 0), Decimal::new(110, 0));
        assert_eq!(diff, Decimal::new(10, 0));
        
        let slippage = math::slippage_percentage(Decimal::new(100, 0), Decimal::new(95, 0));
        assert_eq!(slippage, Decimal::new(5, 0));
    }
    
    #[test]
    fn test_gas_utilities() {
        let gas_cost = gas::calculate_gas_cost(21000, 20_000_000_000);
        assert_eq!(gas_cost, 420_000_000_000_000);
        
        let max_fee = gas::calculate_max_fee(20_000_000_000, 2_000_000_000, 10);
        assert_eq!(max_fee, 24_200_000_000);
        
        let gas_with_buffer = gas::estimate_gas_with_buffer(21000, 10);
        assert_eq!(gas_with_buffer, 23100);
    }
    
    #[test]
    fn test_address_utilities() {
        assert!(address::is_valid_evm_address("0x1234567890123456789012345678901234567890"));
        assert!(!address::is_valid_evm_address("0x123"));
        assert!(!address::is_valid_evm_address("invalid"));
        
        let normalized = address::normalize_address("0xABCDEF1234567890123456789012345678901234");
        assert_eq!(normalized, "0xabcdef1234567890123456789012345678901234");
        
        let deterministic = address::generate_deterministic_address(ChainId::Ethereum, 1);
        assert!(!deterministic.is_native());
    }
    
    #[test]
    #[expect(clippy::panic, reason = "Test code may use panic for assertions")]
    fn test_encoding_utilities() {
        let selector = match encoding::function_selector("transfer(address,uint256)") {
            Ok(sel) => sel,
            Err(e) => panic!("Valid function signature failed: {e}"),
        };
        assert_eq!(selector.len(), 4);
        
        let encoded = encoding::encode_uint256(12345);
        assert_eq!(encoded.len(), 32);
        match encoding::decode_uint256(&encoded) {
            Ok(value) => assert_eq!(value, 12_345),
            Err(e) => panic!("Valid encoded data failed: {e}"),
        }
        
        let address = TokenAddress([1_u8; 20]);
        let encoded_addr = encoding::encode_address(&address);
        assert_eq!(encoded_addr.len(), 32);
        match encoding::decode_address(&encoded_addr) {
            Ok(decoded_addr) => assert_eq!(decoded_addr, address),
            Err(e) => panic!("Valid encoded address failed: {e}"),
        }
    }
    
    #[test]
    fn test_performance_timer() {
        let timer = perf::Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(timer.elapsed_us() >= 1000); // At least 1ms = 1000μs
    }
}
