//! MEV bundle construction for transaction batching.

use crate::Result;
use alloc::{string::String, vec::Vec};

/// MEV bundle containing multiple transactions
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Bundle {
    /// Bundle identifier
    pub id: String,
    /// Transaction hashes in the bundle
    pub transactions: Vec<String>,
    /// Total gas limit
    pub total_gas_limit: u64,
    /// Bundle timestamp
    pub timestamp: u64,
}

/// MEV bundle builder
#[repr(C, align(64))]
#[non_exhaustive]
pub struct BundleBuilder {
    /// Current bundle being built
    current_bundle: Option<Bundle>,
    /// Padding for cache alignment
    padding: [u8; 56],
}

impl BundleBuilder {
    /// Create a new bundle builder
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        return Self {
            current_bundle: None,
            padding: [0; 56],
        };
    }

    /// Start building a new bundle
    #[inline]
    pub fn start_bundle(&mut self, bundle_id: String) {
        self.current_bundle = Some(Bundle {
            id: bundle_id,
            transactions: Vec::with_capacity(10),
            total_gas_limit: 0,
            timestamp: crate::types::get_timestamp_ns(),
        });
    }

    /// Add transaction to current bundle
    ///
    /// # Errors
    ///
    /// Returns an error if no bundle is being built
    #[inline]
    pub fn add_transaction(&mut self, tx_hash: String, gas_limit: u64) -> Result<()> {
        if let Some(ref mut bundle) = self.current_bundle {
            bundle.transactions.push(tx_hash);
            bundle.total_gas_limit += gas_limit;
            return Ok(());
        }
        return Err(crate::HotPathError::InvalidInput("No bundle being built".into()));
    }

    /// Finalize and return the current bundle
    ///
    /// # Errors
    ///
    /// Returns an error if no bundle is being built
    #[inline]
    pub fn finalize_bundle(&mut self) -> Result<Bundle> {
        if let Some(bundle) = self.current_bundle.take() {
            return Ok(bundle);
        }
        return Err(crate::HotPathError::InvalidInput("No bundle being built".into()));
    }
}

impl Default for BundleBuilder {
    #[inline]
    fn default() -> Self {
        return Self::new();
    }
}

/// Initialize bundle builder
///
/// # Errors
///
/// Currently returns `Ok(())` but may return errors in future implementations
#[inline]
pub const fn initialize() -> Result<()> {
    return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_builder() {
        let mut builder = BundleBuilder::new();
        builder.start_bundle("test_bundle".into());
        builder.add_transaction("0x123".into(), 21000).unwrap();
        let bundle = builder.finalize_bundle().unwrap();
        assert_eq!(bundle.transactions.len(), 1);
        assert_eq!(bundle.total_gas_limit, 21000);
    }
}
