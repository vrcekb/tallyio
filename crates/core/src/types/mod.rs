//! Core types for TallyIO engine
//!
//! This module contains all fundamental types used throughout the TallyIO system.
//! All types are designed for ultra-high performance with zero-cost abstractions.

pub mod opportunity;
pub mod result;
pub mod transaction;

// Re-export all public types
pub use opportunity::{Opportunity, OpportunityType};
pub use result::ProcessingResult;
pub use transaction::{Gas, Price, Transaction, TransactionHash, TransactionStatus};

use serde::{Deserialize, Serialize};
use std::fmt;

/// Address type for Ethereum addresses
pub type Address = [u8; 20];

/// Block number type
pub type BlockNumber = u64;

/// Nonce type for transactions
pub type Nonce = u64;

/// Timestamp type in milliseconds since Unix epoch
pub type Timestamp = u64;

/// Chain ID type for different blockchain networks
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChainId(pub u64);

impl ChainId {
    /// Ethereum mainnet chain ID
    pub const ETHEREUM: Self = Self(1);
    /// Polygon mainnet chain ID
    pub const POLYGON: Self = Self(137);
    /// Binance Smart Chain mainnet chain ID
    pub const BSC: Self = Self(56);
    /// Arbitrum One chain ID
    pub const ARBITRUM: Self = Self(42161);
    /// Optimism mainnet chain ID
    pub const OPTIMISM: Self = Self(10);

    /// Create a new chain ID
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the chain ID as u64
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Check if this is a mainnet chain
    #[must_use]
    pub const fn is_mainnet(self) -> bool {
        matches!(
            self.0,
            1 | 137 | 56 | 42161 | 10 | 43114 | 250 | 25 | 100 | 1284
        )
    }

    /// Check if this is a testnet chain
    #[must_use]
    pub const fn is_testnet(self) -> bool {
        !self.is_mainnet()
    }
}

impl fmt::Display for ChainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for ChainId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<ChainId> for u64 {
    fn from(chain_id: ChainId) -> Self {
        chain_id.0
    }
}

/// Priority level for transaction processing
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority - background processing
    Low = 0,
    /// Normal priority - standard processing
    Normal = 1,
    /// High priority - expedited processing
    High = 2,
    /// Critical priority - immediate processing
    Critical = 3,
}

impl Priority {
    /// Get priority as numeric value
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Create priority from numeric value
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Low,
            1 => Self::Normal,
            2 => Self::High,
            _ => Self::Critical,
        }
    }

    /// Check if this is a high priority
    #[must_use]
    pub const fn is_high(self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }

    /// Check if this is critical priority
    #[must_use]
    pub const fn is_critical(self) -> bool {
        matches!(self, Self::Critical)
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Network type for different blockchain networks
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Network {
    /// Ethereum mainnet
    Ethereum,
    /// Polygon mainnet
    Polygon,
    /// Binance Smart Chain
    BinanceSmartChain,
    /// Arbitrum One
    Arbitrum,
    /// Optimism
    Optimism,
    /// Avalanche C-Chain
    Avalanche,
    /// Fantom Opera
    Fantom,
    /// Custom network with chain ID
    Custom(ChainId),
}

impl Network {
    /// Get the chain ID for this network
    #[must_use]
    pub const fn chain_id(self) -> ChainId {
        match self {
            Self::Ethereum => ChainId::ETHEREUM,
            Self::Polygon => ChainId::POLYGON,
            Self::BinanceSmartChain => ChainId::BSC,
            Self::Arbitrum => ChainId::ARBITRUM,
            Self::Optimism => ChainId::OPTIMISM,
            Self::Avalanche => ChainId::new(43114),
            Self::Fantom => ChainId::new(250),
            Self::Custom(chain_id) => chain_id,
        }
    }

    /// Get the network name
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ethereum => "ethereum",
            Self::Polygon => "polygon",
            Self::BinanceSmartChain => "bsc",
            Self::Arbitrum => "arbitrum",
            Self::Optimism => "optimism",
            Self::Avalanche => "avalanche",
            Self::Fantom => "fantom",
            Self::Custom(_) => "custom",
        }
    }

    /// Check if this network supports EIP-1559
    #[must_use]
    pub const fn supports_eip1559(self) -> bool {
        matches!(
            self,
            Self::Ethereum | Self::Polygon | Self::Arbitrum | Self::Optimism
        )
    }

    /// Get the native token symbol
    #[must_use]
    pub const fn native_token(self) -> &'static str {
        match self {
            Self::Ethereum => "ETH",
            Self::Polygon => "MATIC",
            Self::BinanceSmartChain => "BNB",
            Self::Arbitrum => "ETH",
            Self::Optimism => "ETH",
            Self::Avalanche => "AVAX",
            Self::Fantom => "FTM",
            Self::Custom(_) => "UNKNOWN",
        }
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl From<ChainId> for Network {
    fn from(chain_id: ChainId) -> Self {
        match chain_id.as_u64() {
            1 => Self::Ethereum,
            137 => Self::Polygon,
            56 => Self::BinanceSmartChain,
            42161 => Self::Arbitrum,
            10 => Self::Optimism,
            43114 => Self::Avalanche,
            250 => Self::Fantom,
            _ => Self::Custom(chain_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_id() {
        let chain_id = ChainId::ETHEREUM;
        assert_eq!(chain_id.as_u64(), 1);
        assert!(chain_id.is_mainnet());
        assert!(!chain_id.is_testnet());

        let custom_chain = ChainId::new(999);
        assert!(!custom_chain.is_mainnet());
        assert!(custom_chain.is_testnet());
    }

    #[test]
    fn test_priority() {
        let priority = Priority::High;
        assert_eq!(priority.as_u8(), 2);
        assert!(priority.is_high());
        assert!(!priority.is_critical());

        let critical = Priority::Critical;
        assert!(critical.is_critical());
        assert!(critical.is_high());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_network() {
        let network = Network::Ethereum;
        assert_eq!(network.chain_id(), ChainId::ETHEREUM);
        assert_eq!(network.name(), "ethereum");
        assert!(network.supports_eip1559());
        assert_eq!(network.native_token(), "ETH");

        let bsc = Network::BinanceSmartChain;
        assert!(!bsc.supports_eip1559());
        assert_eq!(bsc.native_token(), "BNB");
    }

    #[test]
    fn test_network_from_chain_id() {
        let network = Network::from(ChainId::POLYGON);
        assert_eq!(network, Network::Polygon);

        let custom = Network::from(ChainId::new(999));
        assert!(matches!(custom, Network::Custom(_)));
    }

    #[test]
    fn test_chain_id_display() {
        let chain_id = ChainId::ETHEREUM;
        assert_eq!(format!("{}", chain_id), "1");

        let custom = ChainId::new(999);
        assert_eq!(format!("{}", custom), "999");
    }

    #[test]
    fn test_chain_id_conversions() {
        let chain_id = ChainId::new(42);
        let as_u64: u64 = chain_id.into();
        assert_eq!(as_u64, 42);

        let from_u64 = ChainId::from(123u64);
        assert_eq!(from_u64.as_u64(), 123);
    }

    #[test]
    fn test_chain_id_constants() {
        assert_eq!(ChainId::ETHEREUM.as_u64(), 1);
        assert_eq!(ChainId::POLYGON.as_u64(), 137);
        assert_eq!(ChainId::BSC.as_u64(), 56);
        assert_eq!(ChainId::ARBITRUM.as_u64(), 42161);
        assert_eq!(ChainId::OPTIMISM.as_u64(), 10);
    }

    #[test]
    fn test_priority_from_u8() {
        assert_eq!(Priority::from_u8(0), Priority::Low);
        assert_eq!(Priority::from_u8(1), Priority::Normal);
        assert_eq!(Priority::from_u8(2), Priority::High);
        assert_eq!(Priority::from_u8(3), Priority::Critical);
        assert_eq!(Priority::from_u8(99), Priority::Critical); // Any value >= 3 becomes Critical
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Normal);
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(format!("{}", Priority::Low), "low");
        assert_eq!(format!("{}", Priority::Normal), "normal");
        assert_eq!(format!("{}", Priority::High), "high");
        assert_eq!(format!("{}", Priority::Critical), "critical");
    }

    #[test]
    fn test_network_all_variants() {
        // Test all network variants
        let networks = [
            Network::Ethereum,
            Network::Polygon,
            Network::BinanceSmartChain,
            Network::Arbitrum,
            Network::Optimism,
            Network::Avalanche,
            Network::Fantom,
            Network::Custom(ChainId::new(999)),
        ];

        for network in networks {
            // Test that all methods work
            let _ = network.chain_id();
            let _ = network.name();
            let _ = network.supports_eip1559();
            let _ = network.native_token();
            let _ = format!("{}", network);
        }
    }

    #[test]
    fn test_network_specific_properties() {
        // Test Avalanche
        let avalanche = Network::Avalanche;
        assert_eq!(avalanche.chain_id().as_u64(), 43114);
        assert_eq!(avalanche.name(), "avalanche");
        assert!(!avalanche.supports_eip1559());
        assert_eq!(avalanche.native_token(), "AVAX");

        // Test Fantom
        let fantom = Network::Fantom;
        assert_eq!(fantom.chain_id().as_u64(), 250);
        assert_eq!(fantom.name(), "fantom");
        assert!(!fantom.supports_eip1559());
        assert_eq!(fantom.native_token(), "FTM");

        // Test Custom
        let custom = Network::Custom(ChainId::new(12345));
        assert_eq!(custom.chain_id().as_u64(), 12345);
        assert_eq!(custom.name(), "custom");
        assert!(!custom.supports_eip1559());
        assert_eq!(custom.native_token(), "UNKNOWN");
    }

    #[test]
    fn test_network_display() {
        assert_eq!(format!("{}", Network::Ethereum), "ethereum");
        assert_eq!(format!("{}", Network::Polygon), "polygon");
        assert_eq!(format!("{}", Network::BinanceSmartChain), "bsc");
        assert_eq!(format!("{}", Network::Custom(ChainId::new(999))), "custom");
    }

    #[test]
    fn test_network_from_all_chain_ids() {
        // Test all known chain IDs
        assert_eq!(Network::from(ChainId::new(1)), Network::Ethereum);
        assert_eq!(Network::from(ChainId::new(137)), Network::Polygon);
        assert_eq!(Network::from(ChainId::new(56)), Network::BinanceSmartChain);
        assert_eq!(Network::from(ChainId::new(42161)), Network::Arbitrum);
        assert_eq!(Network::from(ChainId::new(10)), Network::Optimism);
        assert_eq!(Network::from(ChainId::new(43114)), Network::Avalanche);
        assert_eq!(Network::from(ChainId::new(250)), Network::Fantom);

        // Test unknown chain ID
        let unknown = Network::from(ChainId::new(99999));
        assert!(matches!(unknown, Network::Custom(_)));
    }

    #[test]
    fn test_chain_id_mainnet_detection() {
        // Test all mainnet chain IDs mentioned in is_mainnet
        let mainnet_ids = [1, 137, 56, 42161, 10, 43114, 250, 25, 100, 1284];
        for id in mainnet_ids {
            let chain_id = ChainId::new(id);
            assert!(chain_id.is_mainnet(), "Chain ID {} should be mainnet", id);
            assert!(
                !chain_id.is_testnet(),
                "Chain ID {} should not be testnet",
                id
            );
        }

        // Test testnet chain IDs
        let testnet_ids = [3, 4, 5, 42, 97, 80001];
        for id in testnet_ids {
            let chain_id = ChainId::new(id);
            assert!(
                !chain_id.is_mainnet(),
                "Chain ID {} should not be mainnet",
                id
            );
            assert!(chain_id.is_testnet(), "Chain ID {} should be testnet", id);
        }
    }

    #[test]
    fn test_priority_edge_cases() {
        // Test Low priority
        let low = Priority::Low;
        assert_eq!(low.as_u8(), 0);
        assert!(!low.is_high());
        assert!(!low.is_critical());

        // Test Normal priority
        let normal = Priority::Normal;
        assert_eq!(normal.as_u8(), 1);
        assert!(!normal.is_high());
        assert!(!normal.is_critical());
    }
}
