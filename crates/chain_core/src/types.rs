//! Common types used across chain core modules
//!
//! This module defines shared data structures and types used throughout
//! the chain core system for maximum performance and type safety.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chain identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u64)]
pub enum ChainId {
    Ethereum = 1,
    Bsc = 56,
    Polygon = 137,
    Arbitrum = 42161,
    Optimism = 10,
    Base = 8453,
    Avalanche = 43114,
}

impl ChainId {
    /// Convert to u64
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self as u64
    }
    
    /// Get chain name
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ethereum => "ethereum",
            Self::Bsc => "bsc",
            Self::Polygon => "polygon",
            Self::Arbitrum => "arbitrum",
            Self::Optimism => "optimism",
            Self::Base => "base",
            Self::Avalanche => "avalanche",
        }
    }
    
    /// Get native token symbol
    #[must_use]
    pub const fn native_token(self) -> &'static str {
        match self {
            Self::Bsc => "BNB",
            Self::Polygon => "MATIC",
            Self::Avalanche => "AVAX",
            Self::Ethereum | Self::Arbitrum | Self::Optimism | Self::Base => "ETH",
        }
    }
    
    /// Check if chain is L2
    #[must_use]
    pub const fn is_l2(self) -> bool {
        matches!(self, Self::Arbitrum | Self::Optimism | Self::Base)
    }
}

impl TryFrom<u64> for ChainId {
    type Error = crate::ChainCoreError;
    
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ethereum),
            56 => Ok(Self::Bsc),
            137 => Ok(Self::Polygon),
            42161 => Ok(Self::Arbitrum),
            10 => Ok(Self::Optimism),
            8453 => Ok(Self::Base),
            43114 => Ok(Self::Avalanche),
            _ => Err(crate::ChainCoreError::UnsupportedChain { chain_id: value }),
        }
    }
}

/// Token address (20 bytes for EVM chains)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenAddress(pub [u8; 20]);

impl TokenAddress {
    /// Zero address (ETH/native token)
    pub const ZERO: Self = Self([0_u8; 20]);
    
    /// Create from hex string
    ///
    /// # Errors
    ///
    /// Returns error if hex string is invalid format or length
    pub fn from_hex(hex: &str) -> Result<Self, crate::ChainCoreError> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        if hex.len() != 40 {
            return Err(crate::ChainCoreError::Configuration(
                "Invalid token address length".to_string()
            ));
        }
        
        let mut bytes = [0_u8; 20];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk)
                .map_err(|_| crate::ChainCoreError::Configuration("Invalid hex".to_string()))?;
            if let Some(byte_ref) = bytes.get_mut(i) {
                *byte_ref = u8::from_str_radix(hex_str, 16)
                    .map_err(|_| crate::ChainCoreError::Configuration("Invalid hex digit".to_string()))?;
            }
        }
        
        Ok(Self(bytes))
    }
    
    /// Convert to hex string
    #[must_use]
    pub fn to_hex(&self) -> String {
        format!("0x{}", hex::encode(self.0))
    }
    
    /// Check if this is the zero address (native token)
    #[must_use]
    #[expect(clippy::indexing_slicing, reason = "Const fn cannot use get() method, bounds are checked")]
    pub const fn is_native(&self) -> bool {
        let zero = [0_u8; 20];
        let mut i = 0;
        while i < 20 {
            if self.0[i] != zero[i] {
                return false;
            }
            i += 1;
        }
        true
    }
}

/// Trading pair
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TradingPair {
    pub token_a: TokenAddress,
    pub token_b: TokenAddress,
    pub chain_id: ChainId,
}

impl TradingPair {
    /// Create new trading pair
    #[must_use]
    pub const fn new(token_a: TokenAddress, token_b: TokenAddress, chain_id: ChainId) -> Self {
        Self {
            token_a,
            token_b,
            chain_id,
        }
    }
    
    /// Get normalized pair (token_a < token_b)
    #[must_use]
    pub fn normalized(&self) -> Self {
        if self.token_a.0 < self.token_b.0 {
            *self
        } else {
            Self {
                token_a: self.token_b,
                token_b: self.token_a,
                chain_id: self.chain_id,
            }
        }
    }
}

/// DEX identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DexId {
    // Ethereum DEXs
    UniswapV2 = 0,
    UniswapV3 = 1,
    SushiSwap = 2,
    Curve = 3,
    Balancer = 4,
    
    // BSC DEXs
    PancakeSwapV2 = 10,
    PancakeSwapV3 = 11,
    BiSwap = 12,
    
    // Polygon DEXs
    QuickSwap = 20,
    SushiSwapPolygon = 21,
    CurvePolygon = 22,
    AavePolygon = 23,
    
    // Arbitrum DEXs
    UniswapV3Arbitrum = 30,
    SushiSwapArbitrum = 31,
    CurveArbitrum = 32,
    
    // Optimism DEXs
    UniswapV3Optimism = 40,
    Velodrome = 41,
    
    // Base DEXs
    UniswapV3Base = 50,
    Aerodrome = 51,
    
    // Avalanche DEXs
    TraderJoe = 60,
    PangolinDex = 61,
    AaveAvalanche = 62,
}

impl DexId {
    /// Get DEX name
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::UniswapV2 => "Uniswap V2",
            Self::UniswapV3 => "Uniswap V3",
            Self::SushiSwap => "SushiSwap",
            Self::Curve => "Curve",
            Self::Balancer => "Balancer",
            Self::PancakeSwapV2 => "PancakeSwap V2",
            Self::PancakeSwapV3 => "PancakeSwap V3",
            Self::BiSwap => "BiSwap",
            Self::QuickSwap => "QuickSwap",
            Self::SushiSwapPolygon => "SushiSwap Polygon",
            Self::CurvePolygon => "Curve Polygon",
            Self::AavePolygon => "Aave Polygon",
            Self::UniswapV3Arbitrum => "Uniswap V3 Arbitrum",
            Self::SushiSwapArbitrum => "SushiSwap Arbitrum",
            Self::CurveArbitrum => "Curve Arbitrum",
            Self::UniswapV3Optimism => "Uniswap V3 Optimism",
            Self::Velodrome => "Velodrome",
            Self::UniswapV3Base => "Uniswap V3 Base",
            Self::Aerodrome => "Aerodrome",
            Self::TraderJoe => "TraderJoe",
            Self::PangolinDex => "Pangolin",
            Self::AaveAvalanche => "Aave Avalanche",
        }
    }
    
    /// Get chain ID for this DEX
    #[must_use]
    pub const fn chain_id(self) -> ChainId {
        match self {
            Self::UniswapV2 | Self::UniswapV3 | Self::SushiSwap | Self::Curve | Self::Balancer => ChainId::Ethereum,
            Self::PancakeSwapV2 | Self::PancakeSwapV3 | Self::BiSwap => ChainId::Bsc,
            Self::QuickSwap | Self::SushiSwapPolygon | Self::CurvePolygon | Self::AavePolygon => ChainId::Polygon,
            Self::UniswapV3Arbitrum | Self::SushiSwapArbitrum | Self::CurveArbitrum => ChainId::Arbitrum,
            Self::UniswapV3Optimism | Self::Velodrome => ChainId::Optimism,
            Self::UniswapV3Base | Self::Aerodrome => ChainId::Base,
            Self::TraderJoe | Self::PangolinDex | Self::AaveAvalanche => ChainId::Avalanche,
        }
    }
    
    /// Check if DEX supports concentrated liquidity
    #[must_use]
    pub const fn supports_concentrated_liquidity(self) -> bool {
        matches!(
            self,
            Self::UniswapV3 | Self::PancakeSwapV3 | Self::UniswapV3Arbitrum | 
            Self::UniswapV3Optimism | Self::UniswapV3Base
        )
    }
}

/// Price information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceInfo {
    pub price: Decimal,
    pub liquidity: Decimal,
    pub volume_24h: Decimal,
    pub timestamp: u64,
    pub dex_id: DexId,
    pub pair: TradingPair,
}

/// Gas information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasInfo {
    pub base_fee: u64,
    pub priority_fee: u64,
    pub max_fee: u64,
    pub gas_limit: u64,
    pub timestamp: u64,
    pub chain_id: ChainId,
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Confirmed,
    Failed,
    Reverted,
}

/// Opportunity type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpportunityType {
    Arbitrage,
    Liquidation,
    Sandwich,
    Frontrun,
    Backrun,
    CrossChain,
    FlashLoan,
}

/// MEV opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Opportunity {
    pub id: u64,
    pub opportunity_type: OpportunityType,
    pub pair: TradingPair,
    pub estimated_profit: Decimal,
    pub gas_cost: Decimal,
    pub net_profit: Decimal,
    pub urgency: u8,
    pub deadline: u64,
    pub dex_route: Vec<DexId>,
    pub metadata: HashMap<String, String>,
}

impl Opportunity {
    /// Check if opportunity is profitable after gas costs
    #[must_use]
    pub fn is_profitable(&self) -> bool {
        self.net_profit > Decimal::ZERO
    }
    
    /// Get profit margin percentage
    #[must_use]
    pub fn profit_margin(&self) -> Decimal {
        if self.estimated_profit.is_zero() {
            Decimal::ZERO
        } else {
            (self.net_profit / self.estimated_profit) * Decimal::new(100, 0)
        }
    }
    
    /// Check if opportunity is urgent (high priority)
    #[must_use]
    pub const fn is_urgent(&self) -> bool {
        self.urgency > 200
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[expect(clippy::panic, reason = "Test code may use panic for assertions")]
    fn test_chain_id_conversion() {
        assert_eq!(ChainId::Ethereum.as_u64(), 1);
        assert_eq!(ChainId::Bsc.as_u64(), 56);
        assert_eq!(ChainId::Polygon.as_u64(), 137);
        
        match ChainId::try_from(1) {
            Ok(chain) => assert_eq!(chain, ChainId::Ethereum),
            Err(e) => panic!("Valid chain ID failed: {e}"),
        }
        match ChainId::try_from(56) {
            Ok(chain) => assert_eq!(chain, ChainId::Bsc),
            Err(e) => panic!("Valid chain ID failed: {e}"),
        }
        assert!(ChainId::try_from(999).is_err());
    }
    
    #[test]
    fn test_chain_properties() {
        assert_eq!(ChainId::Ethereum.name(), "ethereum");
        assert_eq!(ChainId::Ethereum.native_token(), "ETH");
        assert!(!ChainId::Ethereum.is_l2());
        
        assert!(ChainId::Arbitrum.is_l2());
        assert!(ChainId::Optimism.is_l2());
        assert!(ChainId::Base.is_l2());
    }
    
    #[test]
    #[expect(clippy::panic, reason = "Test code may use panic for assertions")]
    fn test_token_address() {
        let zero = TokenAddress::ZERO;
        assert!(zero.is_native());
        
        let addr = match TokenAddress::from_hex("0x1234567890123456789012345678901234567890") {
            Ok(address) => address,
            Err(e) => panic!("Valid hex address failed: {e}"),
        };
        assert!(!addr.is_native());
        assert_eq!(addr.to_hex(), "0x1234567890123456789012345678901234567890");
    }
    
    #[test]
    fn test_trading_pair_normalization() {
        let token_a = TokenAddress([1_u8; 20]);
        let token_b = TokenAddress([2_u8; 20]);
        
        let pair1 = TradingPair::new(token_a, token_b, ChainId::Ethereum);
        let pair2 = TradingPair::new(token_b, token_a, ChainId::Ethereum);
        
        assert_eq!(pair1.normalized(), pair2.normalized());
    }
    
    #[test]
    fn test_dex_properties() {
        assert_eq!(DexId::UniswapV3.name(), "Uniswap V3");
        assert_eq!(DexId::UniswapV3.chain_id(), ChainId::Ethereum);
        assert!(DexId::UniswapV3.supports_concentrated_liquidity());
        
        assert_eq!(DexId::PancakeSwapV2.chain_id(), ChainId::Bsc);
        assert!(!DexId::PancakeSwapV2.supports_concentrated_liquidity());
    }
    
    #[test]
    fn test_opportunity_profitability() {
        let opportunity = Opportunity {
            id: 1,
            opportunity_type: OpportunityType::Arbitrage,
            pair: TradingPair::new(TokenAddress::ZERO, TokenAddress([1_u8; 20]), ChainId::Ethereum),
            estimated_profit: Decimal::new(100, 0),
            gas_cost: Decimal::new(20, 0),
            net_profit: Decimal::new(80, 0),
            urgency: 250,
            deadline: 1_640_995_200,
            dex_route: vec![DexId::UniswapV3],
            metadata: HashMap::new(),
        };
        
        assert!(opportunity.is_profitable());
        assert!(opportunity.is_urgent());
        assert_eq!(opportunity.profit_margin(), Decimal::new(80, 0)); // 80%
    }
}
