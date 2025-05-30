//! MEV opportunity types for TallyIO core
//!
//! This module provides types for representing and analyzing MEV (Maximal Extractable Value)
//! opportunities with ultra-high performance characteristics.

use crate::types::{Address, Gas, Price, Priority, Timestamp, TransactionHash};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Type of MEV opportunity
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpportunityType {
    /// Arbitrage opportunity between different exchanges
    Arbitrage,
    /// Liquidation opportunity in lending protocols
    Liquidation,
    /// Sandwich attack opportunity
    Sandwich,
    /// Front-running opportunity
    Frontrun,
    /// Back-running opportunity
    Backrun,
    /// Flash loan opportunity
    FlashLoan,
    /// Cross-chain arbitrage
    CrossChain,
    /// NFT arbitrage
    NftArbitrage,
    /// Custom opportunity type
    Custom(u8),
}

impl OpportunityType {
    /// Get the priority level for this opportunity type
    #[must_use]
    pub const fn priority(self) -> Priority {
        match self {
            Self::Liquidation => Priority::Critical,
            Self::Arbitrage | Self::FlashLoan => Priority::High,
            Self::Sandwich | Self::Frontrun => Priority::High,
            Self::Backrun | Self::CrossChain => Priority::Normal,
            Self::NftArbitrage => Priority::Low,
            Self::Custom(_) => Priority::Normal,
        }
    }

    /// Get the expected profit margin for this opportunity type
    #[must_use]
    pub const fn expected_margin(self) -> u16 {
        match self {
            Self::Arbitrage => 50,      // 0.5%
            Self::Liquidation => 500,   // 5%
            Self::Sandwich => 100,      // 1%
            Self::Frontrun => 200,      // 2%
            Self::Backrun => 30,        // 0.3%
            Self::FlashLoan => 150,     // 1.5%
            Self::CrossChain => 300,    // 3%
            Self::NftArbitrage => 1000, // 10%
            Self::Custom(_) => 100,     // 1%
        }
    }

    /// Check if this opportunity type is time-sensitive
    #[must_use]
    pub const fn is_time_sensitive(self) -> bool {
        matches!(
            self,
            Self::Liquidation | Self::Sandwich | Self::Frontrun | Self::Backrun
        )
    }

    /// Get the maximum acceptable latency for this opportunity type in microseconds
    #[must_use]
    pub const fn max_latency_us(self) -> u64 {
        match self {
            Self::Liquidation => 100,    // 100μs
            Self::Sandwich => 200,       // 200μs
            Self::Frontrun => 150,       // 150μs
            Self::Backrun => 300,        // 300μs
            Self::Arbitrage => 500,      // 500μs
            Self::FlashLoan => 1000,     // 1ms
            Self::CrossChain => 5000,    // 5ms
            Self::NftArbitrage => 10000, // 10ms
            Self::Custom(_) => 1000,     // 1ms
        }
    }
}

impl fmt::Display for OpportunityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arbitrage => write!(f, "arbitrage"),
            Self::Liquidation => write!(f, "liquidation"),
            Self::Sandwich => write!(f, "sandwich"),
            Self::Frontrun => write!(f, "frontrun"),
            Self::Backrun => write!(f, "backrun"),
            Self::FlashLoan => write!(f, "flashloan"),
            Self::CrossChain => write!(f, "crosschain"),
            Self::NftArbitrage => write!(f, "nft_arbitrage"),
            Self::Custom(id) => write!(f, "custom_{id}"),
        }
    }
}

/// MEV opportunity with cache-optimized layout
///
/// Memory layout is optimized for cache efficiency with hot fields first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct Opportunity {
    // Hot fields - accessed frequently during processing
    /// Unique opportunity ID
    pub id: Uuid,
    /// Type of opportunity
    pub opportunity_type: OpportunityType,
    /// Expected profit value in wei
    pub value: Price,
    /// Estimated gas cost
    pub gas_cost: Gas,
    /// Processing priority
    pub priority: Priority,
    /// Timestamp when opportunity was discovered
    pub timestamp: Timestamp,

    // Opportunity details
    /// Target transaction hash that triggered this opportunity
    pub target_tx: Option<TransactionHash>,
    /// Source address for the opportunity
    pub source: Option<Address>,
    /// Target address for the opportunity
    pub target: Option<Address>,
    /// Token addresses involved
    pub tokens: Vec<Address>,
    /// Exchange addresses involved
    pub exchanges: Vec<Address>,

    // Execution parameters
    /// Minimum profit threshold in wei
    pub min_profit: Price,
    /// Maximum gas price willing to pay
    pub max_gas_price: Price,
    /// Deadline for execution (timestamp)
    pub deadline: Option<Timestamp>,
    /// Slippage tolerance in basis points (1 = 0.01%)
    pub slippage_tolerance: u16,

    // Metadata
    /// Additional data for opportunity execution
    pub data: Vec<u8>,
    /// Confidence score (0-100)
    pub confidence: u8,
    /// Risk score (0-100, higher = riskier)
    pub risk_score: u8,
}

impl Opportunity {
    /// Create a new opportunity
    #[must_use]
    pub fn new(opportunity_type: OpportunityType, value: Price, gas_cost: Gas) -> Self {
        Self {
            id: Uuid::new_v4(),
            opportunity_type,
            value,
            gas_cost,
            priority: opportunity_type.priority(),
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            target_tx: None,
            source: None,
            target: None,
            tokens: Vec::with_capacity(2),
            exchanges: Vec::with_capacity(2),
            min_profit: Price::new(0),
            max_gas_price: Price::from_gwei(100), // 100 gwei default
            deadline: None,
            slippage_tolerance: 50, // 0.5% default
            data: Vec::with_capacity(0),
            confidence: 80, // 80% default confidence
            risk_score: 30, // 30% default risk
        }
    }

    /// Get the opportunity value
    #[must_use]
    pub const fn value(&self) -> Price {
        self.value
    }

    /// Get the gas cost
    #[must_use]
    pub const fn gas_cost(&self) -> Gas {
        self.gas_cost
    }

    /// Calculate net profit after gas costs
    #[must_use]
    pub fn net_profit(&self, gas_price: Price) -> Price {
        let total_cost = self.gas_cost.cost_at_price(gas_price);
        self.value.sub(total_cost)
    }

    /// Check if opportunity is profitable at given gas price
    #[must_use]
    pub fn is_profitable(&self, gas_price: Price) -> bool {
        let net = self.net_profit(gas_price);
        net.as_wei() > self.min_profit.as_wei()
    }

    /// Check if opportunity is still valid (not expired)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if let Some(deadline) = self.deadline {
            let now = chrono::Utc::now().timestamp_millis() as u64;
            now <= deadline
        } else {
            true
        }
    }

    /// Check if opportunity is time-sensitive
    #[must_use]
    pub fn is_time_sensitive(&self) -> bool {
        self.opportunity_type.is_time_sensitive()
    }

    /// Get maximum acceptable latency for this opportunity
    #[must_use]
    pub fn max_latency_us(&self) -> u64 {
        self.opportunity_type.max_latency_us()
    }

    /// Calculate profit margin in basis points
    #[must_use]
    pub fn profit_margin(&self, gas_price: Price) -> u16 {
        let net_profit = self.net_profit(gas_price);
        if self.value.is_zero() {
            0
        } else {
            // Use u128 to avoid overflow in intermediate calculations
            let net_wei = net_profit.as_wei() as u128;
            let value_wei = self.value.as_wei() as u128;
            let profit_ratio = (net_wei * 10000) / value_wei;
            profit_ratio.min(10000) as u16 // Cap at 100%
        }
    }

    /// Check if opportunity meets minimum confidence threshold
    #[must_use]
    pub fn meets_confidence_threshold(&self, threshold: u8) -> bool {
        self.confidence >= threshold
    }

    /// Check if opportunity is within acceptable risk level
    #[must_use]
    pub fn within_risk_tolerance(&self, max_risk: u8) -> bool {
        self.risk_score <= max_risk
    }

    /// Set target transaction
    pub fn set_target_tx(&mut self, tx_hash: TransactionHash) {
        self.target_tx = Some(tx_hash);
    }

    /// Set source and target addresses
    pub fn set_addresses(&mut self, source: Address, target: Address) {
        self.source = Some(source);
        self.target = Some(target);
    }

    /// Add token address
    pub fn add_token(&mut self, token: Address) {
        if !self.tokens.contains(&token) {
            self.tokens.push(token);
        }
    }

    /// Add exchange address
    pub fn add_exchange(&mut self, exchange: Address) {
        if !self.exchanges.contains(&exchange) {
            self.exchanges.push(exchange);
        }
    }

    /// Set execution parameters
    pub fn set_execution_params(
        &mut self,
        min_profit: Price,
        max_gas_price: Price,
        deadline: Option<Timestamp>,
        slippage_tolerance: u16,
    ) {
        self.min_profit = min_profit;
        self.max_gas_price = max_gas_price;
        self.deadline = deadline;
        self.slippage_tolerance = slippage_tolerance;
    }

    /// Set confidence and risk scores
    pub fn set_scores(&mut self, confidence: u8, risk_score: u8) {
        self.confidence = confidence.min(100);
        self.risk_score = risk_score.min(100);
    }

    /// Set additional execution data
    pub fn set_data(&mut self, data: Vec<u8>) {
        self.data = data;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opportunity_type_properties() {
        let liquidation = OpportunityType::Liquidation;
        assert_eq!(liquidation.priority(), Priority::Critical);
        assert_eq!(liquidation.expected_margin(), 500);
        assert!(liquidation.is_time_sensitive());
        assert_eq!(liquidation.max_latency_us(), 100);

        let arbitrage = OpportunityType::Arbitrage;
        assert_eq!(arbitrage.priority(), Priority::High);
        assert_eq!(arbitrage.expected_margin(), 50);
        assert!(!arbitrage.is_time_sensitive());
    }

    #[test]
    fn test_opportunity_creation() {
        let opp = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        assert_eq!(opp.opportunity_type, OpportunityType::Arbitrage);
        assert_eq!(opp.value().as_ether(), 1);
        assert_eq!(opp.gas_cost().as_units(), 100_000);
        assert_eq!(opp.priority, Priority::High);
    }

    #[test]
    fn test_profit_calculations() {
        let opp = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        let gas_price = Price::from_gwei(20);
        let net_profit = opp.net_profit(gas_price);
        let expected_cost = 100_000 * 20_000_000_000; // 100k gas * 20 gwei
        let expected_net = 1_000_000_000_000_000_000 - expected_cost;

        assert_eq!(net_profit.as_wei(), expected_net);
        assert!(opp.is_profitable(gas_price));

        let margin = opp.profit_margin(gas_price);
        assert!(margin > 0);
    }

    #[test]
    fn test_opportunity_validation() {
        let mut opp = Opportunity::new(
            OpportunityType::Liquidation,
            Price::from_ether(1),
            Gas::new(50_000),
        );

        assert!(opp.is_valid()); // No deadline set

        let future_deadline = chrono::Utc::now().timestamp_millis() as u64 + 10_000;
        opp.deadline = Some(future_deadline);
        assert!(opp.is_valid());

        let past_deadline = chrono::Utc::now().timestamp_millis() as u64 - 1000;
        opp.deadline = Some(past_deadline);
        assert!(!opp.is_valid());
    }

    #[test]
    fn test_confidence_and_risk() {
        let mut opp = Opportunity::new(
            OpportunityType::Sandwich,
            Price::from_ether(1),
            Gas::new(150_000),
        );

        opp.set_scores(90, 20);
        assert!(opp.meets_confidence_threshold(80));
        assert!(!opp.meets_confidence_threshold(95));
        assert!(opp.within_risk_tolerance(30));
        assert!(!opp.within_risk_tolerance(15));
    }

    #[test]
    fn test_opportunity_addresses() {
        let mut opp = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        let token1 = [1u8; 20];
        let token2 = [2u8; 20];
        let exchange1 = [3u8; 20];

        opp.add_token(token1);
        opp.add_token(token2);
        opp.add_token(token1); // Should not duplicate

        opp.add_exchange(exchange1);

        assert_eq!(opp.tokens.len(), 2);
        assert_eq!(opp.exchanges.len(), 1);
        assert!(opp.tokens.contains(&token1));
        assert!(opp.tokens.contains(&token2));
    }

    #[test]
    fn test_opportunity_type_display() {
        assert_eq!(OpportunityType::Arbitrage.to_string(), "arbitrage");
        assert_eq!(OpportunityType::Liquidation.to_string(), "liquidation");
        assert_eq!(OpportunityType::Sandwich.to_string(), "sandwich");
        assert_eq!(OpportunityType::Frontrun.to_string(), "frontrun");
        assert_eq!(OpportunityType::Backrun.to_string(), "backrun");
        assert_eq!(OpportunityType::FlashLoan.to_string(), "flashloan");
        assert_eq!(OpportunityType::CrossChain.to_string(), "crosschain");
        assert_eq!(OpportunityType::NftArbitrage.to_string(), "nft_arbitrage");
        assert_eq!(OpportunityType::Custom(42).to_string(), "custom_42");
    }

    #[test]
    fn test_all_opportunity_type_properties() {
        // Test all opportunity types for complete coverage
        let types = [
            OpportunityType::Arbitrage,
            OpportunityType::Liquidation,
            OpportunityType::Sandwich,
            OpportunityType::Frontrun,
            OpportunityType::Backrun,
            OpportunityType::FlashLoan,
            OpportunityType::CrossChain,
            OpportunityType::NftArbitrage,
            OpportunityType::Custom(123),
        ];

        for opp_type in types {
            // Test all methods for each type
            let _priority = opp_type.priority();
            let _margin = opp_type.expected_margin();
            let _time_sensitive = opp_type.is_time_sensitive();
            let _latency = opp_type.max_latency_us();
            let _display = opp_type.to_string();
        }
    }

    #[test]
    fn test_opportunity_execution_params() {
        let mut opp = Opportunity::new(
            OpportunityType::FlashLoan,
            Price::from_ether(5),
            Gas::new(200_000),
        );

        let min_profit = Price::from_ether(1);
        let max_gas_price = Price::from_gwei(50);
        let deadline = Some(chrono::Utc::now().timestamp_millis() as u64 + 5000);
        let slippage = 100; // 1%

        opp.set_execution_params(min_profit, max_gas_price, deadline, slippage);

        assert_eq!(opp.min_profit, min_profit);
        assert_eq!(opp.max_gas_price, max_gas_price);
        assert_eq!(opp.deadline, deadline);
        assert_eq!(opp.slippage_tolerance, slippage);
    }

    #[test]
    fn test_opportunity_target_tx() {
        let mut opp = Opportunity::new(
            OpportunityType::Backrun,
            Price::from_ether(2),
            Gas::new(80_000),
        );

        let tx_hash = [42u8; 32];
        opp.set_target_tx(tx_hash);
        assert_eq!(opp.target_tx, Some(tx_hash));
    }

    #[test]
    fn test_opportunity_addresses_setting() {
        let mut opp = Opportunity::new(
            OpportunityType::CrossChain,
            Price::from_ether(3),
            Gas::new(300_000),
        );

        let source = [10u8; 20];
        let target = [20u8; 20];
        opp.set_addresses(source, target);

        assert_eq!(opp.source, Some(source));
        assert_eq!(opp.target, Some(target));
    }

    #[test]
    fn test_opportunity_data() {
        let mut opp = Opportunity::new(
            OpportunityType::NftArbitrage,
            Price::from_ether(10),
            Gas::new(500_000),
        );

        let data = vec![1, 2, 3, 4, 5];
        opp.set_data(data.clone());
        assert_eq!(opp.data, data);
    }

    #[test]
    fn test_opportunity_scores_clamping() {
        let mut opp = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        // Test clamping to 100
        opp.set_scores(150, 200);
        assert_eq!(opp.confidence, 100);
        assert_eq!(opp.risk_score, 100);

        // Test normal values
        opp.set_scores(75, 25);
        assert_eq!(opp.confidence, 75);
        assert_eq!(opp.risk_score, 25);
    }

    #[test]
    fn test_profit_margin_edge_cases() {
        // Test with zero value
        let opp_zero =
            Opportunity::new(OpportunityType::Arbitrage, Price::new(0), Gas::new(100_000));
        assert_eq!(opp_zero.profit_margin(Price::from_gwei(20)), 0);

        // Test with very high gas price that exceeds value
        let opp = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(1_000_000), // 1M gas units
        );
        let high_gas_price = Price::from_gwei(2000); // Very high gas price
        let gas_cost = opp.gas_cost.cost_at_price(high_gas_price);

        // Ensure gas cost exceeds value for this test
        assert!(gas_cost.as_wei() > opp.value.as_wei());

        let margin = opp.profit_margin(high_gas_price);
        // Should be 0 for negative profit (saturating_sub returns 0)
        assert_eq!(margin, 0);
    }

    #[test]
    fn test_opportunity_time_sensitive_methods() {
        let time_sensitive = Opportunity::new(
            OpportunityType::Liquidation,
            Price::from_ether(1),
            Gas::new(50_000),
        );
        assert!(time_sensitive.is_time_sensitive());
        assert_eq!(time_sensitive.max_latency_us(), 100);

        let not_time_sensitive = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );
        assert!(!not_time_sensitive.is_time_sensitive());
        assert_eq!(not_time_sensitive.max_latency_us(), 500);
    }

    #[test]
    fn test_opportunity_profitability_edge_cases() {
        let mut opp = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(100_000),
        );

        // Set minimum profit threshold
        opp.min_profit = Price::from_ether(1); // Very high threshold

        let gas_price = Price::from_gwei(20);
        // Should not be profitable due to high min_profit threshold
        assert!(!opp.is_profitable(gas_price));

        // Lower the threshold
        opp.min_profit = Price::new(0);
        assert!(opp.is_profitable(gas_price));
    }
}
