//! # Liquidation Engine (Priority 1)
//!
//! Ultra-performance liquidation coordinator for `TallyIO` MEV operations.
//!
//! This module provides real-time health factor monitoring, multi-protocol
//! liquidation orchestration, and profit optimization across `DeFi` protocols.
//!
//! ## Performance Targets
//!
//! - Health factor monitoring: <1μs detection latency
//! - Liquidation opportunity scoring: <100ns
//! - Multi-protocol coordination: <500μs
//! - Profit calculation: <50ns
//! - Gas optimization: <200μs
//!
//! ## Supported Protocols
//!
//! - Aave v3 (Ethereum, Polygon, Arbitrum, Optimism)
//! - Venus (BSC)
//! - Compound v3 (Ethereum, Polygon, Arbitrum)
//! - Custom refinancing strategies
//!
//! ## Architecture
//!
//! The liquidation engine uses a multi-layered approach:
//! 1. Real-time health monitoring with SIMD optimization
//! 2. Protocol-specific liquidation engines
//! 3. Multicall optimization for batch operations
//! 4. Profit calculation with gas cost modeling
//! 5. Smart refinancing to minimize user losses

use crate::{StrategyResult, ChainId, ProfitAmount, BlockNumber, Timestamp};
use rust_decimal::Decimal;
use thiserror::Error;
use tokio::sync::{RwLock, mpsc};
use tokio::time::{sleep, Duration};
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use ahash::AHashMap;

/// Liquidation-specific error types
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum LiquidationError {
    /// Health factor monitoring errors
    #[error("Health monitor error: {message}")]
    HealthMonitor {
        /// Error message
        message: String
    },

    /// Protocol-specific liquidation errors
    #[error("Protocol liquidation error: protocol={protocol}, message={message}")]
    ProtocolLiquidation {
        /// Protocol name
        protocol: String,
        /// Error message
        message: String
    },

    /// Profit calculation errors
    #[error("Profit calculation error: {message}")]
    ProfitCalculation {
        /// Error message
        message: String
    },

    /// Gas estimation errors
    #[error("Gas estimation error: {message}")]
    GasEstimation {
        /// Error message
        message: String
    },

    /// Multicall optimization errors
    #[error("Multicall optimization error: {message}")]
    MulticallOptimization {
        /// Error message
        message: String
    },

    /// Insufficient profit for liquidation
    #[error("Insufficient profit: required={required}, available={available}")]
    InsufficientProfit {
        /// Required profit amount
        required: Decimal,
        /// Available profit amount
        available: Decimal
    },

    /// Liquidation already in progress
    #[error("Liquidation already in progress for position: {position_id}")]
    LiquidationInProgress {
        /// Position identifier
        position_id: String
    },

    /// Position not liquidatable
    #[error("Position not liquidatable: health_factor={health_factor}")]
    PositionNotLiquidatable {
        /// Current health factor
        health_factor: Decimal
    },
}

/// Liquidation opportunity with profit analysis
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LiquidationOpportunity {
    /// Unique position identifier
    pub position_id: String,
    /// Protocol where position exists
    pub protocol: LiquidationProtocol,
    /// Chain ID
    pub chain_id: ChainId,
    /// User address
    pub user_address: [u8; 20],
    /// Collateral token address
    pub collateral_token: [u8; 20],
    /// Debt token address  
    pub debt_token: [u8; 20],
    /// Current health factor
    pub health_factor: Decimal,
    /// Liquidatable debt amount
    pub liquidatable_debt: Decimal,
    /// Collateral to receive
    pub collateral_amount: Decimal,
    /// Estimated profit in USD
    pub estimated_profit: ProfitAmount,
    /// Gas cost estimate
    pub gas_cost: Decimal,
    /// Net profit (profit - gas cost)
    pub net_profit: ProfitAmount,
    /// Liquidation bonus percentage
    pub liquidation_bonus: Decimal,
    /// Urgency score (0-255, higher = more urgent)
    pub urgency: u8,
    /// Block number when detected
    pub block_number: BlockNumber,
    /// Detection timestamp
    pub timestamp: Timestamp,
}

/// Supported liquidation protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum LiquidationProtocol {
    /// Aave v3 protocol
    AaveV3 = 0,
    /// Venus protocol (BSC)
    Venus = 1,
    /// Compound v3 protocol
    CompoundV3 = 2,
    /// Custom refinancing strategy
    Refinance = 3,
}

/// Liquidation execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LiquidationStatus {
    /// Opportunity detected, pending execution
    Pending,
    /// Liquidation in progress
    Executing,
    /// Liquidation completed successfully
    Completed,
    /// Liquidation failed
    Failed,
    /// Liquidation cancelled (e.g., position became healthy)
    Cancelled,
}

/// Liquidation execution result
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LiquidationResult {
    /// Original opportunity
    pub opportunity: LiquidationOpportunity,
    /// Execution status
    pub status: LiquidationStatus,
    /// Transaction hash if executed
    pub transaction_hash: Option<[u8; 32]>,
    /// Actual profit realized
    pub actual_profit: Option<ProfitAmount>,
    /// Actual gas used
    pub gas_used: Option<u64>,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Liquidation coordinator configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LiquidationConfig {
    /// Minimum health factor threshold for liquidation
    pub min_health_factor: Decimal,
    /// Minimum profit threshold in USD
    pub min_profit_threshold: Decimal,
    /// Maximum gas price in gwei
    pub max_gas_price: u64,
    /// Maximum concurrent liquidations
    pub max_concurrent_liquidations: usize,
    /// Health monitoring interval in milliseconds
    pub health_monitor_interval_ms: u64,
    /// Enable multicall optimization
    pub enable_multicall: bool,
    /// Enable refinancing strategies
    pub enable_refinancing: bool,
}

impl Default for LiquidationConfig {
    #[inline]
    fn default() -> Self {
        Self {
            min_health_factor: Decimal::new(100, 2), // 1.00
            min_profit_threshold: Decimal::new(5, 2), // $0.05
            max_gas_price: 100, // 100 gwei
            max_concurrent_liquidations: 50,
            health_monitor_interval_ms: 100, // 100ms
            enable_multicall: true,
            enable_refinancing: true,
        }
    }
}

/// Main liquidation coordinator
#[expect(dead_code, reason = "Fields will be used in full implementation")]
pub struct LiquidationCoordinator {
    /// Configuration
    #[expect(dead_code, reason = "Field will be used in future implementations")]

    /// Configuration
    config: LiquidationConfig,
    /// Health factor monitor
    health_monitor: Arc<RwLock<HealthMonitor>>,
    /// Protocol-specific liquidators
    liquidators: AHashMap<LiquidationProtocol, Box<dyn ProtocolLiquidator + Send + Sync>>,
    /// Profit calculator
    profit_calculator: Arc<ProfitCalculator>,
    /// Multicall optimizer
    multicall_optimizer: Arc<MulticallOptimizer>,
    /// Active liquidations tracking
    active_liquidations: Arc<RwLock<AHashMap<String, LiquidationOpportunity>>>,
    /// Liquidation statistics
    stats: LiquidationStats,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Opportunity sender channel
    opportunity_sender: mpsc::UnboundedSender<LiquidationOpportunity>,
    /// Result receiver channel
    result_receiver: Arc<RwLock<mpsc::UnboundedReceiver<LiquidationResult>>>,
}

/// Liquidation statistics
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct LiquidationStats {
    /// Total opportunities detected
    pub opportunities_detected: AtomicU64,
    /// Total liquidations executed
    pub liquidations_executed: AtomicU64,
    /// Total profit realized in USD
    pub total_profit_usd: AtomicU64, // Stored as cents to avoid float
    /// Total gas spent in wei
    pub total_gas_spent: AtomicU64,
    /// Average execution time in microseconds
    pub avg_execution_time_us: AtomicU64,
    /// Success rate (percentage * 100)
    pub success_rate: AtomicU64,
}

/// Protocol liquidator trait
#[async_trait::async_trait]
pub trait ProtocolLiquidator {
    /// Execute liquidation for the protocol
    async fn execute_liquidation(&self, opportunity: &LiquidationOpportunity)
        -> StrategyResult<LiquidationResult>;

    /// Estimate gas cost for liquidation
    async fn estimate_gas(&self, opportunity: &LiquidationOpportunity)
        -> StrategyResult<u64>;

    /// Check if liquidation is still valid
    async fn validate_liquidation(&self, opportunity: &LiquidationOpportunity)
        -> StrategyResult<bool>;

    /// Get protocol name
    fn protocol_name(&self) -> &'static str;
}

impl LiquidationCoordinator {
    /// Create new liquidation coordinator
    ///
    /// # Errors
    ///
    /// Returns error if coordinator initialization fails
    /// # Errors
    ///
    /// Returns error if operation fails
    #[inline]
    pub fn new(config: LiquidationConfig) -> StrategyResult<Self> {
        let (opportunity_sender, _opportunity_receiver) = mpsc::unbounded_channel();
        let (_result_sender, result_receiver) = mpsc::unbounded_channel();
        
        let health_monitor = Arc::new(RwLock::new(HealthMonitor::new()));
        let profit_calculator = Arc::new(ProfitCalculator::new());
        let multicall_optimizer = Arc::new(MulticallOptimizer::new());
        
        Ok(Self {
            config,
            health_monitor,
            liquidators: AHashMap::new(),
            profit_calculator,
            multicall_optimizer,
            active_liquidations: Arc::new(RwLock::new(AHashMap::new())),
            stats: LiquidationStats::default(),
            shutdown: Arc::new(AtomicBool::new(false)),
            opportunity_sender,
            result_receiver: Arc::new(RwLock::new(result_receiver)),
        })
    }
    
    /// Register protocol liquidator
    #[inline]
    pub fn register_liquidator<T>(&mut self, protocol: LiquidationProtocol, liquidator: T)
    where 
        T: ProtocolLiquidator + Send + Sync + 'static 
    {
        self.liquidators.insert(protocol, Box::new(liquidator));
        tracing::info!("Registered liquidator for protocol: {:?}", protocol);
    }
    
    /// Start liquidation monitoring and execution
    ///
    /// # Errors
    ///
    /// Returns error if monitoring or execution setup fails    /// Returns error if operation fails
    /// # Errors
    ///
    /// Returns error if operation fails
    #[inline]
    pub fn start(&self) -> StrategyResult<()> {
        tracing::info!("Starting liquidation coordinator");
        
        // Start health monitoring
        self.start_health_monitoring();

        // Start liquidation execution loop
        self.start_execution_loop();
        
        tracing::info!("Liquidation coordinator started successfully");

        Ok(())
    }
    
    /// Stop liquidation coordinator
    ///
    /// # Errors
    ///
    /// Returns error if shutdown process fails    /// Returns error if operation fails
    #[inline]
    pub async fn stop(&self) -> StrategyResult<()> {
        tracing::info!("Stopping liquidation coordinator");
        
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Wait for active liquidations to complete
        let mut retries = 0_u32;
        while !self.active_liquidations.read().await.is_empty() && retries < 100 {
            sleep(Duration::from_millis(100)).await;
            retries = retries.saturating_add(1);
        
        }

        tracing::info!("Liquidation coordinator stopped");

        Ok(())
    }
    
    /// Get liquidation statistics
    #[must_use]
    #[inline]
    pub fn get_stats(&self) -> LiquidationStats {
        LiquidationStats {
            opportunities_detected: AtomicU64::new(
                self.stats.opportunities_detected.load(Ordering::Relaxed)
            ),
            liquidations_executed: AtomicU64::new(
                self.stats.liquidations_executed.load(Ordering::Relaxed)
            ),
            total_profit_usd: AtomicU64::new(
                self.stats.total_profit_usd.load(Ordering::Relaxed)
            ),
            total_gas_spent: AtomicU64::new(
                self.stats.total_gas_spent.load(Ordering::Relaxed)
            ),
            avg_execution_time_us: AtomicU64::new(
                self.stats.avg_execution_time_us.load(Ordering::Relaxed)
            ),
            success_rate: AtomicU64::new(
                self.stats.success_rate.load(Ordering::Relaxed)
            ),
        }
    }
    
    /// Start health monitoring background task
    #[expect(clippy::unused_self, reason = "Will be used in full implementation")]
    #[expect(clippy::missing_const_for_fn, reason = "Will use self in full implementation")]
    fn start_health_monitoring(&self) {
        // Implementation will be in health_monitor.rs
    }

    /// Start liquidation execution loop
    #[expect(clippy::unused_self, reason = "Will be used in full implementation")]
    #[expect(clippy::missing_const_for_fn, reason = "Will use self in full implementation")]
    fn start_execution_loop(&self) {
        // Implementation will handle opportunity processing
    }
}

// Placeholder structs - will be implemented in separate modules
/// Health factor monitoring system (placeholder)
#[non_exhaustive]
pub struct HealthMonitor;

/// Profit calculation engine (placeholder)
#[non_exhaustive]
pub struct ProfitCalculator;

/// Multicall optimization system (placeholder)
#[non_exhaustive]
pub struct MulticallOptimizer;

impl HealthMonitor {
    /// Create new health monitor
    const fn new() -> Self {
        Self
    }
}

impl ProfitCalculator {
    /// Create new profit calculator
    const fn new() -> Self {
        Self
    }
}

impl MulticallOptimizer {
    /// Create new multicall optimizer
    const fn new() -> Self {
        Self
    }
}

// Re-export submodules (will be implemented)
pub mod health_monitor;
pub mod aave_liquidator;
pub mod venus_liquidator;
pub mod compound_liquidator;
pub mod refinance_liquidator;
pub mod multicall_optimizer;
pub mod profit_calculator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidation_config_default() {
        let config = LiquidationConfig::default();
        assert_eq!(config.min_health_factor, Decimal::new(100, 2));
        assert_eq!(config.min_profit_threshold, Decimal::new(5, 2));
        assert_eq!(config.max_gas_price, 100);
        assert_eq!(config.max_concurrent_liquidations, 50);
        assert_eq!(config.health_monitor_interval_ms, 100);
        assert!(config.enable_multicall);
        assert!(config.enable_refinancing);
    }

    #[test]
    fn test_liquidation_protocol_values() {
        assert_eq!(LiquidationProtocol::AaveV3 as u8, 0);
        assert_eq!(LiquidationProtocol::Venus as u8, 1);
        assert_eq!(LiquidationProtocol::CompoundV3 as u8, 2);
        assert_eq!(LiquidationProtocol::Refinance as u8, 3);
    }

    #[test]
    fn test_liquidation_status_equality() {
        assert_eq!(LiquidationStatus::Pending, LiquidationStatus::Pending);
        assert_ne!(LiquidationStatus::Pending, LiquidationStatus::Executing);
        assert_ne!(LiquidationStatus::Completed, LiquidationStatus::Failed);
    }

    #[test]
    fn test_liquidation_opportunity_creation() {
        let opportunity = LiquidationOpportunity {
            position_id: "test_position_123".to_owned(),
            protocol: LiquidationProtocol::AaveV3,
            chain_id: 1,
            user_address: [1_u8; 20],
            collateral_token: [2_u8; 20],
            debt_token: [3_u8; 20],
            health_factor: Decimal::new(95, 2), // 0.95
            liquidatable_debt: Decimal::new(1000, 0), // $1000
            collateral_amount: Decimal::new(1050, 0), // $1050
            estimated_profit: 5000, // $50.00 in cents
            gas_cost: Decimal::new(10, 0), // $10
            net_profit: 4000, // $40.00 in cents
            liquidation_bonus: Decimal::new(5, 2), // 5%
            urgency: 200,
            block_number: 18_500_000,
            timestamp: 1_700_000_000,
        };

        assert_eq!(opportunity.position_id, "test_position_123");
        assert_eq!(opportunity.protocol, LiquidationProtocol::AaveV3);
        assert_eq!(opportunity.chain_id, 1);
        assert_eq!(opportunity.health_factor, Decimal::new(95, 2));
        assert_eq!(opportunity.net_profit, 4000);
        assert_eq!(opportunity.urgency, 200);
    }

    #[tokio::test]
    async fn test_liquidation_coordinator_creation() {
        let config = LiquidationConfig::default();
        let coordinator = LiquidationCoordinator::new(config);
        assert!(coordinator.is_ok());

        if let Ok(coordinator) = coordinator {
            assert_eq!(coordinator.liquidators.len(), 0);
            assert!(!coordinator.shutdown.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn test_liquidation_stats_default() {
        let stats = LiquidationStats::default();
        assert_eq!(stats.opportunities_detected.load(Ordering::Relaxed), 0);
        assert_eq!(stats.liquidations_executed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_profit_usd.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_gas_spent.load(Ordering::Relaxed), 0);
        assert_eq!(stats.avg_execution_time_us.load(Ordering::Relaxed), 0);
        assert_eq!(stats.success_rate.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_liquidation_error_types() {
        let health_error = LiquidationError::HealthMonitor {
            message: "Health monitoring failed".to_owned(),
        };
        assert!(health_error.to_string().contains("Health monitor error"));

        let protocol_error = LiquidationError::ProtocolLiquidation {
            protocol: "Aave".to_owned(),
            message: "Liquidation failed".to_owned(),
        };
        assert!(protocol_error.to_string().contains("Protocol liquidation error"));

        let profit_error = LiquidationError::InsufficientProfit {
            required: Decimal::new(100, 0),
            available: Decimal::new(50, 0),
        };
        assert!(profit_error.to_string().contains("Insufficient profit"));
    }

    #[test]
    fn test_liquidation_result_creation() {
        let opportunity = LiquidationOpportunity {
            position_id: "test_position_456".to_owned(),
            protocol: LiquidationProtocol::Venus,
            chain_id: 56,
            user_address: [4_u8; 20],
            collateral_token: [5_u8; 20],
            debt_token: [6_u8; 20],
            health_factor: Decimal::new(90, 2),
            liquidatable_debt: Decimal::new(500, 0),
            collateral_amount: Decimal::new(525, 0),
            estimated_profit: 2500, // $25.00 in cents
            gas_cost: Decimal::new(5, 0),
            net_profit: 2000, // $20.00 in cents
            liquidation_bonus: Decimal::new(5, 2),
            urgency: 150,
            block_number: 32_000_000,
            timestamp: 1_700_001_000,
        };

        let result = LiquidationResult {
            opportunity,
            status: LiquidationStatus::Completed,
            transaction_hash: Some([7_u8; 32]),
            actual_profit: Some(2200), // $22.00 in cents
            gas_used: Some(150_000),
            execution_time_us: 2_500,
            error_message: None,
        };

        assert_eq!(result.status, LiquidationStatus::Completed);
        assert!(result.transaction_hash.is_some());
        assert_eq!(result.actual_profit, Some(2200));
        assert_eq!(result.gas_used, Some(150_000));
        assert_eq!(result.execution_time_us, 2_500);
        assert!(result.error_message.is_none());
    }

    #[tokio::test]
    async fn test_liquidation_coordinator_start_stop() {
        let config = LiquidationConfig::default();
        let coordinator = LiquidationCoordinator::new(config);
        assert!(coordinator.is_ok());
        assert!(coordinator.is_ok());

        if let Ok(coordinator) = coordinator {
            // Test start
            let start_result = coordinator.start();
            assert!(start_result.is_ok());

            // Test stop
            let stop_result = coordinator.stop().await;
            assert!(stop_result.is_ok());
            assert!(coordinator.shutdown.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn test_liquidation_opportunity_profitability() {
        let profitable_opportunity = LiquidationOpportunity {
            position_id: "profitable_123".to_owned(),
            protocol: LiquidationProtocol::CompoundV3,
            chain_id: 137,
            user_address: [8_u8; 20],
            collateral_token: [9_u8; 20],
            debt_token: [10_u8; 20],
            health_factor: Decimal::new(85, 2), // 0.85 - liquidatable
            liquidatable_debt: Decimal::new(2000, 0),
            collateral_amount: Decimal::new(2100, 0),
            estimated_profit: 10000, // $100.00 in cents
            gas_cost: Decimal::new(20, 0),
            net_profit: 8000, // $80.00 in cents - Profitable
            liquidation_bonus: Decimal::new(5, 2),
            urgency: 255, // Maximum urgency
            block_number: 48_000_000,
            timestamp: 1_700_002_000,
        };

        // Verify profitability
        assert!(profitable_opportunity.net_profit > 0);
        assert!(Decimal::from(profitable_opportunity.estimated_profit) > profitable_opportunity.gas_cost);
        assert!(profitable_opportunity.health_factor < Decimal::ONE);
        assert_eq!(profitable_opportunity.urgency, 255);
    }
}
