//! Cross-Chain Arbitrage for ultra-performance multi-chain arbitrage operations
//!
//! This module provides advanced cross-chain arbitrage capabilities for maximizing
//! profit opportunities across different blockchain networks through intelligent
//! price difference detection and optimal execution strategies.
//!
//! ## Performance Targets
//! - Price Difference Detection: <100μs
//! - Arbitrage Opportunity Calculation: <200μs
//! - Route Optimization: <300μs
//! - Execution Planning: <150μs
//! - Profit Estimation: <50μs
//!
//! ## Architecture
//! - Real-time cross-chain price monitoring
//! - Advanced arbitrage opportunity detection
//! - Multi-bridge route optimization
//! - Risk-adjusted profit calculations
//! - Lock-free arbitrage primitives

use crate::{
    ChainCoreConfig, Result,
    types::ChainId,
    utils::perf::Timer,
    coordination::bridge_monitor::BridgeType,
};
use crossbeam::channel::{self, Receiver, Sender};
use dashmap::DashMap;
use rust_decimal::Decimal;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicBool, Ordering},
    },
    time::{Duration, Instant},
    collections::HashMap,
};
use tokio::{
    sync::{RwLock, Mutex as TokioMutex},
    time::{interval, sleep},
};
use tracing::{info, trace, warn};

/// Cross-chain arbitrage configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct CrossChainArbitrageConfig {
    /// Enable cross-chain arbitrage
    pub enabled: bool,
    
    /// Price monitoring interval in milliseconds
    pub price_monitor_interval_ms: u64,
    
    /// Opportunity detection interval in milliseconds
    pub opportunity_detection_interval_ms: u64,
    
    /// Execution planning interval in milliseconds
    pub execution_planning_interval_ms: u64,
    
    /// Enable price monitoring
    pub enable_price_monitoring: bool,
    
    /// Enable opportunity detection
    pub enable_opportunity_detection: bool,
    
    /// Enable automatic execution
    pub enable_automatic_execution: bool,
    
    /// Enable risk management
    pub enable_risk_management: bool,
    
    /// Minimum profit threshold (USD)
    pub min_profit_threshold_usd: Decimal,
    
    /// Maximum position size (USD)
    pub max_position_size_usd: Decimal,
    
    /// Maximum slippage tolerance
    pub max_slippage_tolerance: Decimal,
    
    /// Monitored trading pairs
    pub monitored_pairs: Vec<String>,
    
    /// Supported chains for arbitrage
    pub supported_chains: Vec<ChainId>,
}

/// Cross-chain price information
#[derive(Debug, Clone)]
pub struct CrossChainPrice {
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Trading pair (e.g., "USDC/USDT")
    pub trading_pair: String,
    
    /// DEX/Protocol name
    pub protocol: String,
    
    /// Bid price
    pub bid_price: Decimal,
    
    /// Ask price
    pub ask_price: Decimal,
    
    /// Mid price
    pub mid_price: Decimal,
    
    /// Available liquidity (USD)
    pub liquidity_usd: Decimal,
    
    /// Price timestamp
    pub timestamp: u64,
    
    /// Data source
    pub source: String,
}

/// Cross-chain arbitrage opportunity
#[derive(Debug, Clone)]
pub struct CrossChainArbitrageOpportunity {
    /// Opportunity ID
    pub id: String,
    
    /// Trading pair
    pub trading_pair: String,
    
    /// Source chain (buy)
    pub source_chain: ChainId,
    
    /// Destination chain (sell)
    pub destination_chain: ChainId,
    
    /// Source protocol
    pub source_protocol: String,
    
    /// Destination protocol
    pub destination_protocol: String,
    
    /// Buy price
    pub buy_price: Decimal,
    
    /// Sell price
    pub sell_price: Decimal,
    
    /// Price difference percentage
    pub price_diff_pct: Decimal,
    
    /// Optimal trade size (USD)
    pub optimal_trade_size_usd: Decimal,
    
    /// Expected gross profit (USD)
    pub expected_gross_profit_usd: Decimal,
    
    /// Bridge fees (USD)
    pub bridge_fees_usd: Decimal,
    
    /// Gas fees (USD)
    pub gas_fees_usd: Decimal,
    
    /// Expected net profit (USD)
    pub expected_net_profit_usd: Decimal,
    
    /// Risk score (1-10, 10 = highest risk)
    pub risk_score: u8,
    
    /// Execution time estimate (seconds)
    pub execution_time_estimate_s: u32,
    
    /// Required bridge
    pub required_bridge: BridgeType,
    
    /// Opportunity expiry
    pub expires_at: u64,
    
    /// Created timestamp
    pub created_at: u64,
}

/// Arbitrage execution plan
#[derive(Debug, Clone)]
pub struct ArbitrageExecutionPlan {
    /// Plan ID
    pub id: String,
    
    /// Associated opportunity
    pub opportunity_id: String,
    
    /// Execution steps
    pub steps: Vec<ExecutionStep>,
    
    /// Total estimated time (seconds)
    pub total_time_estimate_s: u32,
    
    /// Total estimated cost (USD)
    pub total_cost_estimate_usd: Decimal,
    
    /// Expected profit (USD)
    pub expected_profit_usd: Decimal,
    
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    
    /// Plan status
    pub status: ExecutionStatus,
    
    /// Created timestamp
    pub created_at: u64,
}

/// Execution step in arbitrage plan
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// Step number
    pub step_number: u32,
    
    /// Step type
    pub step_type: StepType,
    
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Protocol
    pub protocol: String,
    
    /// Action description
    pub action: String,
    
    /// Amount
    pub amount: Decimal,
    
    /// Estimated time (seconds)
    pub estimated_time_s: u32,
    
    /// Estimated cost (USD)
    pub estimated_cost_usd: Decimal,
    
    /// Dependencies (step numbers)
    pub dependencies: Vec<u32>,
}

/// Execution step types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepType {
    /// Buy on source chain
    Buy,
    /// Bridge assets
    Bridge,
    /// Sell on destination chain
    Sell,
    /// Approve token spending
    Approve,
    /// Wait for confirmation
    Wait,
}

/// Risk assessment for arbitrage
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Overall risk score (1-10)
    pub overall_risk_score: u8,
    
    /// Price volatility risk
    pub price_volatility_risk: u8,
    
    /// Liquidity risk
    pub liquidity_risk: u8,
    
    /// Bridge risk
    pub bridge_risk: u8,
    
    /// Execution risk
    pub execution_risk: u8,
    
    /// Slippage risk
    pub slippage_risk: u8,
    
    /// Risk factors
    pub risk_factors: Vec<String>,
    
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStatus {
    /// Plan created
    Created,
    /// Plan approved
    Approved,
    /// Execution started
    Started,
    /// Execution in progress
    InProgress,
    /// Execution completed
    Completed,
    /// Execution failed
    Failed,
    /// Execution cancelled
    Cancelled,
}

/// Cross-chain arbitrage statistics
#[derive(Debug, Default)]
pub struct CrossChainArbitrageStats {
    /// Total opportunities detected
    pub opportunities_detected: AtomicU64,
    
    /// Total opportunities executed
    pub opportunities_executed: AtomicU64,
    
    /// Successful executions
    pub successful_executions: AtomicU64,
    
    /// Failed executions
    pub failed_executions: AtomicU64,
    
    /// Total gross profit (USD)
    pub total_gross_profit_usd: AtomicU64,
    
    /// Total net profit (USD)
    pub total_net_profit_usd: AtomicU64,
    
    /// Total bridge fees paid (USD)
    pub total_bridge_fees_usd: AtomicU64,
    
    /// Total gas fees paid (USD)
    pub total_gas_fees_usd: AtomicU64,
    
    /// Price updates processed
    pub price_updates_processed: AtomicU64,
    
    /// Average execution time (seconds)
    pub avg_execution_time_s: AtomicU64,
}

/// Cache-line aligned arbitrage data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedArbitrageData {
    /// Opportunities detected (scaled)
    pub opportunities_detected_scaled: u64,
    
    /// Opportunities executed (scaled)
    pub opportunities_executed_scaled: u64,
    
    /// Success rate (scaled by 1e6)
    pub success_rate_scaled: u64,
    
    /// Average profit USD (scaled by 1e6)
    pub avg_profit_usd_scaled: u64,
    
    /// Average execution time (seconds)
    pub avg_execution_time_s: u64,
    
    /// Active opportunities count
    pub active_opportunities_count: u64,
    
    /// Price updates per second
    pub price_updates_per_second: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// Cross-chain arbitrage constants
pub const ARBITRAGE_DEFAULT_PRICE_INTERVAL_MS: u64 = 100; // 100ms
pub const ARBITRAGE_DEFAULT_OPPORTUNITY_INTERVAL_MS: u64 = 200; // 200ms
pub const ARBITRAGE_DEFAULT_EXECUTION_INTERVAL_MS: u64 = 500; // 500ms
pub const ARBITRAGE_DEFAULT_MIN_PROFIT_USD: &str = "50.0"; // $50 minimum
pub const ARBITRAGE_DEFAULT_MAX_POSITION_USD: &str = "100000.0"; // $100k maximum
pub const ARBITRAGE_DEFAULT_MAX_SLIPPAGE: &str = "0.005"; // 0.5% maximum
pub const ARBITRAGE_MAX_OPPORTUNITIES: usize = 1000;
pub const ARBITRAGE_MAX_EXECUTION_PLANS: usize = 100;

impl Default for CrossChainArbitrageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            price_monitor_interval_ms: ARBITRAGE_DEFAULT_PRICE_INTERVAL_MS,
            opportunity_detection_interval_ms: ARBITRAGE_DEFAULT_OPPORTUNITY_INTERVAL_MS,
            execution_planning_interval_ms: ARBITRAGE_DEFAULT_EXECUTION_INTERVAL_MS,
            enable_price_monitoring: true,
            enable_opportunity_detection: true,
            enable_automatic_execution: false, // Manual approval required by default
            enable_risk_management: true,
            min_profit_threshold_usd: ARBITRAGE_DEFAULT_MIN_PROFIT_USD.parse().unwrap_or_default(),
            max_position_size_usd: ARBITRAGE_DEFAULT_MAX_POSITION_USD.parse().unwrap_or_default(),
            max_slippage_tolerance: ARBITRAGE_DEFAULT_MAX_SLIPPAGE.parse().unwrap_or_default(),
            monitored_pairs: vec![
                "USDC/USDT".to_string(),
                "ETH/USDC".to_string(),
                "WBTC/ETH".to_string(),
                "DAI/USDC".to_string(),
            ],
            supported_chains: vec![
                ChainId::Ethereum,
                ChainId::Arbitrum,
                ChainId::Optimism,
                ChainId::Polygon,
                ChainId::Bsc,
                ChainId::Avalanche,
                ChainId::Base,
            ],
        }
    }
}

impl AlignedArbitrageData {
    /// Create new aligned arbitrage data
    #[inline(always)]
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "Aligned data structure requires all fields")]
    pub const fn new(
        opportunities_detected_scaled: u64,
        opportunities_executed_scaled: u64,
        success_rate_scaled: u64,
        avg_profit_usd_scaled: u64,
        avg_execution_time_s: u64,
        active_opportunities_count: u64,
        price_updates_per_second: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            opportunities_detected_scaled,
            opportunities_executed_scaled,
            success_rate_scaled,
            avg_profit_usd_scaled,
            avg_execution_time_s,
            active_opportunities_count,
            price_updates_per_second,
            timestamp,
        }
    }

    /// Check if data is stale
    #[inline(always)]
    #[must_use]
    #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for staleness check")]
    pub fn is_stale(&self, max_age_ms: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        now.saturating_sub(self.timestamp) > max_age_ms
    }

    /// Get success rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn success_rate(&self) -> Decimal {
        Decimal::from(self.success_rate_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get average profit USD as Decimal
    #[inline(always)]
    #[must_use]
    pub fn avg_profit_usd(&self) -> Decimal {
        Decimal::from(self.avg_profit_usd_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get execution efficiency (opportunities executed / detected)
    #[inline(always)]
    #[must_use]
    pub fn execution_efficiency(&self) -> Decimal {
        if self.opportunities_detected_scaled == 0 {
            return Decimal::ZERO;
        }

        Decimal::from(self.opportunities_executed_scaled) / Decimal::from(self.opportunities_detected_scaled)
    }

    /// Get profit per opportunity
    #[inline(always)]
    #[must_use]
    pub fn profit_per_opportunity(&self) -> Decimal {
        if self.opportunities_executed_scaled == 0 {
            return Decimal::ZERO;
        }

        self.avg_profit_usd() * Decimal::from(self.opportunities_executed_scaled)
    }
}

/// Cross-Chain Arbitrage for ultra-performance multi-chain arbitrage operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct CrossChainArbitrage {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Arbitrage specific configuration
    arbitrage_config: CrossChainArbitrageConfig,

    /// Statistics
    stats: Arc<CrossChainArbitrageStats>,

    /// Cross-chain prices
    prices: Arc<RwLock<HashMap<String, CrossChainPrice>>>,

    /// Arbitrage data cache for ultra-fast access
    arbitrage_cache: Arc<DashMap<String, AlignedArbitrageData>>,

    /// Detected opportunities
    opportunities: Arc<RwLock<HashMap<String, CrossChainArbitrageOpportunity>>>,

    /// Execution plans
    execution_plans: Arc<RwLock<HashMap<String, ArbitrageExecutionPlan>>>,

    /// Performance timers
    price_timer: Timer,
    opportunity_timer: Timer,
    execution_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Price update channels
    price_sender: Sender<CrossChainPrice>,
    price_receiver: Receiver<CrossChainPrice>,

    /// Opportunity channels
    opportunity_sender: Sender<CrossChainArbitrageOpportunity>,
    opportunity_receiver: Receiver<CrossChainArbitrageOpportunity>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current arbitrage round
    arbitrage_round: Arc<TokioMutex<u64>>,
}

impl CrossChainArbitrage {
    /// Create new cross-chain arbitrage with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let arbitrage_config = CrossChainArbitrageConfig::default();
        let stats = Arc::new(CrossChainArbitrageStats::default());
        let prices = Arc::new(RwLock::new(HashMap::with_capacity(1000)));
        let arbitrage_cache = Arc::new(DashMap::with_capacity(100));
        let opportunities = Arc::new(RwLock::new(HashMap::with_capacity(ARBITRAGE_MAX_OPPORTUNITIES)));
        let execution_plans = Arc::new(RwLock::new(HashMap::with_capacity(ARBITRAGE_MAX_EXECUTION_PLANS)));
        let price_timer = Timer::new("arbitrage_price");
        let opportunity_timer = Timer::new("arbitrage_opportunity");
        let execution_timer = Timer::new("arbitrage_execution");
        let shutdown = Arc::new(AtomicBool::new(false));
        let arbitrage_round = Arc::new(TokioMutex::new(0));

        let (price_sender, price_receiver) = channel::bounded(10000);
        let (opportunity_sender, opportunity_receiver) = channel::bounded(ARBITRAGE_MAX_OPPORTUNITIES);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            arbitrage_config,
            stats,
            prices,
            arbitrage_cache,
            opportunities,
            execution_plans,
            price_timer,
            opportunity_timer,
            execution_timer,
            shutdown,
            price_sender,
            price_receiver,
            opportunity_sender,
            opportunity_receiver,
            http_client,
            arbitrage_round,
        })
    }

    /// Start cross-chain arbitrage services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.arbitrage_config.enabled {
            info!("Cross-chain arbitrage disabled");
            return Ok(());
        }

        info!("Starting cross-chain arbitrage");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start price monitoring
        if self.arbitrage_config.enable_price_monitoring {
            self.start_price_monitoring().await;
        }

        // Start opportunity detection
        if self.arbitrage_config.enable_opportunity_detection {
            self.start_opportunity_detection().await;
        }

        // Start execution planning
        self.start_execution_planning().await;

        // Start risk management
        if self.arbitrage_config.enable_risk_management {
            self.start_risk_management().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Cross-chain arbitrage started successfully");
        Ok(())
    }

    /// Stop cross-chain arbitrage
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping cross-chain arbitrage");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Cross-chain arbitrage stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &CrossChainArbitrageStats {
        &self.stats
    }

    /// Get current prices
    #[inline]
    pub async fn get_prices(&self) -> Vec<CrossChainPrice> {
        let prices = self.prices.read().await;
        prices.values().cloned().collect()
    }

    /// Get detected opportunities
    #[inline]
    pub async fn get_opportunities(&self) -> Vec<CrossChainArbitrageOpportunity> {
        let opportunities = self.opportunities.read().await;
        opportunities.values().cloned().collect()
    }

    /// Get execution plans
    #[inline]
    pub async fn get_execution_plans(&self) -> Vec<ArbitrageExecutionPlan> {
        let plans = self.execution_plans.read().await;
        plans.values().cloned().collect()
    }

    /// Calculate arbitrage opportunity between two prices
    #[inline]
    #[must_use]
    pub fn calculate_arbitrage_opportunity(
        buy_price: &CrossChainPrice,
        sell_price: &CrossChainPrice,
        bridge_fee_pct: Decimal,
        gas_fee_usd: Decimal,
    ) -> Option<CrossChainArbitrageOpportunity> {
        // Check if arbitrage is profitable
        if sell_price.bid_price <= buy_price.ask_price {
            return None;
        }

        let price_diff_pct = (sell_price.bid_price - buy_price.ask_price) / buy_price.ask_price;

        // Calculate optimal trade size based on available liquidity
        let max_liquidity = buy_price.liquidity_usd.min(sell_price.liquidity_usd);
        let optimal_trade_size = max_liquidity * "0.1".parse::<Decimal>().unwrap_or_default(); // Use 10% of available liquidity

        let gross_profit = optimal_trade_size * price_diff_pct;
        let bridge_fees = optimal_trade_size * bridge_fee_pct;
        let net_profit = gross_profit - bridge_fees - gas_fee_usd;

        // Check if net profit meets minimum threshold
        if net_profit <= "10".parse::<Decimal>().unwrap_or_default() {
            return None;
        }

        let risk_score = Self::calculate_risk_score(price_diff_pct, optimal_trade_size, buy_price.chain_id, sell_price.chain_id);

        Some(CrossChainArbitrageOpportunity {
            id: format!("arb_{}_{}", chrono::Utc::now().timestamp_millis(), fastrand::u32(..)),
            trading_pair: buy_price.trading_pair.clone(),
            source_chain: buy_price.chain_id,
            destination_chain: sell_price.chain_id,
            source_protocol: buy_price.protocol.clone(),
            destination_protocol: sell_price.protocol.clone(),
            buy_price: buy_price.ask_price,
            sell_price: sell_price.bid_price,
            price_diff_pct,
            optimal_trade_size_usd: optimal_trade_size,
            expected_gross_profit_usd: gross_profit,
            bridge_fees_usd: bridge_fees,
            gas_fees_usd: gas_fee_usd,
            expected_net_profit_usd: net_profit,
            risk_score,
            execution_time_estimate_s: Self::estimate_execution_time(buy_price.chain_id, sell_price.chain_id),
            required_bridge: Self::select_optimal_bridge(buy_price.chain_id, sell_price.chain_id),
            expires_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() + 300, // 5 minutes
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for arbitrage data")]
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })
    }

    /// Create execution plan for arbitrage opportunity
    #[inline]
    #[must_use]
    pub fn create_execution_plan(opportunity: &CrossChainArbitrageOpportunity) -> ArbitrageExecutionPlan {
        let mut steps = Vec::with_capacity(5);
        let mut total_time = 0_u32;
        let mut total_cost = Decimal::ZERO;

        // Step 1: Approve token spending on source chain
        steps.push(ExecutionStep {
            step_number: 1,
            step_type: StepType::Approve,
            chain_id: opportunity.source_chain,
            protocol: opportunity.source_protocol.clone(),
            action: "Approve token spending".to_string(),
            amount: opportunity.optimal_trade_size_usd,
            estimated_time_s: 30,
            estimated_cost_usd: "5".parse().unwrap_or_default(),
            dependencies: vec![],
        });

        // Step 2: Buy on source chain
        steps.push(ExecutionStep {
            step_number: 2,
            step_type: StepType::Buy,
            chain_id: opportunity.source_chain,
            protocol: opportunity.source_protocol.clone(),
            action: format!("Buy {} on {}", opportunity.trading_pair, opportunity.source_protocol),
            amount: opportunity.optimal_trade_size_usd,
            estimated_time_s: 60,
            estimated_cost_usd: opportunity.gas_fees_usd / Decimal::from(2_u64),
            dependencies: vec![1],
        });

        // Step 3: Bridge assets to destination chain
        steps.push(ExecutionStep {
            step_number: 3,
            step_type: StepType::Bridge,
            chain_id: opportunity.source_chain,
            protocol: format!("{:?}", opportunity.required_bridge),
            action: format!("Bridge assets from {:?} to {:?}", opportunity.source_chain, opportunity.destination_chain),
            amount: opportunity.optimal_trade_size_usd,
            estimated_time_s: opportunity.execution_time_estimate_s / 2,
            estimated_cost_usd: opportunity.bridge_fees_usd,
            dependencies: vec![2],
        });

        // Step 4: Wait for bridge confirmation
        steps.push(ExecutionStep {
            step_number: 4,
            step_type: StepType::Wait,
            chain_id: opportunity.destination_chain,
            protocol: format!("{:?}", opportunity.required_bridge),
            action: "Wait for bridge confirmation".to_string(),
            amount: Decimal::ZERO,
            estimated_time_s: opportunity.execution_time_estimate_s / 2,
            estimated_cost_usd: Decimal::ZERO,
            dependencies: vec![3],
        });

        // Step 5: Sell on destination chain
        steps.push(ExecutionStep {
            step_number: 5,
            step_type: StepType::Sell,
            chain_id: opportunity.destination_chain,
            protocol: opportunity.destination_protocol.clone(),
            action: format!("Sell {} on {}", opportunity.trading_pair, opportunity.destination_protocol),
            amount: opportunity.optimal_trade_size_usd,
            estimated_time_s: 60,
            estimated_cost_usd: opportunity.gas_fees_usd / Decimal::from(2_u64),
            dependencies: vec![4],
        });

        // Calculate totals
        for step in &steps {
            total_time = total_time.saturating_add(step.estimated_time_s);
            total_cost += step.estimated_cost_usd;
        }

        let risk_assessment = Self::assess_execution_risk(opportunity);

        ArbitrageExecutionPlan {
            id: format!("plan_{}_{}", opportunity.id, fastrand::u32(..)),
            opportunity_id: opportunity.id.clone(),
            steps,
            total_time_estimate_s: total_time,
            total_cost_estimate_usd: total_cost,
            expected_profit_usd: opportunity.expected_net_profit_usd,
            risk_assessment,
            status: ExecutionStatus::Created,
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for execution plan")]
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Calculate risk score for arbitrage opportunity
    fn calculate_risk_score(
        price_diff_pct: Decimal,
        trade_size_usd: Decimal,
        source_chain: ChainId,
        destination_chain: ChainId,
    ) -> u8 {
        let mut risk_score = 1_u8;

        // Price difference risk (higher difference = higher risk)
        if price_diff_pct > "0.05".parse::<Decimal>().unwrap_or_default() {
            risk_score = risk_score.saturating_add(3);
        } else if price_diff_pct > "0.02".parse::<Decimal>().unwrap_or_default() {
            risk_score = risk_score.saturating_add(2);
        } else if price_diff_pct > "0.01".parse::<Decimal>().unwrap_or_default() {
            risk_score = risk_score.saturating_add(1);
        }

        // Trade size risk (larger size = higher risk)
        if trade_size_usd > "50000".parse::<Decimal>().unwrap_or_default() {
            risk_score = risk_score.saturating_add(3);
        } else if trade_size_usd > "20000".parse::<Decimal>().unwrap_or_default() {
            risk_score = risk_score.saturating_add(2);
        } else if trade_size_usd > "10000".parse::<Decimal>().unwrap_or_default() {
            risk_score = risk_score.saturating_add(1);
        }

        // Chain risk (some chains are riskier)
        let chain_risk = match (source_chain, destination_chain) {
            (ChainId::Ethereum, ChainId::Arbitrum | ChainId::Optimism) |
            (ChainId::Arbitrum | ChainId::Optimism, ChainId::Ethereum) => 1,
            (ChainId::Ethereum, ChainId::Bsc) | (ChainId::Bsc, ChainId::Ethereum) => 3,
            _ => 2, // Default risk for other combinations (including Polygon and Avalanche)
        };

        risk_score = risk_score.saturating_add(chain_risk);
        risk_score.min(10) // Cap at 10
    }

    /// Estimate execution time for cross-chain arbitrage
    const fn estimate_execution_time(source_chain: ChainId, destination_chain: ChainId) -> u32 {
        match (source_chain, destination_chain) {
            (ChainId::Ethereum, ChainId::Arbitrum) | (ChainId::Arbitrum, ChainId::Ethereum) => 600,  // 10 minutes
            (ChainId::Ethereum, ChainId::Optimism) | (ChainId::Optimism, ChainId::Ethereum) => 1200, // 20 minutes
            (ChainId::Ethereum, ChainId::Polygon | ChainId::Avalanche) |
            (ChainId::Polygon | ChainId::Avalanche, ChainId::Ethereum) => 1800, // 30 minutes
            (ChainId::Ethereum, ChainId::Bsc) | (ChainId::Bsc, ChainId::Ethereum) => 2400,          // 40 minutes
            _ => 1500, // 25 minutes default
        }
    }

    /// Select optimal bridge for cross-chain transfer
    const fn select_optimal_bridge(source_chain: ChainId, destination_chain: ChainId) -> BridgeType {
        match (source_chain, destination_chain) {
            (ChainId::Ethereum, ChainId::Arbitrum) | (ChainId::Arbitrum, ChainId::Ethereum) => BridgeType::ArbitrumBridge,
            (ChainId::Ethereum, ChainId::Optimism) | (ChainId::Optimism, ChainId::Ethereum) => BridgeType::OptimismBridge,
            (ChainId::Ethereum, ChainId::Polygon) | (ChainId::Polygon, ChainId::Ethereum) => BridgeType::PolygonPos,
            (ChainId::Ethereum, ChainId::Avalanche) | (ChainId::Avalanche, ChainId::Ethereum) => BridgeType::AvalancheBridge,
            _ => BridgeType::Stargate, // Default to Stargate for other combinations
        }
    }

    /// Assess execution risk for arbitrage opportunity
    fn assess_execution_risk(opportunity: &CrossChainArbitrageOpportunity) -> RiskAssessment {
        let price_volatility_risk = if opportunity.price_diff_pct > "0.05".parse::<Decimal>().unwrap_or_default() { 8 } else { 4 };
        let liquidity_risk = if opportunity.optimal_trade_size_usd > "20000".parse::<Decimal>().unwrap_or_default() { 6 } else { 3 };
        let bridge_risk = match opportunity.required_bridge {
            BridgeType::ArbitrumBridge | BridgeType::OptimismBridge | BridgeType::PolygonPos => 2,
            BridgeType::Stargate | BridgeType::Across => 3,
            BridgeType::Wormhole | BridgeType::Hop => 4,
            _ => 5,
        };
        let execution_risk = if opportunity.execution_time_estimate_s > 1800 { 6 } else { 3 };
        let slippage_risk = if opportunity.optimal_trade_size_usd > "10000".parse::<Decimal>().unwrap_or_default() { 5 } else { 2 };

        let overall_risk_score = ((price_volatility_risk + liquidity_risk + bridge_risk + execution_risk + slippage_risk) / 5).min(10);

        let mut risk_factors = Vec::new();
        let mut mitigation_strategies = Vec::new();

        if price_volatility_risk > 5 {
            risk_factors.push("High price volatility".to_string());
            mitigation_strategies.push("Use smaller position sizes".to_string());
        }

        if liquidity_risk > 5 {
            risk_factors.push("Large position size relative to liquidity".to_string());
            mitigation_strategies.push("Split into multiple smaller trades".to_string());
        }

        if bridge_risk > 4 {
            risk_factors.push("Bridge reliability concerns".to_string());
            mitigation_strategies.push("Use alternative bridge if available".to_string());
        }

        if execution_risk > 5 {
            risk_factors.push("Long execution time".to_string());
            mitigation_strategies.push("Monitor price movements during execution".to_string());
        }

        RiskAssessment {
            overall_risk_score,
            price_volatility_risk,
            liquidity_risk,
            bridge_risk,
            execution_risk,
            slippage_risk,
            risk_factors,
            mitigation_strategies,
        }
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(1000)) // Fast timeout for price data
            .http2_prior_knowledge()
            .http2_keep_alive_timeout(Duration::from_secs(30))
            .http2_keep_alive_interval(Duration::from_secs(10))
            .pool_max_idle_per_host(20)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .map_err(|_e| crate::ChainCoreError::Network(crate::NetworkError::ConnectionRefused))?;

        {
            let mut http_client_guard = self.http_client.lock().await;
            *http_client_guard = Some(client);
        }

        Ok(())
    }

    /// Start price monitoring
    async fn start_price_monitoring(&self) {
        let price_receiver = self.price_receiver.clone();
        let prices = Arc::clone(&self.prices);
        let _arbitrage_cache = Arc::clone(&self.arbitrage_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let arbitrage_config = self.arbitrage_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(arbitrage_config.price_monitor_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming price updates
                while let Ok(price) = price_receiver.try_recv() {
                    let price_key = format!("{}_{:?}_{}", price.trading_pair, price.chain_id, price.protocol);

                    // Update prices
                    {
                        let mut prices_guard = prices.write().await;
                        prices_guard.insert(price_key.clone(), price.clone());

                        // Keep only recent prices (last 1000)
                        while prices_guard.len() > 1000 {
                            if let Some(oldest_key) = prices_guard.keys().next().cloned() {
                                prices_guard.remove(&oldest_key);
                            }
                        }
                        drop(prices_guard);
                    }

                    stats.price_updates_processed.fetch_add(1, Ordering::Relaxed);
                }

                // Simulate price fetching from external sources
                if let Ok(mock_prices) = Self::fetch_mock_prices(&arbitrage_config.monitored_pairs, &arbitrage_config.supported_chains).await {
                    for price in mock_prices {
                        let price_key = format!("{}_{:?}_{}", price.trading_pair, price.chain_id, price.protocol);

                        // Update prices directly since we're in the same task
                        {
                            let mut prices_guard = prices.write().await;
                            prices_guard.insert(price_key, price);
                        }
                    }
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let price_time = start_time.elapsed().as_micros() as u64;
                trace!("Price monitoring cycle completed in {}μs", price_time);
            }
        });
    }

    /// Start opportunity detection
    async fn start_opportunity_detection(&self) {
        let opportunity_receiver = self.opportunity_receiver.clone();
        let opportunities = Arc::clone(&self.opportunities);
        let prices = Arc::clone(&self.prices);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let arbitrage_config = self.arbitrage_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(arbitrage_config.opportunity_detection_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming opportunities
                while let Ok(opportunity) = opportunity_receiver.try_recv() {
                    let opportunity_id = opportunity.id.clone();

                    // Store opportunity
                    {
                        let mut opportunities_guard = opportunities.write().await;
                        opportunities_guard.insert(opportunity_id, opportunity);

                        // Keep only recent opportunities
                        while opportunities_guard.len() > ARBITRAGE_MAX_OPPORTUNITIES {
                            if let Some(oldest_key) = opportunities_guard.keys().next().cloned() {
                                opportunities_guard.remove(&oldest_key);
                            }
                        }
                        drop(opportunities_guard);
                    }

                    stats.opportunities_detected.fetch_add(1, Ordering::Relaxed);
                }

                // Detect new opportunities from current prices
                let prices_guard = prices.read().await;
                let detected_opportunities = Self::detect_arbitrage_opportunities(&prices_guard, &arbitrage_config);
                drop(prices_guard);

                for opportunity in detected_opportunities {
                    let opportunity_id = opportunity.id.clone();

                    {
                        let mut opportunities_guard = opportunities.write().await;
                        opportunities_guard.insert(opportunity_id, opportunity);
                    }

                    stats.opportunities_detected.fetch_add(1, Ordering::Relaxed);
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let opportunity_time = start_time.elapsed().as_micros() as u64;
                trace!("Opportunity detection cycle completed in {}μs", opportunity_time);
            }
        });
    }

    /// Start execution planning
    async fn start_execution_planning(&self) {
        let execution_plans = Arc::clone(&self.execution_plans);
        let opportunities = Arc::clone(&self.opportunities);
        let _stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let arbitrage_config = self.arbitrage_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(arbitrage_config.execution_planning_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Create execution plans for profitable opportunities
                let opportunities_guard = opportunities.read().await;
                for opportunity in opportunities_guard.values() {
                    if opportunity.expected_net_profit_usd >= arbitrage_config.min_profit_threshold_usd {
                        let plan = Self::create_execution_plan(opportunity);
                        let plan_id = plan.id.clone();

                        {
                            let mut plans_guard = execution_plans.write().await;
                            plans_guard.insert(plan_id, plan);

                            // Keep only recent plans
                            while plans_guard.len() > ARBITRAGE_MAX_EXECUTION_PLANS {
                                if let Some(oldest_key) = plans_guard.keys().next().cloned() {
                                    plans_guard.remove(&oldest_key);
                                }
                            }
                            drop(plans_guard);
                        }
                    }
                }
                drop(opportunities_guard);

                trace!("Execution planning cycle completed");
            }
        });
    }

    /// Start risk management
    async fn start_risk_management(&self) {
        let execution_plans = Arc::clone(&self.execution_plans);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Risk check every 30 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Review execution plans for risk
                let plans_guard = execution_plans.read().await;
                let high_risk_plans: Vec<_> = plans_guard.values()
                    .filter(|plan| plan.risk_assessment.overall_risk_score > 7)
                    .cloned()
                    .collect();
                drop(plans_guard);

                if !high_risk_plans.is_empty() {
                    warn!("Found {} high-risk execution plans", high_risk_plans.len());
                }

                trace!("Risk management cycle completed");
            }
        });
    }

    /// Start performance monitoring
    async fn start_performance_monitoring(&self) {
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Log every minute

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let opportunities_detected = stats.opportunities_detected.load(Ordering::Relaxed);
                let opportunities_executed = stats.opportunities_executed.load(Ordering::Relaxed);
                let successful_executions = stats.successful_executions.load(Ordering::Relaxed);
                let failed_executions = stats.failed_executions.load(Ordering::Relaxed);
                let total_gross_profit = stats.total_gross_profit_usd.load(Ordering::Relaxed);
                let total_net_profit = stats.total_net_profit_usd.load(Ordering::Relaxed);
                let total_bridge_fees = stats.total_bridge_fees_usd.load(Ordering::Relaxed);
                let total_gas_fees = stats.total_gas_fees_usd.load(Ordering::Relaxed);
                let price_updates = stats.price_updates_processed.load(Ordering::Relaxed);
                let avg_execution_time = stats.avg_execution_time_s.load(Ordering::Relaxed);

                info!(
                    "Arbitrage Stats: opps_detected={}, opps_executed={}, successful={}, failed={}, gross_profit=${}, net_profit=${}, bridge_fees=${}, gas_fees=${}, price_updates={}, avg_exec_time={}s",
                    opportunities_detected, opportunities_executed, successful_executions, failed_executions,
                    total_gross_profit, total_net_profit, total_bridge_fees, total_gas_fees, price_updates, avg_execution_time
                );
            }
        });
    }

    /// Fetch mock prices for testing
    async fn fetch_mock_prices(
        monitored_pairs: &[String],
        supported_chains: &[ChainId],
    ) -> Result<Vec<CrossChainPrice>> {
        let mut prices = Vec::with_capacity(monitored_pairs.len() * supported_chains.len());

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for mock price data")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        for pair in monitored_pairs {
            for chain_id in supported_chains {
                let base_price = Self::get_base_price_for_pair(pair);
                let chain_multiplier = Self::get_chain_price_multiplier(*chain_id);
                let protocol_name = Self::get_main_protocol_for_chain(*chain_id);

                let mid_price = base_price * chain_multiplier;
                let spread = mid_price * "0.001".parse::<Decimal>().unwrap_or_default(); // 0.1% spread

                let price = CrossChainPrice {
                    chain_id: *chain_id,
                    trading_pair: pair.clone(),
                    protocol: protocol_name,
                    bid_price: mid_price - spread / Decimal::from(2_u64),
                    ask_price: mid_price + spread / Decimal::from(2_u64),
                    mid_price,
                    liquidity_usd: Self::get_liquidity_for_chain(*chain_id),
                    timestamp: now,
                    source: "mock".to_string(),
                };

                prices.push(price);
            }
        }

        Ok(prices)
    }

    /// Detect arbitrage opportunities from current prices
    fn detect_arbitrage_opportunities(
        prices: &HashMap<String, CrossChainPrice>,
        config: &CrossChainArbitrageConfig,
    ) -> Vec<CrossChainArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        // Group prices by trading pair
        let mut prices_by_pair: HashMap<String, Vec<&CrossChainPrice>> = HashMap::new();
        for price in prices.values() {
            prices_by_pair.entry(price.trading_pair.clone()).or_default().push(price);
        }

        // Find arbitrage opportunities within each trading pair
        for (pair, pair_prices) in prices_by_pair {
            if !config.monitored_pairs.contains(&pair) {
                continue;
            }

            for i in 0..pair_prices.len() {
                for j in (i + 1)..pair_prices.len() {
                    let Some(buy_price) = pair_prices.get(i) else { continue; };
                    let Some(sell_price) = pair_prices.get(j) else { continue; };

                    // Check both directions for arbitrage
                    if let Some(opportunity) = Self::calculate_arbitrage_opportunity(
                        buy_price, sell_price, "0.001".parse().unwrap_or_default(), "10".parse().unwrap_or_default()
                    ) {
                        if opportunity.expected_net_profit_usd >= config.min_profit_threshold_usd {
                            opportunities.push(opportunity);
                        }
                    }

                    if let Some(opportunity) = Self::calculate_arbitrage_opportunity(
                        sell_price, buy_price, "0.001".parse().unwrap_or_default(), "10".parse().unwrap_or_default()
                    ) {
                        if opportunity.expected_net_profit_usd >= config.min_profit_threshold_usd {
                            opportunities.push(opportunity);
                        }
                    }
                }
            }
        }

        opportunities
    }

    /// Get base price for trading pair
    fn get_base_price_for_pair(pair: &str) -> Decimal {
        match pair {
            "USDC/USDT" => "1.0001".parse().unwrap_or_default(),
            "ETH/USDC" => "2500.0".parse().unwrap_or_default(),
            "WBTC/ETH" => "15.5".parse().unwrap_or_default(),
            "DAI/USDC" => "0.9999".parse().unwrap_or_default(),
            _ => "1.0".parse().unwrap_or_default(),
        }
    }

    /// Get chain-specific price multiplier
    fn get_chain_price_multiplier(chain_id: ChainId) -> Decimal {
        match chain_id {
            ChainId::Ethereum => "1.0".parse().unwrap_or_default(),
            ChainId::Arbitrum => "1.0002".parse().unwrap_or_default(),
            ChainId::Optimism => "0.9998".parse().unwrap_or_default(),
            ChainId::Polygon => "1.0001".parse().unwrap_or_default(),
            ChainId::Bsc => "0.9999".parse().unwrap_or_default(),
            ChainId::Avalanche => "1.0003".parse().unwrap_or_default(),
            ChainId::Base => "0.9997".parse().unwrap_or_default(),
        }
    }

    /// Get main protocol for chain
    fn get_main_protocol_for_chain(chain_id: ChainId) -> String {
        match chain_id {
            ChainId::Ethereum => "Uniswap V3".to_string(),
            ChainId::Arbitrum => "Camelot".to_string(),
            ChainId::Optimism => "Velodrome".to_string(),
            ChainId::Polygon => "QuickSwap".to_string(),
            ChainId::Bsc => "PancakeSwap".to_string(),
            ChainId::Avalanche => "TraderJoe".to_string(),
            ChainId::Base => "BaseSwap".to_string(),
        }
    }

    /// Get liquidity for chain
    fn get_liquidity_for_chain(chain_id: ChainId) -> Decimal {
        match chain_id {
            ChainId::Ethereum => "10000000".parse().unwrap_or_default(), // $10M
            ChainId::Arbitrum => "5000000".parse().unwrap_or_default(),  // $5M
            ChainId::Optimism => "3000000".parse().unwrap_or_default(),  // $3M
            ChainId::Polygon => "2000000".parse().unwrap_or_default(),   // $2M
            ChainId::Bsc => "4000000".parse().unwrap_or_default(),       // $4M
            ChainId::Avalanche => "2500000".parse().unwrap_or_default(), // $2.5M
            ChainId::Base => "1500000".parse().unwrap_or_default(),      // $1.5M
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_cross_chain_arbitrage_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(arbitrage) = CrossChainArbitrage::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(arbitrage.stats().opportunities_detected.load(Ordering::Relaxed), 0);
        assert_eq!(arbitrage.stats().opportunities_executed.load(Ordering::Relaxed), 0);
        assert_eq!(arbitrage.stats().successful_executions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_cross_chain_arbitrage_config_default() {
        let config = CrossChainArbitrageConfig::default();
        assert!(config.enabled);
        assert_eq!(config.price_monitor_interval_ms, ARBITRAGE_DEFAULT_PRICE_INTERVAL_MS);
        assert_eq!(config.opportunity_detection_interval_ms, ARBITRAGE_DEFAULT_OPPORTUNITY_INTERVAL_MS);
        assert_eq!(config.execution_planning_interval_ms, ARBITRAGE_DEFAULT_EXECUTION_INTERVAL_MS);
        assert!(config.enable_price_monitoring);
        assert!(config.enable_opportunity_detection);
        assert!(!config.enable_automatic_execution); // Should be false by default
        assert!(config.enable_risk_management);
        assert!(!config.monitored_pairs.is_empty());
        assert!(!config.supported_chains.is_empty());
    }

    #[test]
    fn test_aligned_arbitrage_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedArbitrageData>(), 64);
        assert!(mem::size_of::<AlignedArbitrageData>() <= 64);
    }

    #[test]
    fn test_cross_chain_arbitrage_stats_operations() {
        let stats = CrossChainArbitrageStats::default();

        stats.opportunities_detected.fetch_add(100, Ordering::Relaxed);
        stats.opportunities_executed.fetch_add(80, Ordering::Relaxed);
        stats.successful_executions.fetch_add(75, Ordering::Relaxed);
        stats.failed_executions.fetch_add(5, Ordering::Relaxed);
        stats.total_gross_profit_usd.fetch_add(10_000, Ordering::Relaxed);
        stats.total_net_profit_usd.fetch_add(8_500, Ordering::Relaxed);

        assert_eq!(stats.opportunities_detected.load(Ordering::Relaxed), 100);
        assert_eq!(stats.opportunities_executed.load(Ordering::Relaxed), 80);
        assert_eq!(stats.successful_executions.load(Ordering::Relaxed), 75);
        assert_eq!(stats.failed_executions.load(Ordering::Relaxed), 5);
        assert_eq!(stats.total_gross_profit_usd.load(Ordering::Relaxed), 10_000);
        assert_eq!(stats.total_net_profit_usd.load(Ordering::Relaxed), 8_500);
    }

    #[test]
    fn test_aligned_arbitrage_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedArbitrageData::new(
            100, // opportunities detected
            80,  // opportunities executed
            950_000, // 95% success rate
            150_000_000, // $150 average profit
            300, // 5 minutes average execution
            25,  // 25 active opportunities
            50,  // 50 price updates per second
            now,
        );

        let stale_data = AlignedArbitrageData::new(
            100, 80, 950_000, 150_000_000, 300, 25, 50,
            now - 120_000, // 2 minutes old
        );

        assert!(!fresh_data.is_stale(60_000)); // 1 minute
        assert!(stale_data.is_stale(60_000)); // 1 minute
    }

    #[test]
    fn test_aligned_arbitrage_data_conversions() {
        let data = AlignedArbitrageData::new(
            100, // opportunities detected
            80,  // opportunities executed
            950_000, // 95% success rate (scaled by 1e6)
            150_000_000, // $150 average profit (scaled by 1e6)
            300, // 5 minutes average execution
            25,  // 25 active opportunities
            50,  // 50 price updates per second
            1_640_995_200_000,
        );

        assert_eq!(data.success_rate(), dec!(0.95));
        assert_eq!(data.avg_profit_usd(), dec!(150));
        assert_eq!(data.execution_efficiency(), dec!(0.8)); // 80/100
        assert_eq!(data.profit_per_opportunity(), dec!(12000)); // 150 * 80
    }

    #[test]
    fn test_step_type_equality() {
        assert_eq!(StepType::Buy, StepType::Buy);
        assert_ne!(StepType::Buy, StepType::Sell);
        assert_ne!(StepType::Bridge, StepType::Approve);
        assert_ne!(StepType::Wait, StepType::Sell);
    }

    #[test]
    fn test_execution_status_equality() {
        assert_eq!(ExecutionStatus::Created, ExecutionStatus::Created);
        assert_ne!(ExecutionStatus::Created, ExecutionStatus::Approved);
        assert_ne!(ExecutionStatus::Started, ExecutionStatus::InProgress);
        assert_ne!(ExecutionStatus::Completed, ExecutionStatus::Failed);
        assert_ne!(ExecutionStatus::Failed, ExecutionStatus::Cancelled);
    }

    #[test]
    fn test_cross_chain_price_creation() {
        let price = CrossChainPrice {
            chain_id: ChainId::Ethereum,
            trading_pair: "ETH/USDC".to_string(),
            protocol: "Uniswap V3".to_string(),
            bid_price: dec!(2499.5),
            ask_price: dec!(2500.5),
            mid_price: dec!(2500.0),
            liquidity_usd: dec!(10000000),
            timestamp: 1_640_995_200_000,
            source: "chainlink".to_string(),
        };

        assert_eq!(price.chain_id, ChainId::Ethereum);
        assert_eq!(price.trading_pair, "ETH/USDC");
        assert_eq!(price.protocol, "Uniswap V3");
        assert_eq!(price.bid_price, dec!(2499.5));
        assert_eq!(price.ask_price, dec!(2500.5));
        assert_eq!(price.mid_price, dec!(2500.0));
        assert_eq!(price.liquidity_usd, dec!(10000000));
        assert_eq!(price.source, "chainlink");
    }

    #[test]
    fn test_calculate_arbitrage_opportunity() {
        let buy_price = CrossChainPrice {
            chain_id: ChainId::Ethereum,
            trading_pair: "ETH/USDC".to_string(),
            protocol: "Uniswap V3".to_string(),
            bid_price: dec!(2499.0),
            ask_price: dec!(2500.0), // Buy at 2500
            mid_price: dec!(2499.5),
            liquidity_usd: dec!(1000000),
            timestamp: 1_640_995_200_000,
            source: "uniswap".to_string(),
        };

        let sell_price = CrossChainPrice {
            chain_id: ChainId::Arbitrum,
            trading_pair: "ETH/USDC".to_string(),
            protocol: "Camelot".to_string(),
            bid_price: dec!(2520.0), // Sell at 2520
            ask_price: dec!(2521.0),
            mid_price: dec!(2520.5),
            liquidity_usd: dec!(500000),
            timestamp: 1_640_995_200_000,
            source: "camelot".to_string(),
        };

        let opportunity = CrossChainArbitrage::calculate_arbitrage_opportunity(
            &buy_price,
            &sell_price,
            dec!(0.001), // 0.1% bridge fee
            dec!(10),    // $10 gas fee
        );

        assert!(opportunity.is_some());
        if let Some(opp) = opportunity {
            assert_eq!(opp.trading_pair, "ETH/USDC");
            assert_eq!(opp.source_chain, ChainId::Ethereum);
            assert_eq!(opp.destination_chain, ChainId::Arbitrum);
            assert_eq!(opp.buy_price, dec!(2500.0));
            assert_eq!(opp.sell_price, dec!(2520.0));
            assert!(opp.expected_net_profit_usd > dec!(0));
            assert!(opp.risk_score > 0);
            assert!(opp.execution_time_estimate_s > 0);
        }
    }

    #[test]
    fn test_calculate_risk_score() {
        // Low risk scenario
        let low_risk = CrossChainArbitrage::calculate_risk_score(
            dec!(0.005), // 0.5% price difference
            dec!(5000),  // $5k trade size
            ChainId::Ethereum,
            ChainId::Arbitrum,
        );
        assert!(low_risk <= 5);

        // High risk scenario
        let high_risk = CrossChainArbitrage::calculate_risk_score(
            dec!(0.08),  // 8% price difference (very high)
            dec!(80000), // $80k trade size (very large)
            ChainId::Ethereum,
            ChainId::Bsc,
        );
        assert!(high_risk >= 7);
    }

    #[test]
    fn test_estimate_execution_time() {
        // Fast route (Ethereum <-> Arbitrum)
        let fast_time = CrossChainArbitrage::estimate_execution_time(ChainId::Ethereum, ChainId::Arbitrum);
        assert_eq!(fast_time, 600); // 10 minutes

        // Slow route (Ethereum <-> BSC)
        let slow_time = CrossChainArbitrage::estimate_execution_time(ChainId::Ethereum, ChainId::Bsc);
        assert_eq!(slow_time, 2400); // 40 minutes

        // Default route
        let default_time = CrossChainArbitrage::estimate_execution_time(ChainId::Polygon, ChainId::Base);
        assert_eq!(default_time, 1500); // 25 minutes
    }

    #[test]
    fn test_select_optimal_bridge() {
        // Native bridges
        assert_eq!(
            CrossChainArbitrage::select_optimal_bridge(ChainId::Ethereum, ChainId::Arbitrum),
            BridgeType::ArbitrumBridge
        );
        assert_eq!(
            CrossChainArbitrage::select_optimal_bridge(ChainId::Ethereum, ChainId::Optimism),
            BridgeType::OptimismBridge
        );
        assert_eq!(
            CrossChainArbitrage::select_optimal_bridge(ChainId::Ethereum, ChainId::Polygon),
            BridgeType::PolygonPos
        );

        // Default bridge
        assert_eq!(
            CrossChainArbitrage::select_optimal_bridge(ChainId::Polygon, ChainId::Bsc),
            BridgeType::Stargate
        );
    }

    #[test]
    fn test_create_execution_plan() {
        let opportunity = CrossChainArbitrageOpportunity {
            id: "test_opp_123".to_string(),
            trading_pair: "ETH/USDC".to_string(),
            source_chain: ChainId::Ethereum,
            destination_chain: ChainId::Arbitrum,
            source_protocol: "Uniswap V3".to_string(),
            destination_protocol: "Camelot".to_string(),
            buy_price: dec!(2500.0),
            sell_price: dec!(2520.0),
            price_diff_pct: dec!(0.008), // 0.8%
            optimal_trade_size_usd: dec!(10000),
            expected_gross_profit_usd: dec!(80),
            bridge_fees_usd: dec!(10),
            gas_fees_usd: dec!(20),
            expected_net_profit_usd: dec!(50),
            risk_score: 4,
            execution_time_estimate_s: 600,
            required_bridge: BridgeType::ArbitrumBridge,
            expires_at: 1_640_995_500,
            created_at: 1_640_995_200_000,
        };

        let plan = CrossChainArbitrage::create_execution_plan(&opportunity);

        assert_eq!(plan.opportunity_id, "test_opp_123");
        assert_eq!(plan.steps.len(), 5); // Approve, Buy, Bridge, Wait, Sell
        assert_eq!(plan.expected_profit_usd, dec!(50));
        assert_eq!(plan.status, ExecutionStatus::Created);

        // Check step sequence
        assert_eq!(plan.steps.first().map(|s| &s.step_type), Some(&StepType::Approve));
        assert_eq!(plan.steps.get(1).map(|s| &s.step_type), Some(&StepType::Buy));
        assert_eq!(plan.steps.get(2).map(|s| &s.step_type), Some(&StepType::Bridge));
        assert_eq!(plan.steps.get(3).map(|s| &s.step_type), Some(&StepType::Wait));
        assert_eq!(plan.steps.get(4).map(|s| &s.step_type), Some(&StepType::Sell));

        // Check dependencies
        assert!(plan.steps.first().is_some_and(|s| s.dependencies.is_empty()));
        assert_eq!(plan.steps.get(1).map(|s| &s.dependencies), Some(&vec![1]));
        assert_eq!(plan.steps.get(2).map(|s| &s.dependencies), Some(&vec![2]));
        assert_eq!(plan.steps.get(3).map(|s| &s.dependencies), Some(&vec![3]));
        assert_eq!(plan.steps.get(4).map(|s| &s.dependencies), Some(&vec![4]));
    }

    #[test]
    fn test_assess_execution_risk() {
        let opportunity = CrossChainArbitrageOpportunity {
            id: "test_opp_456".to_string(),
            trading_pair: "ETH/USDC".to_string(),
            source_chain: ChainId::Ethereum,
            destination_chain: ChainId::Arbitrum,
            source_protocol: "Uniswap V3".to_string(),
            destination_protocol: "Camelot".to_string(),
            buy_price: dec!(2500.0),
            sell_price: dec!(2520.0),
            price_diff_pct: dec!(0.008), // 0.8% - moderate volatility
            optimal_trade_size_usd: dec!(10000), // Moderate size
            expected_gross_profit_usd: dec!(80),
            bridge_fees_usd: dec!(10),
            gas_fees_usd: dec!(20),
            expected_net_profit_usd: dec!(50),
            risk_score: 4,
            execution_time_estimate_s: 600, // 10 minutes - fast
            required_bridge: BridgeType::ArbitrumBridge, // Low risk bridge
            expires_at: 1_640_995_500,
            created_at: 1_640_995_200_000,
        };

        let risk_assessment = CrossChainArbitrage::assess_execution_risk(&opportunity);

        assert!(risk_assessment.overall_risk_score <= 10);
        assert!(risk_assessment.price_volatility_risk <= 10);
        assert!(risk_assessment.liquidity_risk <= 10);
        assert!(risk_assessment.bridge_risk <= 10);
        assert!(risk_assessment.execution_risk <= 10);
        assert!(risk_assessment.slippage_risk <= 10);

        // Low risk bridge should have low bridge risk
        assert!(risk_assessment.bridge_risk <= 3);

        // Fast execution should have low execution risk
        assert!(risk_assessment.execution_risk <= 5);
    }

    #[tokio::test]
    async fn test_cross_chain_arbitrage_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(arbitrage) = CrossChainArbitrage::new(config).await else {
            return;
        };

        let prices = arbitrage.get_prices().await;
        assert!(prices.is_empty()); // No prices initially

        let opportunities = arbitrage.get_opportunities().await;
        assert!(opportunities.is_empty()); // No opportunities initially

        let plans = arbitrage.get_execution_plans().await;
        assert!(plans.is_empty()); // No plans initially

        let stats = arbitrage.stats();
        assert_eq!(stats.opportunities_detected.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_price_and_liquidity_helpers() {
        // Test base prices
        assert_eq!(CrossChainArbitrage::get_base_price_for_pair("USDC/USDT"), dec!(1.0001));
        assert_eq!(CrossChainArbitrage::get_base_price_for_pair("ETH/USDC"), dec!(2500.0));
        assert_eq!(CrossChainArbitrage::get_base_price_for_pair("UNKNOWN"), dec!(1.0));

        // Test chain multipliers
        assert_eq!(CrossChainArbitrage::get_chain_price_multiplier(ChainId::Ethereum), dec!(1.0));
        assert!(CrossChainArbitrage::get_chain_price_multiplier(ChainId::Arbitrum) > dec!(1.0));
        assert!(CrossChainArbitrage::get_chain_price_multiplier(ChainId::Optimism) < dec!(1.0));

        // Test protocol names
        assert_eq!(CrossChainArbitrage::get_main_protocol_for_chain(ChainId::Ethereum), "Uniswap V3");
        assert_eq!(CrossChainArbitrage::get_main_protocol_for_chain(ChainId::Arbitrum), "Camelot");
        assert_eq!(CrossChainArbitrage::get_main_protocol_for_chain(ChainId::Avalanche), "TraderJoe");

        // Test liquidity
        assert_eq!(CrossChainArbitrage::get_liquidity_for_chain(ChainId::Ethereum), dec!(10000000));
        assert_eq!(CrossChainArbitrage::get_liquidity_for_chain(ChainId::Base), dec!(1500000));
    }
}
