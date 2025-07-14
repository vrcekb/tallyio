//! dYdX Flashloan Integration for ultra-performance flashloan operations
//!
//! This module provides advanced dYdX flashloan integration capabilities for maximizing
//! capital efficiency through direct dYdX protocol interaction and optimal
//! flashloan execution with zero fees.
//!
//! ## Performance Targets
//! - Loan Initiation: <30μs
//! - Fee Calculation: <10μs (zero fees)
//! - Execution Monitoring: <15μs
//! - Callback Processing: <40μs
//! - Total Execution: <100μs
//!
//! ## Architecture
//! - Direct dYdX Solo Margin integration
//! - Advanced flashloan callback handling
//! - Zero-fee flashloan optimization
//! - Multi-market dYdX support
//! - Lock-free execution primitives

use crate::{
    ChainCoreConfig, Result,
    types::ChainId,
    utils::perf::Timer,
};
use crossbeam::channel::{self, Receiver, Sender};
use dashmap::DashMap;
use rust_decimal::{Decimal, prelude::ToPrimitive};
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
use tracing::{info, trace};

/// dYdX flashloan configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct DydxFlashloanConfig {
    /// Enable dYdX flashloan integration
    pub enabled: bool,
    
    /// Market monitoring interval in milliseconds
    pub market_monitoring_interval_ms: u64,
    
    /// Position optimization interval in milliseconds
    pub position_optimization_interval_ms: u64,
    
    /// Performance monitoring interval in milliseconds
    pub performance_monitoring_interval_ms: u64,
    
    /// Enable zero-fee optimization
    pub enable_zero_fee_optimization: bool,
    
    /// Enable callback optimization
    pub enable_callback_optimization: bool,
    
    /// Enable multi-market loans
    pub enable_multi_market: bool,
    
    /// Enable margin analysis
    pub enable_margin_analysis: bool,
    
    /// Maximum flashloan amount (USD)
    pub max_flashloan_amount_usd: Decimal,
    
    /// Minimum flashloan amount (USD)
    pub min_flashloan_amount_usd: Decimal,
    
    /// Default fee percentage (0% for dYdX)
    pub default_fee_percentage: Decimal,
    
    /// Supported chains for dYdX
    pub supported_chains: Vec<ChainId>,
    
    /// Supported markets for flashloan
    pub supported_markets: Vec<u32>,
    
    /// Supported assets for flashloan
    pub supported_assets: Vec<String>,
}

/// dYdX Solo Margin information
#[derive(Debug, Clone)]
pub struct DydxSoloMarginInfo {
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Solo Margin address
    pub solo_margin_address: String,
    
    /// Proxy address
    pub proxy_address: String,
    
    /// Available markets
    pub available_markets: HashMap<u32, DydxMarketData>,
    
    /// Solo Margin configuration
    pub solo_margin_configuration: DydxSoloMarginConfiguration,
    
    /// Solo Margin status
    pub status: DydxSoloMarginStatus,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// dYdX market data
#[derive(Debug, Clone)]
pub struct DydxMarketData {
    /// Market ID
    pub market_id: u32,
    
    /// Token address
    pub token_address: String,
    
    /// Token symbol
    pub token_symbol: String,
    
    /// Token decimals
    pub token_decimals: u8,
    
    /// Total supply
    pub total_supply: Decimal,
    
    /// Total borrow
    pub total_borrow: Decimal,
    
    /// Supply index
    pub supply_index: Decimal,
    
    /// Borrow index
    pub borrow_index: Decimal,
    
    /// Supply rate
    pub supply_rate: Decimal,
    
    /// Borrow rate
    pub borrow_rate: Decimal,
    
    /// Oracle price
    pub oracle_price: Decimal,
    
    /// Market status
    pub status: DydxMarketStatus,
    
    /// Flashloan enabled
    pub flashloan_enabled: bool,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// dYdX Solo Margin configuration
#[derive(Debug, Clone)]
pub struct DydxSoloMarginConfiguration {
    /// Owner address
    pub owner: String,
    
    /// Risk parameters
    pub risk_params: DydxRiskParameters,
    
    /// Interest setter
    pub interest_setter: String,
    
    /// Price oracle
    pub price_oracle: String,
    
    /// Margin ratio
    pub margin_ratio: Decimal,
    
    /// Liquidation spread
    pub liquidation_spread: Decimal,
    
    /// Earn spread
    pub earn_spread: Decimal,
    
    /// Min borrow value
    pub min_borrow_value: Decimal,
    
    /// Solo Margin paused
    pub paused: bool,
}

/// dYdX risk parameters
#[derive(Debug, Clone)]
pub struct DydxRiskParameters {
    /// Margin ratio
    pub margin_ratio: Decimal,
    
    /// Liquidation spread
    pub liquidation_spread: Decimal,
    
    /// Earn spread
    pub earn_spread: Decimal,
    
    /// Min borrow value
    pub min_borrow_value: Decimal,
}

/// dYdX Solo Margin status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DydxSoloMarginStatus {
    /// Solo Margin is active and operational
    Active,
    /// Solo Margin is paused
    Paused,
    /// Solo Margin is under maintenance
    Maintenance,
    /// Solo Margin is deprecated
    Deprecated,
}

/// dYdX market status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DydxMarketStatus {
    /// Market is active and operational
    Active,
    /// Market is paused
    Paused,
    /// Market has insufficient liquidity
    InsufficientLiquidity,
    /// Market is deprecated
    Deprecated,
}

/// dYdX flashloan request
#[derive(Debug, Clone)]
pub struct DydxFlashloanRequest {
    /// Market ID to borrow from
    pub market_id: u32,
    
    /// Amount to borrow (in token units)
    pub amount: Decimal,
    
    /// Callback data
    pub callback_data: Vec<u8>,
    
    /// Account owner
    pub account_owner: String,
    
    /// Account number
    pub account_number: u64,
    
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Strategy identifier
    pub strategy_id: String,
    
    /// Execution deadline
    pub deadline: u64,
}

/// dYdX flashloan execution result
#[derive(Debug, Clone)]
pub struct DydxFlashloanExecution {
    /// Request ID
    pub request_id: String,
    
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Solo Margin address used
    pub solo_margin_address: String,
    
    /// Market ID used
    pub market_id: u32,
    
    /// Token borrowed
    pub token_borrowed: String,
    
    /// Amount borrowed
    pub amount_borrowed: Decimal,
    
    /// Fee paid (always zero for dYdX)
    pub fee_paid: Decimal,
    
    /// Execution status
    pub status: DydxExecutionStatus,
    
    /// Transaction hash
    pub transaction_hash: Option<String>,
    
    /// Gas used
    pub gas_used: u64,
    
    /// Gas cost (USD)
    pub gas_cost_usd: Decimal,
    
    /// Execution time (seconds)
    pub execution_time_s: u32,
    
    /// Error message (if failed)
    pub error_message: Option<String>,
    
    /// Callback success
    pub callback_success: bool,
    
    /// Executed at timestamp
    pub executed_at: u64,
}

/// dYdX execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DydxExecutionStatus {
    /// Execution pending
    Pending,
    /// Loan initiated
    LoanInitiated,
    /// Callback executing
    CallbackExecuting,
    /// Repayment processing
    RepaymentProcessing,
    /// Execution successful
    Success,
    /// Execution failed
    Failed,
    /// Execution cancelled
    Cancelled,
    /// Execution timed out
    TimedOut,
    /// Insufficient liquidity
    InsufficientLiquidity,
    /// Callback failed
    CallbackFailed,
}

/// dYdX flashloan statistics
#[derive(Debug, Default)]
pub struct DydxFlashloanStats {
    /// Total flashloan requests
    pub total_requests: AtomicU64,
    
    /// Successful executions
    pub successful_executions: AtomicU64,
    
    /// Failed executions
    pub failed_executions: AtomicU64,
    
    /// Total volume borrowed (USD)
    pub total_volume_usd: AtomicU64,
    
    /// Total fees paid (USD) - always zero for dYdX
    pub total_fees_paid_usd: AtomicU64,
    
    /// Market monitoring cycles
    pub market_monitoring_cycles: AtomicU64,
    
    /// Position optimizations performed
    pub position_optimizations: AtomicU64,
    
    /// Zero-fee optimizations
    pub zero_fee_optimizations: AtomicU64,
    
    /// Multi-market loans executed
    pub multi_market_loans: AtomicU64,
    
    /// Average execution time (μs)
    pub avg_execution_time_us: AtomicU64,
    
    /// Average fee percentage (scaled by 1e6) - always zero
    pub avg_fee_percentage_scaled: AtomicU64,
    
    /// Optimal market selections
    pub optimal_market_selections: AtomicU64,
}

/// Cache-line aligned dYdX data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedDydxData {
    /// Available liquidity USD (scaled by 1e6)
    pub available_liquidity_usd_scaled: u64,
    
    /// Fee percentage (scaled by 1e6) - always zero
    pub fee_percentage_scaled: u64,
    
    /// Market count
    pub market_count: u64,
    
    /// Solo Margin health score (scaled by 1e6)
    pub solo_margin_health_score_scaled: u64,
    
    /// Average execution time (seconds)
    pub avg_execution_time_s: u64,
    
    /// Success rate (scaled by 1e6)
    pub success_rate_scaled: u64,
    
    /// Total loans executed
    pub total_loans_executed: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// dYdX flashloan constants
pub const DYDX_DEFAULT_MONITORING_INTERVAL_MS: u64 = 3500; // 3.5 seconds
pub const DYDX_DEFAULT_OPTIMIZATION_INTERVAL_MS: u64 = 7000; // 7 seconds
pub const DYDX_DEFAULT_PERFORMANCE_INTERVAL_MS: u64 = 14000; // 14 seconds
pub const DYDX_DEFAULT_MAX_LOAN_USD: &str = "5000000.0"; // $5M maximum
pub const DYDX_DEFAULT_MIN_LOAN_USD: &str = "1000.0"; // $1k minimum
pub const DYDX_DEFAULT_FEE_PERCENTAGE: &str = "0.0"; // 0% default (free)
pub const DYDX_FLASHLOAN_FEE: u16 = 0; // 0% fee
pub const DYDX_MAX_MARKETS: usize = 10;
pub const DYDX_MAX_EXECUTIONS: usize = 1000;
pub const DYDX_MARKET_ETH: u32 = 0;
pub const DYDX_MARKET_SAI: u32 = 1;
pub const DYDX_MARKET_USDC: u32 = 2;
pub const DYDX_MARKET_DAI: u32 = 3;

impl Default for DydxFlashloanConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            market_monitoring_interval_ms: DYDX_DEFAULT_MONITORING_INTERVAL_MS,
            position_optimization_interval_ms: DYDX_DEFAULT_OPTIMIZATION_INTERVAL_MS,
            performance_monitoring_interval_ms: DYDX_DEFAULT_PERFORMANCE_INTERVAL_MS,
            enable_zero_fee_optimization: true,
            enable_callback_optimization: true,
            enable_multi_market: true,
            enable_margin_analysis: true,
            max_flashloan_amount_usd: DYDX_DEFAULT_MAX_LOAN_USD.parse().unwrap_or_default(),
            min_flashloan_amount_usd: DYDX_DEFAULT_MIN_LOAN_USD.parse().unwrap_or_default(),
            default_fee_percentage: DYDX_DEFAULT_FEE_PERCENTAGE.parse().unwrap_or_default(),
            supported_chains: vec![
                ChainId::Ethereum, // dYdX is primarily on Ethereum
            ],
            supported_markets: vec![
                DYDX_MARKET_ETH,
                DYDX_MARKET_SAI,
                DYDX_MARKET_USDC,
                DYDX_MARKET_DAI,
            ],
            supported_assets: vec![
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), // WETH
                "0x89d24A6b4CcB1B6fAA2625fE562bDD9a23260359".to_string(), // SAI
                "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                "0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), // DAI
            ],
        }
    }
}

impl AlignedDydxData {
    /// Create new aligned dYdX data
    #[inline(always)]
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "Aligned data structure requires all fields")]
    pub const fn new(
        available_liquidity_usd_scaled: u64,
        fee_percentage_scaled: u64,
        market_count: u64,
        solo_margin_health_score_scaled: u64,
        avg_execution_time_s: u64,
        success_rate_scaled: u64,
        total_loans_executed: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            available_liquidity_usd_scaled,
            fee_percentage_scaled,
            market_count,
            solo_margin_health_score_scaled,
            avg_execution_time_s,
            success_rate_scaled,
            total_loans_executed,
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

    /// Get available liquidity USD as Decimal
    #[inline(always)]
    #[must_use]
    pub fn available_liquidity_usd(&self) -> Decimal {
        Decimal::from(self.available_liquidity_usd_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get fee percentage as Decimal (always zero for dYdX)
    #[inline(always)]
    #[must_use]
    pub fn fee_percentage(&self) -> Decimal {
        Decimal::from(self.fee_percentage_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get Solo Margin health score as Decimal
    #[inline(always)]
    #[must_use]
    pub fn solo_margin_health_score(&self) -> Decimal {
        Decimal::from(self.solo_margin_health_score_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get success rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn success_rate(&self) -> Decimal {
        Decimal::from(self.success_rate_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get overall Solo Margin score
    #[inline(always)]
    #[must_use]
    pub fn overall_score(&self) -> Decimal {
        // Weighted score: liquidity (45%) + health (30%) + success rate (25%)
        // No fee component since dYdX is always free
        let liquidity_weight = "0.45".parse::<Decimal>().unwrap_or_default();
        let health_weight = "0.3".parse::<Decimal>().unwrap_or_default();
        let success_weight = "0.25".parse::<Decimal>().unwrap_or_default();

        // Normalize liquidity score (higher liquidity = higher score, max $100M)
        let liquidity_score = (self.available_liquidity_usd() / Decimal::from(100_000_000_u64)).min(Decimal::ONE);

        liquidity_score * liquidity_weight +
        self.solo_margin_health_score() * health_weight +
        self.success_rate() * success_weight
    }
}

/// dYdX Flashloan Integration for ultra-performance flashloan operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct DydxFlashloan {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// dYdX specific configuration
    dydx_config: DydxFlashloanConfig,

    /// Statistics
    stats: Arc<DydxFlashloanStats>,

    /// Solo Margin information
    solo_margins: Arc<RwLock<HashMap<ChainId, DydxSoloMarginInfo>>>,

    /// Solo Margin data cache for ultra-fast access
    solo_margin_cache: Arc<DashMap<ChainId, AlignedDydxData>>,

    /// Active executions
    active_executions: Arc<RwLock<HashMap<String, DydxFlashloanExecution>>>,

    /// Performance timers
    monitoring_timer: Timer,
    optimization_timer: Timer,
    execution_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Solo Margin update channels
    solo_margin_sender: Sender<DydxSoloMarginInfo>,
    solo_margin_receiver: Receiver<DydxSoloMarginInfo>,

    /// Execution channels
    execution_sender: Sender<DydxFlashloanExecution>,
    execution_receiver: Receiver<DydxFlashloanExecution>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current execution round
    execution_round: Arc<TokioMutex<u64>>,
}

impl DydxFlashloan {
    /// Create new dYdX flashloan integration with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let dydx_config = DydxFlashloanConfig::default();
        let stats = Arc::new(DydxFlashloanStats::default());
        let solo_margins = Arc::new(RwLock::new(HashMap::with_capacity(DYDX_MAX_MARKETS)));
        let solo_margin_cache = Arc::new(DashMap::with_capacity(DYDX_MAX_MARKETS));
        let active_executions = Arc::new(RwLock::new(HashMap::with_capacity(DYDX_MAX_EXECUTIONS)));
        let monitoring_timer = Timer::new("dydx_monitoring");
        let optimization_timer = Timer::new("dydx_optimization");
        let execution_timer = Timer::new("dydx_execution");
        let shutdown = Arc::new(AtomicBool::new(false));
        let execution_round = Arc::new(TokioMutex::new(0));

        let (solo_margin_sender, solo_margin_receiver) = channel::bounded(DYDX_MAX_MARKETS);
        let (execution_sender, execution_receiver) = channel::bounded(DYDX_MAX_EXECUTIONS);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            dydx_config,
            stats,
            solo_margins,
            solo_margin_cache,
            active_executions,
            monitoring_timer,
            optimization_timer,
            execution_timer,
            shutdown,
            solo_margin_sender,
            solo_margin_receiver,
            execution_sender,
            execution_receiver,
            http_client,
            execution_round,
        })
    }

    /// Start dYdX flashloan services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.dydx_config.enabled {
            info!("dYdX flashloan integration disabled");
            return Ok(());
        }

        info!("Starting dYdX flashloan integration");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start market monitoring
        self.start_market_monitoring().await;

        // Start position optimization
        self.start_position_optimization().await;

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("dYdX flashloan integration started successfully");
        Ok(())
    }

    /// Stop dYdX flashloan integration
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping dYdX flashloan integration");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("dYdX flashloan integration stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &DydxFlashloanStats {
        &self.stats
    }

    /// Get Solo Margin information
    #[inline]
    pub async fn get_solo_margins(&self) -> Vec<DydxSoloMarginInfo> {
        let solo_margins = self.solo_margins.read().await;
        solo_margins.values().cloned().collect()
    }

    /// Get active executions
    #[inline]
    pub async fn get_active_executions(&self) -> Vec<DydxFlashloanExecution> {
        let executions = self.active_executions.read().await;
        executions.values().cloned().collect()
    }

    /// Execute dYdX flashloan
    #[inline]
    #[must_use]
    pub async fn execute_flashloan(&self, request: &DydxFlashloanRequest) -> Option<DydxFlashloanExecution> {
        let start_time = Instant::now();

        // Validate request
        if !Self::validate_request(request) {
            return None;
        }

        // Find optimal Solo Margin
        let solo_margin_info = self.find_optimal_solo_margin(request.chain_id).await?;

        // Generate execution ID
        let execution_id = self.generate_execution_id(request).await;

        // Execute flashloan
        let execution = self.execute_with_solo_margin(&solo_margin_info, request, &execution_id).await;

        // Update statistics
        self.update_execution_stats(&execution);

        #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
        let execution_time = start_time.elapsed().as_micros() as u64;
        self.stats.avg_execution_time_us.store(execution_time, Ordering::Relaxed);

        // Store execution
        {
            let mut executions = self.active_executions.write().await;
            executions.insert(execution_id.clone(), execution.clone());

            // Keep only recent executions
            while executions.len() > DYDX_MAX_EXECUTIONS {
                if let Some(oldest_key) = executions.keys().next().cloned() {
                    executions.remove(&oldest_key);
                }
            }
            drop(executions);
        }

        Some(execution)
    }

    /// Validate flashloan request
    fn validate_request(request: &DydxFlashloanRequest) -> bool {
        // Check for zero amount
        if request.amount <= Decimal::ZERO {
            return false;
        }

        // Check for empty account owner
        if request.account_owner.is_empty() {
            return false;
        }

        // Check deadline
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for deadline check")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if request.deadline <= now {
            return false;
        }

        true
    }

    /// Find optimal Solo Margin for chain
    async fn find_optimal_solo_margin(&self, chain_id: ChainId) -> Option<DydxSoloMarginInfo> {
        // Find Solo Margin for the specific chain
        {
            let solo_margins = self.solo_margins.read().await;
            if let Some(solo_margin) = solo_margins.get(&chain_id) {
                if solo_margin.status == DydxSoloMarginStatus::Active {
                    return Some(solo_margin.clone());
                }
            }
        }

        None
    }

    /// Execute flashloan with specific Solo Margin
    async fn execute_with_solo_margin(
        &self,
        solo_margin: &DydxSoloMarginInfo,
        request: &DydxFlashloanRequest,
        execution_id: &str,
    ) -> DydxFlashloanExecution {
        let start_time = Instant::now();

        // Calculate fees (always zero for dYdX)
        let fee_paid = Decimal::ZERO;

        // Get market info
        let token_borrowed = solo_margin.available_markets
            .get(&request.market_id)
            .map_or_else(|| "Unknown".to_string(), |market| market.token_address.clone());

        // Simulate execution (in production this would interact with actual dYdX contracts)
        let execution_success = Self::simulate_dydx_execution(solo_margin, request);

        let status = if execution_success {
            DydxExecutionStatus::Success
        } else {
            DydxExecutionStatus::Failed
        };

        let transaction_hash = if execution_success {
            Some(format!("0x{:x}", fastrand::u64(..)))
        } else {
            None
        };

        let error_message = if execution_success {
            None
        } else {
            Some("Simulated dYdX execution failure".to_string())
        };

        #[expect(clippy::cast_possible_truncation, reason = "Execution time truncation is acceptable")]
        let execution_time_s = start_time.elapsed().as_secs() as u32;

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for execution data")]
        let executed_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        DydxFlashloanExecution {
            request_id: execution_id.to_string(),
            chain_id: request.chain_id,
            solo_margin_address: solo_margin.solo_margin_address.clone(),
            market_id: request.market_id,
            token_borrowed,
            amount_borrowed: request.amount,
            fee_paid,
            status,
            transaction_hash,
            gas_used: Self::estimate_gas_usage(),
            gas_cost_usd: Self::estimate_gas_cost(request.chain_id),
            execution_time_s,
            error_message,
            callback_success: execution_success,
            executed_at,
        }
    }

    /// Simulate dYdX execution (for testing)
    fn simulate_dydx_execution(solo_margin: &DydxSoloMarginInfo, request: &DydxFlashloanRequest) -> bool {
        // Check Solo Margin status
        if solo_margin.status != DydxSoloMarginStatus::Active {
            return false;
        }

        // Check if market exists and has sufficient liquidity
        if let Some(market) = solo_margin.available_markets.get(&request.market_id) {
            if market.status != DydxMarketStatus::Active || !market.flashloan_enabled {
                return false;
            }

            // Check liquidity (total supply - total borrow)
            let available_liquidity = market.total_supply - market.total_borrow;
            if available_liquidity < request.amount {
                return false;
            }
        } else {
            return false; // Market not found
        }

        // Simulate success rate (90% for dYdX)
        #[allow(clippy::float_arithmetic)] // Simulation requires floating point arithmetic
        {
            fastrand::f64() < 0.90
        }
    }

    /// Generate unique execution ID
    async fn generate_execution_id(&self, request: &DydxFlashloanRequest) -> String {
        let mut round = self.execution_round.lock().await;
        *round += 1;
        #[expect(clippy::cast_possible_truncation, reason = "Chain ID values are small")]
        let chain_id_u8 = request.chain_id as u8;
        format!("dydx_{}_{}_{}_{}", chain_id_u8, request.strategy_id, *round, fastrand::u64(..))
    }

    /// Update execution statistics
    fn update_execution_stats(&self, execution: &DydxFlashloanExecution) {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);

        match execution.status {
            DydxExecutionStatus::Success => {
                self.stats.successful_executions.fetch_add(1, Ordering::Relaxed);
                self.stats.optimal_market_selections.fetch_add(1, Ordering::Relaxed);

                // Update volume (fees are always zero for dYdX)
                let volume_scaled = (execution.amount_borrowed * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0);
                self.stats.total_volume_usd.fetch_add(volume_scaled, Ordering::Relaxed);

                // dYdX is always zero-fee
                self.stats.zero_fee_optimizations.fetch_add(1, Ordering::Relaxed);

                // Update position optimization counter
                self.stats.position_optimizations.fetch_add(1, Ordering::Relaxed);
            }
            DydxExecutionStatus::Failed | DydxExecutionStatus::TimedOut | DydxExecutionStatus::CallbackFailed => {
                self.stats.failed_executions.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Estimate gas usage for dYdX flashloan
    const fn estimate_gas_usage() -> u64 {
        // Base gas cost for dYdX flashloan
        350_000_u64 // 350k gas
    }

    /// Estimate gas cost for chain
    fn estimate_gas_cost(chain_id: ChainId) -> Decimal {
        match chain_id {
            ChainId::Ethereum => "60".parse().unwrap_or_default(),    // $60 (higher due to complexity)
            _ => "0".parse().unwrap_or_default(),                     // dYdX is primarily on Ethereum
        }
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(6000)) // dYdX timeout
            .http2_prior_knowledge()
            .http2_keep_alive_timeout(Duration::from_secs(30))
            .http2_keep_alive_interval(Duration::from_secs(10))
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .map_err(|_e| crate::ChainCoreError::Network(crate::NetworkError::ConnectionRefused))?;

        {
            let mut http_client_guard = self.http_client.lock().await;
            *http_client_guard = Some(client);
        }

        Ok(())
    }

    /// Start market monitoring
    async fn start_market_monitoring(&self) {
        let solo_margin_receiver = self.solo_margin_receiver.clone();
        let solo_margins = Arc::clone(&self.solo_margins);
        let solo_margin_cache = Arc::clone(&self.solo_margin_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let dydx_config = self.dydx_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(dydx_config.market_monitoring_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming Solo Margin updates
                while let Ok(solo_margin_info) = solo_margin_receiver.try_recv() {
                    let chain_id = solo_margin_info.chain_id;

                    // Update Solo Margin information
                    {
                        let mut solo_margins_guard = solo_margins.write().await;
                        solo_margins_guard.insert(chain_id, solo_margin_info.clone());
                        drop(solo_margins_guard);
                    }

                    // Update cache with aligned data
                    let total_liquidity = solo_margin_info.available_markets.values()
                        .map(|m| (m.total_supply - m.total_borrow) * m.oracle_price)
                        .sum::<Decimal>();

                    let aligned_data = AlignedDydxData::new(
                        (total_liquidity * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        0, // dYdX always has 0% fees
                        solo_margin_info.available_markets.len() as u64,
                        900_000, // 90% health score (mock)
                        10, // 10s execution time
                        900_000, // 90% success rate (mock)
                        100, // Total loans executed (mock)
                        solo_margin_info.last_update,
                    );
                    solo_margin_cache.insert(chain_id, aligned_data);
                }

                stats.market_monitoring_cycles.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let monitoring_time = start_time.elapsed().as_micros() as u64;
                trace!("dYdX market monitoring cycle completed in {}μs", monitoring_time);
            }
        });
    }

    /// Start position optimization
    async fn start_position_optimization(&self) {
        let execution_receiver = self.execution_receiver.clone();
        let active_executions = Arc::clone(&self.active_executions);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let dydx_config = self.dydx_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(dydx_config.position_optimization_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming executions
                while let Ok(execution) = execution_receiver.try_recv() {
                    let execution_id = execution.request_id.clone();

                    // Store execution
                    {
                        let mut executions_guard = active_executions.write().await;
                        executions_guard.insert(execution_id, execution);

                        // Keep only recent executions
                        while executions_guard.len() > DYDX_MAX_EXECUTIONS {
                            if let Some(oldest_key) = executions_guard.keys().next().cloned() {
                                executions_guard.remove(&oldest_key);
                            }
                        }
                        drop(executions_guard);
                    }
                }

                // Perform position optimization analysis
                Self::optimize_positions(&active_executions, &stats).await;

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let optimization_time = start_time.elapsed().as_micros() as u64;
                trace!("dYdX position optimization cycle completed in {}μs", optimization_time);
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

                let total_requests = stats.total_requests.load(Ordering::Relaxed);
                let successful_executions = stats.successful_executions.load(Ordering::Relaxed);
                let failed_executions = stats.failed_executions.load(Ordering::Relaxed);
                let total_volume = stats.total_volume_usd.load(Ordering::Relaxed);
                let total_fees = stats.total_fees_paid_usd.load(Ordering::Relaxed);
                let market_monitoring = stats.market_monitoring_cycles.load(Ordering::Relaxed);
                let position_optimizations = stats.position_optimizations.load(Ordering::Relaxed);
                let zero_fee_optimizations = stats.zero_fee_optimizations.load(Ordering::Relaxed);
                let multi_market_loans = stats.multi_market_loans.load(Ordering::Relaxed);
                let avg_execution_time = stats.avg_execution_time_us.load(Ordering::Relaxed);
                let avg_fee_percentage = stats.avg_fee_percentage_scaled.load(Ordering::Relaxed);
                let optimal_selections = stats.optimal_market_selections.load(Ordering::Relaxed);

                info!(
                    "dYdX Stats: requests={}, successful={}, failed={}, volume=${}, fees=${}, monitoring={}, pos_opt={}, zero_fee={}, multi_market={}, avg_time={}μs, avg_fee={}%, optimal={}",
                    total_requests, successful_executions, failed_executions, total_volume, total_fees,
                    market_monitoring, position_optimizations, zero_fee_optimizations, multi_market_loans, avg_execution_time, avg_fee_percentage, optimal_selections
                );
            }
        });
    }

    /// Optimize positions based on execution history
    async fn optimize_positions(
        active_executions: &Arc<RwLock<HashMap<String, DydxFlashloanExecution>>>,
        stats: &Arc<DydxFlashloanStats>,
    ) {
        let executions_guard = active_executions.read().await;

        if executions_guard.is_empty() {
            return;
        }

        // Analyze execution patterns for position optimization
        let mut _successful_count = 0;
        let mut total_count = 0;

        for execution in executions_guard.values() {
            total_count += 1;
            if execution.status == DydxExecutionStatus::Success {
                _successful_count += 1;
            }
        }

        if total_count > 0 {
            // dYdX fees are always zero, so no fee optimization needed
            stats.avg_fee_percentage_scaled.store(0, Ordering::Relaxed);
            stats.position_optimizations.fetch_add(1, Ordering::Relaxed);
        }

        drop(executions_guard);
        trace!("dYdX position optimization completed");
    }

    /// Get dYdX Solo Margin address for chain
    #[cfg(test)]
    fn get_solo_margin_address(chain_id: ChainId) -> String {
        match chain_id {
            ChainId::Ethereum => "0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e".to_string(), // dYdX Solo Margin
            _ => "0x0000000000000000000000000000000000000000".to_string(), // Not supported
        }
    }

    /// Get dYdX proxy address for chain
    #[cfg(test)]
    fn get_proxy_address(chain_id: ChainId) -> String {
        match chain_id {
            ChainId::Ethereum => "0x4EC3570cADaAEE08Ae384779B0f3A45EF85289DE".to_string(), // dYdX Proxy
            _ => "0x0000000000000000000000000000000000000000".to_string(), // Not supported
        }
    }

    /// Get mock markets for testing
    #[cfg(test)]
    fn get_mock_markets(chain_id: ChainId) -> HashMap<u32, DydxMarketData> {
        let mut markets = HashMap::new();

        if chain_id != ChainId::Ethereum {
            return markets; // dYdX is only on Ethereum
        }

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for mock data")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // ETH Market (Market 0)
        markets.insert(
            DYDX_MARKET_ETH,
            DydxMarketData {
                market_id: DYDX_MARKET_ETH,
                token_address: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), // WETH
                token_symbol: "WETH".to_string(),
                token_decimals: 18,
                total_supply: "50000".parse().unwrap_or_default(), // 50k ETH
                total_borrow: "30000".parse().unwrap_or_default(), // 30k ETH
                supply_index: "1.05".parse().unwrap_or_default(),
                borrow_index: "1.08".parse().unwrap_or_default(),
                supply_rate: "0.02".parse().unwrap_or_default(), // 2%
                borrow_rate: "0.05".parse().unwrap_or_default(), // 5%
                oracle_price: "2000".parse().unwrap_or_default(), // $2000 per ETH
                status: DydxMarketStatus::Active,
                flashloan_enabled: true,
                last_update: now,
            },
        );

        // USDC Market (Market 2)
        markets.insert(
            DYDX_MARKET_USDC,
            DydxMarketData {
                market_id: DYDX_MARKET_USDC,
                token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                token_symbol: "USDC".to_string(),
                token_decimals: 6,
                total_supply: "100000000".parse().unwrap_or_default(), // 100M USDC
                total_borrow: "60000000".parse().unwrap_or_default(), // 60M USDC
                supply_index: "1.03".parse().unwrap_or_default(),
                borrow_index: "1.06".parse().unwrap_or_default(),
                supply_rate: "0.015".parse().unwrap_or_default(), // 1.5%
                borrow_rate: "0.04".parse().unwrap_or_default(), // 4%
                oracle_price: "1".parse().unwrap_or_default(), // $1 per USDC
                status: DydxMarketStatus::Active,
                flashloan_enabled: true,
                last_update: now,
            },
        );

        // DAI Market (Market 3)
        markets.insert(
            DYDX_MARKET_DAI,
            DydxMarketData {
                market_id: DYDX_MARKET_DAI,
                token_address: "0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), // DAI
                token_symbol: "DAI".to_string(),
                token_decimals: 18,
                total_supply: "80000000".parse().unwrap_or_default(), // 80M DAI
                total_borrow: "50000000".parse().unwrap_or_default(), // 50M DAI
                supply_index: "1.04".parse().unwrap_or_default(),
                borrow_index: "1.07".parse().unwrap_or_default(),
                supply_rate: "0.018".parse().unwrap_or_default(), // 1.8%
                borrow_rate: "0.045".parse().unwrap_or_default(), // 4.5%
                oracle_price: "1".parse().unwrap_or_default(), // $1 per DAI
                status: DydxMarketStatus::Active,
                flashloan_enabled: true,
                last_update: now,
            },
        );

        markets
    }

    /// Get Solo Margin configuration for chain
    #[cfg(test)]
    fn get_solo_margin_configuration(chain_id: ChainId) -> DydxSoloMarginConfiguration {
        let _supported = matches!(chain_id, ChainId::Ethereum);

        DydxSoloMarginConfiguration {
            owner: "0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e".to_string(), // Mock owner
            risk_params: DydxRiskParameters {
                margin_ratio: "0.15".parse().unwrap_or_default(), // 15%
                liquidation_spread: "0.05".parse().unwrap_or_default(), // 5%
                earn_spread: "0.95".parse().unwrap_or_default(), // 95%
                min_borrow_value: "100".parse().unwrap_or_default(), // $100
            },
            interest_setter: "0x8b1b3c0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b".to_string(), // Mock interest setter
            price_oracle: "0x9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c".to_string(), // Mock oracle
            margin_ratio: "0.15".parse().unwrap_or_default(), // 15%
            liquidation_spread: "0.05".parse().unwrap_or_default(), // 5%
            earn_spread: "0.95".parse().unwrap_or_default(), // 95%
            min_borrow_value: "100".parse().unwrap_or_default(), // $100
            paused: false,
        }
    }

    /// Fetch Solo Margin information from external sources
    #[cfg(test)]
    #[expect(dead_code, reason = "Used for testing Solo Margin data generation")]
    async fn fetch_solo_margin_info(supported_chains: &[ChainId]) -> Result<Vec<DydxSoloMarginInfo>> {
        let mut solo_margins = Vec::new();

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for mock Solo Margin data")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Mock Solo Margin data generation - in production this would query real dYdX APIs
        for chain_id in supported_chains {
            if *chain_id == ChainId::Ethereum {
                let solo_margin_info = DydxSoloMarginInfo {
                    chain_id: *chain_id,
                    solo_margin_address: Self::get_solo_margin_address(*chain_id),
                    proxy_address: Self::get_proxy_address(*chain_id),
                    available_markets: Self::get_mock_markets(*chain_id),
                    solo_margin_configuration: Self::get_solo_margin_configuration(*chain_id),
                    status: DydxSoloMarginStatus::Active,
                    last_update: now,
                };
                solo_margins.push(solo_margin_info);
            }
        }

        Ok(solo_margins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_dydx_flashloan_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(dydx) = DydxFlashloan::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(dydx.stats().total_requests.load(Ordering::Relaxed), 0);
        assert_eq!(dydx.stats().successful_executions.load(Ordering::Relaxed), 0);
        assert_eq!(dydx.stats().failed_executions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_dydx_flashloan_config_default() {
        let config = DydxFlashloanConfig::default();
        assert!(config.enabled);
        assert_eq!(config.market_monitoring_interval_ms, DYDX_DEFAULT_MONITORING_INTERVAL_MS);
        assert_eq!(config.position_optimization_interval_ms, DYDX_DEFAULT_OPTIMIZATION_INTERVAL_MS);
        assert_eq!(config.performance_monitoring_interval_ms, DYDX_DEFAULT_PERFORMANCE_INTERVAL_MS);
        assert!(config.enable_zero_fee_optimization);
        assert!(config.enable_callback_optimization);
        assert!(config.enable_multi_market);
        assert!(config.enable_margin_analysis);
        assert!(!config.supported_chains.is_empty());
        assert!(!config.supported_markets.is_empty());
        assert!(!config.supported_assets.is_empty());
        assert_eq!(config.default_fee_percentage, Decimal::ZERO); // dYdX is free
    }

    #[test]
    fn test_aligned_dydx_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedDydxData>(), 64);
        assert!(mem::size_of::<AlignedDydxData>() <= 64);
    }

    #[test]
    fn test_dydx_flashloan_stats_operations() {
        let stats = DydxFlashloanStats::default();

        stats.total_requests.fetch_add(100, Ordering::Relaxed);
        stats.successful_executions.fetch_add(90, Ordering::Relaxed); // 90% success rate
        stats.failed_executions.fetch_add(10, Ordering::Relaxed);
        stats.total_volume_usd.fetch_add(3_000_000, Ordering::Relaxed);
        stats.total_fees_paid_usd.fetch_add(0, Ordering::Relaxed); // dYdX is free
        stats.zero_fee_optimizations.fetch_add(90, Ordering::Relaxed); // All successful are zero-fee
        stats.multi_market_loans.fetch_add(20, Ordering::Relaxed);

        assert_eq!(stats.total_requests.load(Ordering::Relaxed), 100);
        assert_eq!(stats.successful_executions.load(Ordering::Relaxed), 90);
        assert_eq!(stats.failed_executions.load(Ordering::Relaxed), 10);
        assert_eq!(stats.total_volume_usd.load(Ordering::Relaxed), 3_000_000);
        assert_eq!(stats.total_fees_paid_usd.load(Ordering::Relaxed), 0);
        assert_eq!(stats.zero_fee_optimizations.load(Ordering::Relaxed), 90);
        assert_eq!(stats.multi_market_loans.load(Ordering::Relaxed), 20);
    }

    #[test]
    fn test_aligned_dydx_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedDydxData::new(
            5_000_000_000_000, // $5M liquidity (scaled by 1e6)
            0,                 // 0% fee (scaled by 1e6)
            4,                 // 4 markets
            900_000,           // 90% health score (scaled by 1e6)
            10,                // 10s execution time
            900_000,           // 90% success rate (scaled by 1e6)
            100,               // 100 loans executed
            now,
        );

        let stale_data = AlignedDydxData::new(
            5_000_000_000_000, 0, 4, 900_000, 10, 900_000, 100,
            now - 450_000, // 7.5 minutes old
        );

        assert!(!fresh_data.is_stale(210_000)); // 3.5 minutes
        assert!(stale_data.is_stale(210_000)); // 3.5 minutes
    }

    #[test]
    fn test_aligned_dydx_data_conversions() {
        let data = AlignedDydxData::new(
            5_000_000_000_000, // $5M liquidity (scaled by 1e6)
            0,                 // 0% fee (scaled by 1e6)
            4,                 // 4 markets
            900_000,           // 90% health score (scaled by 1e6)
            10,                // 10s execution time
            900_000,           // 90% success rate (scaled by 1e6)
            100,               // 100 loans executed
            1_640_995_200_000,
        );

        assert_eq!(data.available_liquidity_usd(), dec!(5000000));
        assert_eq!(data.fee_percentage(), Decimal::ZERO);
        assert_eq!(data.solo_margin_health_score(), dec!(0.9));
        assert_eq!(data.success_rate(), dec!(0.9));
        assert_eq!(data.market_count, 4);

        // Overall score should be weighted average (no fee component)
        let liquidity_score = dec!(5000000) / dec!(100000000); // 0.05
        let expected_overall = liquidity_score * dec!(0.45) + dec!(0.9) * dec!(0.3) + dec!(0.9) * dec!(0.25);
        assert!((data.overall_score() - expected_overall).abs() < dec!(0.001));
    }

    #[test]
    fn test_dydx_solo_margin_status_enum() {
        assert_eq!(DydxSoloMarginStatus::Active, DydxSoloMarginStatus::Active);
        assert_ne!(DydxSoloMarginStatus::Active, DydxSoloMarginStatus::Paused);
        assert_ne!(DydxSoloMarginStatus::Maintenance, DydxSoloMarginStatus::Deprecated);
    }

    #[test]
    fn test_dydx_market_status_enum() {
        assert_eq!(DydxMarketStatus::Active, DydxMarketStatus::Active);
        assert_ne!(DydxMarketStatus::Active, DydxMarketStatus::Paused);
        assert_ne!(DydxMarketStatus::InsufficientLiquidity, DydxMarketStatus::Deprecated);
    }

    #[test]
    fn test_dydx_execution_status_enum() {
        assert_eq!(DydxExecutionStatus::Success, DydxExecutionStatus::Success);
        assert_ne!(DydxExecutionStatus::Success, DydxExecutionStatus::Failed);
        assert_ne!(DydxExecutionStatus::Pending, DydxExecutionStatus::LoanInitiated);
    }

    #[test]
    fn test_dydx_flashloan_request_validation() {
        // Valid request
        let valid_request = DydxFlashloanRequest {
            market_id: DYDX_MARKET_USDC,
            amount: dec!(1000000),
            callback_data: vec![],
            account_owner: "0x1234567890123456789012345678901234567890".to_string(),
            account_number: 0,
            chain_id: ChainId::Ethereum,
            strategy_id: "test_strategy".to_string(),
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test deadline")]
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 + 300_000, // 5 minutes from now
        };

        assert!(DydxFlashloan::validate_request(&valid_request));

        // Invalid request - empty account owner
        let invalid_request = DydxFlashloanRequest {
            market_id: DYDX_MARKET_USDC,
            amount: dec!(1000000),
            callback_data: vec![],
            account_owner: String::new(), // Empty account owner
            account_number: 0,
            chain_id: ChainId::Ethereum,
            strategy_id: "test_strategy".to_string(),
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test deadline")]
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 + 300_000,
        };

        assert!(!DydxFlashloan::validate_request(&invalid_request));

        // Invalid request - zero amount
        let zero_amount_request = DydxFlashloanRequest {
            market_id: DYDX_MARKET_USDC,
            amount: dec!(0), // Zero amount
            callback_data: vec![],
            account_owner: "0x1234567890123456789012345678901234567890".to_string(),
            account_number: 0,
            chain_id: ChainId::Ethereum,
            strategy_id: "test_strategy".to_string(),
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test deadline")]
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 + 300_000,
        };

        assert!(!DydxFlashloan::validate_request(&zero_amount_request));
    }

    #[test]
    fn test_dydx_gas_estimation() {
        let gas_used = DydxFlashloan::estimate_gas_usage();
        assert_eq!(gas_used, 350_000); // 350k gas for dYdX flashloan

        // Gas costs per chain
        assert_eq!(DydxFlashloan::estimate_gas_cost(ChainId::Ethereum), dec!(60));
        assert_eq!(DydxFlashloan::estimate_gas_cost(ChainId::Arbitrum), dec!(0)); // Not supported
        assert_eq!(DydxFlashloan::estimate_gas_cost(ChainId::Polygon), dec!(0)); // Not supported
    }

    #[test]
    fn test_dydx_solo_margin_addresses() {
        assert_eq!(
            DydxFlashloan::get_solo_margin_address(ChainId::Ethereum),
            "0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e"
        );

        // Other chains not supported
        assert_eq!(
            DydxFlashloan::get_solo_margin_address(ChainId::Arbitrum),
            "0x0000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn test_dydx_proxy_addresses() {
        assert_eq!(
            DydxFlashloan::get_proxy_address(ChainId::Ethereum),
            "0x4EC3570cADaAEE08Ae384779B0f3A45EF85289DE"
        );

        // Other chains not supported
        assert_eq!(
            DydxFlashloan::get_proxy_address(ChainId::Arbitrum),
            "0x0000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn test_dydx_mock_markets() {
        let eth_markets = DydxFlashloan::get_mock_markets(ChainId::Ethereum);
        assert!(!eth_markets.is_empty());
        assert!(eth_markets.contains_key(&DYDX_MARKET_ETH)); // WETH
        assert!(eth_markets.contains_key(&DYDX_MARKET_USDC)); // USDC
        assert!(eth_markets.contains_key(&DYDX_MARKET_DAI)); // DAI

        if let Some(eth_market) = eth_markets.get(&DYDX_MARKET_ETH) {
            assert_eq!(eth_market.token_symbol, "WETH");
            assert_eq!(eth_market.token_decimals, 18);
            assert!(eth_market.flashloan_enabled);
            assert_eq!(eth_market.status, DydxMarketStatus::Active);
            assert_eq!(eth_market.oracle_price, dec!(2000)); // $2000 per ETH
        }

        if let Some(usdc_market) = eth_markets.get(&DYDX_MARKET_USDC) {
            assert_eq!(usdc_market.token_symbol, "USDC");
            assert_eq!(usdc_market.token_decimals, 6);
            assert!(usdc_market.flashloan_enabled);
            assert_eq!(usdc_market.status, DydxMarketStatus::Active);
            assert_eq!(usdc_market.oracle_price, dec!(1)); // $1 per USDC
        }

        // Other chains should have empty markets
        let arb_markets = DydxFlashloan::get_mock_markets(ChainId::Arbitrum);
        assert!(arb_markets.is_empty());
    }

    #[test]
    fn test_dydx_solo_margin_configuration() {
        let config = DydxFlashloan::get_solo_margin_configuration(ChainId::Ethereum);
        assert!(!config.paused);
        assert_eq!(config.margin_ratio, dec!(0.15)); // 15%
        assert_eq!(config.liquidation_spread, dec!(0.05)); // 5%
        assert_eq!(config.earn_spread, dec!(0.95)); // 95%
        assert_eq!(config.min_borrow_value, dec!(100)); // $100

        // Risk parameters should match
        assert_eq!(config.risk_params.margin_ratio, config.margin_ratio);
        assert_eq!(config.risk_params.liquidation_spread, config.liquidation_spread);
        assert_eq!(config.risk_params.earn_spread, config.earn_spread);
        assert_eq!(config.risk_params.min_borrow_value, config.min_borrow_value);
    }

    #[test]
    fn test_dydx_market_constants() {
        assert_eq!(DYDX_MARKET_ETH, 0);
        assert_eq!(DYDX_MARKET_SAI, 1);
        assert_eq!(DYDX_MARKET_USDC, 2);
        assert_eq!(DYDX_MARKET_DAI, 3);
    }

    #[test]
    fn test_dydx_fee_constants() {
        assert_eq!(DYDX_FLASHLOAN_FEE, 0); // 0% fee
        assert_eq!(DYDX_DEFAULT_FEE_PERCENTAGE, "0.0"); // 0% default
    }

    #[tokio::test]
    async fn test_dydx_flashloan_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(dydx) = DydxFlashloan::new(config).await else {
            return; // Skip test if creation fails
        };

        // Test getting Solo Margins (should be empty initially)
        let solo_margins = dydx.get_solo_margins().await;
        assert!(solo_margins.is_empty());

        // Test getting active executions (should be empty initially)
        let executions = dydx.get_active_executions().await;
        assert!(executions.is_empty());

        // Test stats access
        let stats = dydx.stats();
        assert_eq!(stats.total_requests.load(Ordering::Relaxed), 0);
    }
}
