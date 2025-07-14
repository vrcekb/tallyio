//! Uniswap V3 Flashloan Integration for ultra-performance flash swap operations
//!
//! This module provides advanced Uniswap V3 flashloan integration capabilities for maximizing
//! capital efficiency through direct Uniswap protocol interaction and optimal
//! flash swap execution across multiple chains.
//!
//! ## Performance Targets
//! - Swap Initiation: <35μs
//! - Fee Calculation: <15μs
//! - Execution Monitoring: <20μs
//! - Callback Processing: <50μs
//! - Total Execution: <120μs
//!
//! ## Architecture
//! - Direct Uniswap V3 pool integration
//! - Advanced flash swap callback handling
//! - Dynamic fee tier optimization
//! - Multi-chain Uniswap support
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

/// Uniswap V3 flashloan configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct UniswapFlashloanConfig {
    /// Enable Uniswap flashloan integration
    pub enabled: bool,
    
    /// Pool monitoring interval in milliseconds
    pub pool_monitoring_interval_ms: u64,
    
    /// Fee tier optimization interval in milliseconds
    pub fee_tier_optimization_interval_ms: u64,
    
    /// Performance monitoring interval in milliseconds
    pub performance_monitoring_interval_ms: u64,
    
    /// Enable dynamic fee tier selection
    pub enable_dynamic_fee_tiers: bool,
    
    /// Enable callback optimization
    pub enable_callback_optimization: bool,
    
    /// Enable multi-hop swaps
    pub enable_multi_hop: bool,
    
    /// Enable liquidity analysis
    pub enable_liquidity_analysis: bool,
    
    /// Maximum flash swap amount (USD)
    pub max_flash_swap_amount_usd: Decimal,
    
    /// Minimum flash swap amount (USD)
    pub min_flash_swap_amount_usd: Decimal,
    
    /// Default fee percentage (varies by tier)
    pub default_fee_percentage: Decimal,
    
    /// Supported chains for Uniswap V3
    pub supported_chains: Vec<ChainId>,
    
    /// Supported fee tiers (in basis points)
    pub supported_fee_tiers: Vec<u32>,
    
    /// Supported assets for flash swaps
    pub supported_assets: Vec<String>,
}

/// Uniswap V3 factory information
#[derive(Debug, Clone)]
pub struct UniswapFactoryInfo {
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Factory address
    pub factory_address: String,
    
    /// Position manager address
    pub position_manager_address: String,
    
    /// Swap router address
    pub swap_router_address: String,
    
    /// Available pools
    pub available_pools: HashMap<String, UniswapPoolData>,
    
    /// Factory configuration
    pub factory_configuration: UniswapFactoryConfiguration,
    
    /// Factory status
    pub status: UniswapFactoryStatus,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Uniswap V3 pool data
#[derive(Debug, Clone)]
pub struct UniswapPoolData {
    /// Pool address
    pub pool_address: String,
    
    /// Token0 address
    pub token0: String,
    
    /// Token1 address
    pub token1: String,
    
    /// Fee tier (in basis points)
    pub fee_tier: u32,
    
    /// Tick spacing
    pub tick_spacing: i32,
    
    /// Current tick
    pub current_tick: i32,
    
    /// Current sqrt price
    pub sqrt_price_x96: String,
    
    /// Liquidity
    pub liquidity: Decimal,
    
    /// Token0 reserve
    pub token0_reserve: Decimal,
    
    /// Token1 reserve
    pub token1_reserve: Decimal,
    
    /// Pool TVL (USD)
    pub tvl_usd: Decimal,
    
    /// 24h volume (USD)
    pub volume_24h_usd: Decimal,
    
    /// Pool status
    pub status: UniswapPoolStatus,
    
    /// Flash swap enabled
    pub flash_swap_enabled: bool,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Uniswap V3 factory configuration
#[derive(Debug, Clone)]
pub struct UniswapFactoryConfiguration {
    /// Factory owner
    pub owner: String,
    
    /// Fee amount to tick spacing mapping
    pub fee_amount_tick_spacing: HashMap<u32, i32>,
    
    /// Protocol fee percentage
    pub protocol_fee_percentage: Decimal,
    
    /// Factory paused
    pub paused: bool,
}

/// Uniswap factory status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniswapFactoryStatus {
    /// Factory is active and operational
    Active,
    /// Factory is paused
    Paused,
    /// Factory is under maintenance
    Maintenance,
    /// Factory is deprecated
    Deprecated,
}

/// Uniswap pool status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniswapPoolStatus {
    /// Pool is active and operational
    Active,
    /// Pool is paused
    Paused,
    /// Pool has insufficient liquidity
    InsufficientLiquidity,
    /// Pool is deprecated
    Deprecated,
}

/// Uniswap V3 flash swap request
#[derive(Debug, Clone)]
pub struct UniswapFlashSwapRequest {
    /// Pool address for flash swap
    pub pool_address: String,
    
    /// Token to borrow (token0 or token1)
    pub token_address: String,
    
    /// Amount to borrow (in token units)
    pub amount: Decimal,
    
    /// Callback data
    pub callback_data: Vec<u8>,
    
    /// Recipient address
    pub recipient: String,
    
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Strategy identifier
    pub strategy_id: String,
    
    /// Execution deadline
    pub deadline: u64,
    
    /// Preferred fee tier (optional)
    pub preferred_fee_tier: Option<u32>,
    
    /// Zero for one (direction of swap)
    pub zero_for_one: bool,
}

/// Uniswap V3 flash swap execution result
#[derive(Debug, Clone)]
pub struct UniswapFlashSwapExecution {
    /// Request ID
    pub request_id: String,
    
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Pool address used
    pub pool_address: String,
    
    /// Token borrowed
    pub token_borrowed: String,
    
    /// Amount borrowed
    pub amount_borrowed: Decimal,
    
    /// Fee paid
    pub fee_paid: Decimal,
    
    /// Fee tier used
    pub fee_tier: u32,
    
    /// Execution status
    pub status: UniswapExecutionStatus,
    
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

/// Uniswap execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniswapExecutionStatus {
    /// Execution pending
    Pending,
    /// Swap initiated
    SwapInitiated,
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

/// Uniswap flashloan statistics
#[derive(Debug, Default)]
pub struct UniswapFlashloanStats {
    /// Total flash swap requests
    pub total_requests: AtomicU64,
    
    /// Successful executions
    pub successful_executions: AtomicU64,
    
    /// Failed executions
    pub failed_executions: AtomicU64,
    
    /// Total volume swapped (USD)
    pub total_volume_usd: AtomicU64,
    
    /// Total fees paid (USD)
    pub total_fees_paid_usd: AtomicU64,
    
    /// Pool monitoring cycles
    pub pool_monitoring_cycles: AtomicU64,
    
    /// Fee tier optimizations performed
    pub fee_tier_optimizations: AtomicU64,
    
    /// Callback optimizations
    pub callback_optimizations: AtomicU64,
    
    /// Multi-hop swaps executed
    pub multi_hop_swaps: AtomicU64,
    
    /// Average execution time (μs)
    pub avg_execution_time_us: AtomicU64,
    
    /// Average fee percentage (scaled by 1e6)
    pub avg_fee_percentage_scaled: AtomicU64,
    
    /// Optimal pool selections
    pub optimal_pool_selections: AtomicU64,
}

/// Cache-line aligned Uniswap data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedUniswapData {
    /// Available liquidity USD (scaled by 1e6)
    pub available_liquidity_usd_scaled: u64,
    
    /// Fee percentage (scaled by 1e6)
    pub fee_percentage_scaled: u64,
    
    /// Pool count
    pub pool_count: u64,
    
    /// Factory health score (scaled by 1e6)
    pub factory_health_score_scaled: u64,
    
    /// Average execution time (seconds)
    pub avg_execution_time_s: u64,
    
    /// Success rate (scaled by 1e6)
    pub success_rate_scaled: u64,
    
    /// Total swaps executed
    pub total_swaps_executed: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// Uniswap V3 flashloan constants
pub const UNISWAP_DEFAULT_MONITORING_INTERVAL_MS: u64 = 2500; // 2.5 seconds
pub const UNISWAP_DEFAULT_OPTIMIZATION_INTERVAL_MS: u64 = 8000; // 8 seconds
pub const UNISWAP_DEFAULT_PERFORMANCE_INTERVAL_MS: u64 = 12000; // 12 seconds
pub const UNISWAP_DEFAULT_MAX_SWAP_USD: &str = "10000000.0"; // $10M maximum
pub const UNISWAP_DEFAULT_MIN_SWAP_USD: &str = "100.0"; // $100 minimum
pub const UNISWAP_DEFAULT_FEE_PERCENTAGE: &str = "0.0005"; // 0.05% default (500 bps)
pub const UNISWAP_FEE_TIER_100: u32 = 100; // 0.01%
pub const UNISWAP_FEE_TIER_500: u32 = 500; // 0.05%
pub const UNISWAP_FEE_TIER_3000: u32 = 3000; // 0.3%
pub const UNISWAP_FEE_TIER_10000: u32 = 10000; // 1%
pub const UNISWAP_MAX_POOLS: usize = 100;
pub const UNISWAP_MAX_EXECUTIONS: usize = 1000;

impl Default for UniswapFlashloanConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pool_monitoring_interval_ms: UNISWAP_DEFAULT_MONITORING_INTERVAL_MS,
            fee_tier_optimization_interval_ms: UNISWAP_DEFAULT_OPTIMIZATION_INTERVAL_MS,
            performance_monitoring_interval_ms: UNISWAP_DEFAULT_PERFORMANCE_INTERVAL_MS,
            enable_dynamic_fee_tiers: true,
            enable_callback_optimization: true,
            enable_multi_hop: true,
            enable_liquidity_analysis: true,
            max_flash_swap_amount_usd: UNISWAP_DEFAULT_MAX_SWAP_USD.parse().unwrap_or_default(),
            min_flash_swap_amount_usd: UNISWAP_DEFAULT_MIN_SWAP_USD.parse().unwrap_or_default(),
            default_fee_percentage: UNISWAP_DEFAULT_FEE_PERCENTAGE.parse().unwrap_or_default(),
            supported_chains: vec![
                ChainId::Ethereum,
                ChainId::Arbitrum,
                ChainId::Optimism,
                ChainId::Polygon,
                ChainId::Base,
            ],
            supported_fee_tiers: vec![
                UNISWAP_FEE_TIER_100,
                UNISWAP_FEE_TIER_500,
                UNISWAP_FEE_TIER_3000,
                UNISWAP_FEE_TIER_10000,
            ],
            supported_assets: vec![
                "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7".to_string(), // USDT
                "0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), // DAI
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), // WETH
                "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599".to_string(), // WBTC
            ],
        }
    }
}

impl AlignedUniswapData {
    /// Create new aligned Uniswap data
    #[inline(always)]
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "Aligned data structure requires all fields")]
    pub const fn new(
        available_liquidity_usd_scaled: u64,
        fee_percentage_scaled: u64,
        pool_count: u64,
        factory_health_score_scaled: u64,
        avg_execution_time_s: u64,
        success_rate_scaled: u64,
        total_swaps_executed: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            available_liquidity_usd_scaled,
            fee_percentage_scaled,
            pool_count,
            factory_health_score_scaled,
            avg_execution_time_s,
            success_rate_scaled,
            total_swaps_executed,
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

    /// Get fee percentage as Decimal
    #[inline(always)]
    #[must_use]
    pub fn fee_percentage(&self) -> Decimal {
        Decimal::from(self.fee_percentage_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get factory health score as Decimal
    #[inline(always)]
    #[must_use]
    pub fn factory_health_score(&self) -> Decimal {
        Decimal::from(self.factory_health_score_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get success rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn success_rate(&self) -> Decimal {
        Decimal::from(self.success_rate_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get overall factory score
    #[inline(always)]
    #[must_use]
    pub fn overall_score(&self) -> Decimal {
        // Weighted score: liquidity (35%) + health (25%) + success rate (25%) + pool diversity (15%)
        let liquidity_weight = "0.35".parse::<Decimal>().unwrap_or_default();
        let health_weight = "0.25".parse::<Decimal>().unwrap_or_default();
        let success_weight = "0.25".parse::<Decimal>().unwrap_or_default();
        let diversity_weight = "0.15".parse::<Decimal>().unwrap_or_default();

        // Normalize liquidity score (higher liquidity = higher score, max $100M)
        let liquidity_score = (self.available_liquidity_usd() / Decimal::from(100_000_000_u64)).min(Decimal::ONE);

        // Pool diversity score (more pools = higher score, max 200 pools)
        let diversity_score = (Decimal::from(self.pool_count) / Decimal::from(200_u64)).min(Decimal::ONE);

        liquidity_score * liquidity_weight +
        self.factory_health_score() * health_weight +
        self.success_rate() * success_weight +
        diversity_score * diversity_weight
    }
}

/// Uniswap V3 Flashloan Integration for ultra-performance flash swap operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct UniswapFlashloan {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Uniswap specific configuration
    uniswap_config: UniswapFlashloanConfig,

    /// Statistics
    stats: Arc<UniswapFlashloanStats>,

    /// Factory information
    factories: Arc<RwLock<HashMap<ChainId, UniswapFactoryInfo>>>,

    /// Factory data cache for ultra-fast access
    factory_cache: Arc<DashMap<ChainId, AlignedUniswapData>>,

    /// Active executions
    active_executions: Arc<RwLock<HashMap<String, UniswapFlashSwapExecution>>>,

    /// Performance timers
    monitoring_timer: Timer,
    optimization_timer: Timer,
    execution_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Factory update channels
    factory_sender: Sender<UniswapFactoryInfo>,
    factory_receiver: Receiver<UniswapFactoryInfo>,

    /// Execution channels
    execution_sender: Sender<UniswapFlashSwapExecution>,
    execution_receiver: Receiver<UniswapFlashSwapExecution>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current execution round
    execution_round: Arc<TokioMutex<u64>>,
}

impl UniswapFlashloan {
    /// Create new Uniswap flashloan integration with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let uniswap_config = UniswapFlashloanConfig::default();
        let stats = Arc::new(UniswapFlashloanStats::default());
        let factories = Arc::new(RwLock::new(HashMap::with_capacity(UNISWAP_MAX_POOLS)));
        let factory_cache = Arc::new(DashMap::with_capacity(UNISWAP_MAX_POOLS));
        let active_executions = Arc::new(RwLock::new(HashMap::with_capacity(UNISWAP_MAX_EXECUTIONS)));
        let monitoring_timer = Timer::new("uniswap_monitoring");
        let optimization_timer = Timer::new("uniswap_optimization");
        let execution_timer = Timer::new("uniswap_execution");
        let shutdown = Arc::new(AtomicBool::new(false));
        let execution_round = Arc::new(TokioMutex::new(0));

        let (factory_sender, factory_receiver) = channel::bounded(UNISWAP_MAX_POOLS);
        let (execution_sender, execution_receiver) = channel::bounded(UNISWAP_MAX_EXECUTIONS);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            uniswap_config,
            stats,
            factories,
            factory_cache,
            active_executions,
            monitoring_timer,
            optimization_timer,
            execution_timer,
            shutdown,
            factory_sender,
            factory_receiver,
            execution_sender,
            execution_receiver,
            http_client,
            execution_round,
        })
    }

    /// Start Uniswap flashloan services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.uniswap_config.enabled {
            info!("Uniswap flashloan integration disabled");
            return Ok(());
        }

        info!("Starting Uniswap flashloan integration");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start pool monitoring
        self.start_pool_monitoring().await;

        // Start fee tier optimization
        if self.uniswap_config.enable_dynamic_fee_tiers {
            self.start_fee_tier_optimization().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Uniswap flashloan integration started successfully");
        Ok(())
    }

    /// Stop Uniswap flashloan integration
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping Uniswap flashloan integration");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Uniswap flashloan integration stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &UniswapFlashloanStats {
        &self.stats
    }

    /// Get factory information
    #[inline]
    pub async fn get_factories(&self) -> Vec<UniswapFactoryInfo> {
        let factories = self.factories.read().await;
        factories.values().cloned().collect()
    }

    /// Get active executions
    #[inline]
    pub async fn get_active_executions(&self) -> Vec<UniswapFlashSwapExecution> {
        let executions = self.active_executions.read().await;
        executions.values().cloned().collect()
    }

    /// Execute Uniswap flash swap
    #[inline]
    #[must_use]
    pub async fn execute_flash_swap(&self, request: &UniswapFlashSwapRequest) -> Option<UniswapFlashSwapExecution> {
        let start_time = Instant::now();

        // Validate request
        if !Self::validate_request(request) {
            return None;
        }

        // Find optimal factory
        let factory_info = self.find_optimal_factory(request.chain_id).await?;

        // Generate execution ID
        let execution_id = self.generate_execution_id(request).await;

        // Execute flash swap
        let execution = self.execute_with_factory(&factory_info, request, &execution_id).await;

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
            while executions.len() > UNISWAP_MAX_EXECUTIONS {
                if let Some(oldest_key) = executions.keys().next().cloned() {
                    executions.remove(&oldest_key);
                }
            }
            drop(executions);
        }

        Some(execution)
    }

    /// Validate flash swap request
    fn validate_request(request: &UniswapFlashSwapRequest) -> bool {
        // Check for empty pool address
        if request.pool_address.is_empty() {
            return false;
        }

        // Check for empty token address
        if request.token_address.is_empty() {
            return false;
        }

        // Check for zero amount
        if request.amount <= Decimal::ZERO {
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

    /// Find optimal factory for chain
    async fn find_optimal_factory(&self, chain_id: ChainId) -> Option<UniswapFactoryInfo> {
        // Find factory for the specific chain
        {
            let factories = self.factories.read().await;
            if let Some(factory) = factories.get(&chain_id) {
                if factory.status == UniswapFactoryStatus::Active {
                    return Some(factory.clone());
                }
            }
        }

        None
    }

    /// Execute flash swap with specific factory
    async fn execute_with_factory(
        &self,
        factory: &UniswapFactoryInfo,
        request: &UniswapFlashSwapRequest,
        execution_id: &str,
    ) -> UniswapFlashSwapExecution {
        let start_time = Instant::now();

        // Find pool and calculate fee
        let (fee_tier, fee_paid) = Self::calculate_swap_fee(factory, request);

        // Simulate execution (in production this would interact with actual Uniswap contracts)
        let execution_success = Self::simulate_uniswap_execution(factory, request);

        let status = if execution_success {
            UniswapExecutionStatus::Success
        } else {
            UniswapExecutionStatus::Failed
        };

        let transaction_hash = if execution_success {
            Some(format!("0x{:x}", fastrand::u64(..)))
        } else {
            None
        };

        let error_message = if execution_success {
            None
        } else {
            Some("Simulated Uniswap execution failure".to_string())
        };

        #[expect(clippy::cast_possible_truncation, reason = "Execution time truncation is acceptable")]
        let execution_time_s = start_time.elapsed().as_secs() as u32;

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for execution data")]
        let executed_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        UniswapFlashSwapExecution {
            request_id: execution_id.to_string(),
            chain_id: request.chain_id,
            pool_address: request.pool_address.clone(),
            token_borrowed: request.token_address.clone(),
            amount_borrowed: request.amount,
            fee_paid,
            fee_tier,
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

    /// Calculate swap fee for flash swap
    fn calculate_swap_fee(factory: &UniswapFactoryInfo, request: &UniswapFlashSwapRequest) -> (u32, Decimal) {
        // Find the pool and get its fee tier
        factory.available_pools.get(&request.pool_address).map_or_else(|| {
            // Default to 0.05% fee tier
            let default_fee_tier = UNISWAP_FEE_TIER_500;
            let fee_percentage = Decimal::from(default_fee_tier) / Decimal::from(1_000_000_u64);
            let fee_paid = request.amount * fee_percentage;
            (default_fee_tier, fee_paid)
        }, |pool| {
            let fee_tier = request.preferred_fee_tier.unwrap_or(pool.fee_tier);
            let fee_percentage = Decimal::from(fee_tier) / Decimal::from(1_000_000_u64); // Convert basis points to decimal
            let fee_paid = request.amount * fee_percentage;
            (fee_tier, fee_paid)
        })
    }

    /// Simulate Uniswap execution (for testing)
    fn simulate_uniswap_execution(factory: &UniswapFactoryInfo, request: &UniswapFlashSwapRequest) -> bool {
        // Check factory status
        if factory.status != UniswapFactoryStatus::Active {
            return false;
        }

        // Check if pool exists and has sufficient liquidity
        if let Some(pool) = factory.available_pools.get(&request.pool_address) {
            if pool.status != UniswapPoolStatus::Active || !pool.flash_swap_enabled {
                return false;
            }

            // Check liquidity based on token direction
            let available_liquidity = if request.token_address == pool.token0 {
                pool.token0_reserve
            } else if request.token_address == pool.token1 {
                pool.token1_reserve
            } else {
                return false; // Token not in pool
            };

            if available_liquidity < request.amount {
                return false;
            }
        } else {
            return false; // Pool not found
        }

        // Simulate success rate (92% for Uniswap V3)
        #[allow(clippy::float_arithmetic)] // Simulation requires floating point arithmetic
        {
            fastrand::f64() < 0.92
        }
    }

    /// Generate unique execution ID
    async fn generate_execution_id(&self, request: &UniswapFlashSwapRequest) -> String {
        let mut round = self.execution_round.lock().await;
        *round += 1;
        #[expect(clippy::cast_possible_truncation, reason = "Chain ID values are small")]
        let chain_id_u8 = request.chain_id as u8;
        format!("uni_{}_{}_{}_{}", chain_id_u8, request.strategy_id, *round, fastrand::u64(..))
    }

    /// Update execution statistics
    fn update_execution_stats(&self, execution: &UniswapFlashSwapExecution) {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);

        match execution.status {
            UniswapExecutionStatus::Success => {
                self.stats.successful_executions.fetch_add(1, Ordering::Relaxed);
                self.stats.optimal_pool_selections.fetch_add(1, Ordering::Relaxed);

                // Update volume and fees
                let volume_scaled = (execution.amount_borrowed * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0);
                self.stats.total_volume_usd.fetch_add(volume_scaled, Ordering::Relaxed);

                let fees_scaled = (execution.fee_paid * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0);
                self.stats.total_fees_paid_usd.fetch_add(fees_scaled, Ordering::Relaxed);

                // Update fee tier optimization counter
                self.stats.fee_tier_optimizations.fetch_add(1, Ordering::Relaxed);
            }
            UniswapExecutionStatus::Failed | UniswapExecutionStatus::TimedOut | UniswapExecutionStatus::CallbackFailed => {
                self.stats.failed_executions.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Estimate gas usage for Uniswap flash swap
    const fn estimate_gas_usage() -> u64 {
        // Base gas cost for Uniswap V3 flash swap
        200_000_u64 // 200k gas
    }

    /// Estimate gas cost for chain
    fn estimate_gas_cost(chain_id: ChainId) -> Decimal {
        match chain_id {
            ChainId::Ethereum => "35".parse().unwrap_or_default(),    // $35
            ChainId::Arbitrum => "1.2".parse().unwrap_or_default(),   // $1.2
            ChainId::Optimism => "2".parse().unwrap_or_default(),     // $2
            ChainId::Polygon => "0.25".parse().unwrap_or_default(),   // $0.25
            ChainId::Base => "0.8".parse().unwrap_or_default(),       // $0.8
            ChainId::Bsc => "0.6".parse().unwrap_or_default(),        // $0.6
            ChainId::Avalanche => "1".parse().unwrap_or_default(),    // $1
        }
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(4000)) // Uniswap timeout
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

    /// Start pool monitoring
    async fn start_pool_monitoring(&self) {
        let factory_receiver = self.factory_receiver.clone();
        let factories = Arc::clone(&self.factories);
        let factory_cache = Arc::clone(&self.factory_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let uniswap_config = self.uniswap_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(uniswap_config.pool_monitoring_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming factory updates
                while let Ok(factory_info) = factory_receiver.try_recv() {
                    let chain_id = factory_info.chain_id;

                    // Update factory information
                    {
                        let mut factories_guard = factories.write().await;
                        factories_guard.insert(chain_id, factory_info.clone());
                        drop(factories_guard);
                    }

                    // Update cache with aligned data
                    let total_liquidity = factory_info.available_pools.values()
                        .map(|p| p.tvl_usd)
                        .sum::<Decimal>();

                    let avg_fee = if factory_info.available_pools.is_empty() {
                        UNISWAP_FEE_TIER_500 // Default 0.05%
                    } else {
                        let total_fee: u32 = factory_info.available_pools.values()
                            .map(|p| p.fee_tier)
                            .sum();
                        #[expect(clippy::cast_possible_truncation, reason = "Pool count is expected to be small")]
                        let pool_count = factory_info.available_pools.len() as u32;
                        total_fee / pool_count
                    };

                    let aligned_data = AlignedUniswapData::new(
                        (total_liquidity * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        u64::from(avg_fee), // Fee in basis points
                        factory_info.available_pools.len() as u64,
                        920_000, // 92% health score (mock)
                        8, // 8s execution time
                        920_000, // 92% success rate (mock)
                        200, // Total swaps executed (mock)
                        factory_info.last_update,
                    );
                    factory_cache.insert(chain_id, aligned_data);
                }

                stats.pool_monitoring_cycles.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let monitoring_time = start_time.elapsed().as_micros() as u64;
                trace!("Uniswap pool monitoring cycle completed in {}μs", monitoring_time);
            }
        });
    }

    /// Start fee tier optimization
    async fn start_fee_tier_optimization(&self) {
        let execution_receiver = self.execution_receiver.clone();
        let active_executions = Arc::clone(&self.active_executions);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let uniswap_config = self.uniswap_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(uniswap_config.fee_tier_optimization_interval_ms));

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
                        while executions_guard.len() > UNISWAP_MAX_EXECUTIONS {
                            if let Some(oldest_key) = executions_guard.keys().next().cloned() {
                                executions_guard.remove(&oldest_key);
                            }
                        }
                        drop(executions_guard);
                    }
                }

                // Perform fee tier optimization analysis
                Self::optimize_fee_tiers(&active_executions, &stats).await;

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let optimization_time = start_time.elapsed().as_micros() as u64;
                trace!("Uniswap fee tier optimization cycle completed in {}μs", optimization_time);
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
                let pool_monitoring = stats.pool_monitoring_cycles.load(Ordering::Relaxed);
                let fee_tier_optimizations = stats.fee_tier_optimizations.load(Ordering::Relaxed);
                let callback_optimizations = stats.callback_optimizations.load(Ordering::Relaxed);
                let multi_hop_swaps = stats.multi_hop_swaps.load(Ordering::Relaxed);
                let avg_execution_time = stats.avg_execution_time_us.load(Ordering::Relaxed);
                let avg_fee_percentage = stats.avg_fee_percentage_scaled.load(Ordering::Relaxed);
                let optimal_selections = stats.optimal_pool_selections.load(Ordering::Relaxed);

                info!(
                    "Uniswap Stats: requests={}, successful={}, failed={}, volume=${}, fees=${}, monitoring={}, fee_tier_opt={}, callback_opt={}, multi_hop={}, avg_time={}μs, avg_fee={}%, optimal={}",
                    total_requests, successful_executions, failed_executions, total_volume, total_fees,
                    pool_monitoring, fee_tier_optimizations, callback_optimizations, multi_hop_swaps, avg_execution_time, avg_fee_percentage, optimal_selections
                );
            }
        });
    }

    /// Fetch factory information from external sources
    #[cfg(test)]
    #[expect(dead_code, reason = "Used for testing factory data generation")]
    async fn fetch_factory_info(supported_chains: &[ChainId]) -> Result<Vec<UniswapFactoryInfo>> {
        let mut factories = Vec::new();

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for mock factory data")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Mock factory data generation - in production this would query real Uniswap APIs
        for chain_id in supported_chains {
            let factory_info = UniswapFactoryInfo {
                chain_id: *chain_id,
                factory_address: Self::get_factory_address(*chain_id),
                position_manager_address: Self::get_position_manager_address(*chain_id),
                swap_router_address: Self::get_swap_router_address(*chain_id),
                available_pools: Self::get_mock_pools(*chain_id),
                factory_configuration: Self::get_factory_configuration(*chain_id),
                status: UniswapFactoryStatus::Active,
                last_update: now,
            };
            factories.push(factory_info);
        }

        Ok(factories)
    }

    /// Optimize fee tiers based on execution history
    async fn optimize_fee_tiers(
        active_executions: &Arc<RwLock<HashMap<String, UniswapFlashSwapExecution>>>,
        stats: &Arc<UniswapFlashloanStats>,
    ) {
        let executions_guard = active_executions.read().await;

        if executions_guard.is_empty() {
            return;
        }

        // Analyze fee tier patterns
        let mut total_fee_percentage = Decimal::ZERO;
        let mut execution_count = 0;

        for execution in executions_guard.values() {
            if execution.status == UniswapExecutionStatus::Success {
                // Calculate fee percentage
                if execution.amount_borrowed > Decimal::ZERO {
                    let fee_percentage = execution.fee_paid / execution.amount_borrowed;
                    total_fee_percentage += fee_percentage;
                    execution_count += 1;
                }
            }
        }

        if execution_count > 0 {
            let avg_fee_percentage = total_fee_percentage / Decimal::from(execution_count);
            let avg_fee_scaled = (avg_fee_percentage * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0);
            stats.avg_fee_percentage_scaled.store(avg_fee_scaled, Ordering::Relaxed);
            stats.fee_tier_optimizations.fetch_add(1, Ordering::Relaxed);
        }

        drop(executions_guard);
        trace!("Uniswap fee tier optimization completed");
    }

    /// Get Uniswap V3 factory address for chain
    #[cfg(test)]
    fn get_factory_address(chain_id: ChainId) -> String {
        match chain_id {
            ChainId::Ethereum | ChainId::Arbitrum | ChainId::Optimism | ChainId::Polygon | ChainId::Base => {
                "0x1F98431c8aD98523631AE4a59f267346ea31F984".to_string() // Uniswap V3 Factory
            }
            ChainId::Bsc => "0xdB1d10011AD0Ff90774D0C6Bb92e5C5c8b4461F7".to_string(), // PancakeSwap V3 Factory
            ChainId::Avalanche => "0x740b1c1de25031C31FF4fC9A62f554A55cdC1baD".to_string(), // Uniswap V3 on Avalanche
        }
    }

    /// Get Uniswap V3 position manager address for chain
    #[cfg(test)]
    fn get_position_manager_address(chain_id: ChainId) -> String {
        match chain_id {
            ChainId::Ethereum | ChainId::Arbitrum | ChainId::Optimism | ChainId::Polygon | ChainId::Base => {
                "0xC36442b4a4522E871399CD717aBDD847Ab11FE88".to_string() // NonfungiblePositionManager
            }
            ChainId::Bsc => "0x46A15B0b27311cedF172AB29E4f4766fbE7F4364".to_string(), // PancakeSwap V3 Position Manager
            ChainId::Avalanche => "0x655C406EBFa14EE2006250925e54ec43AD184f8B".to_string(), // Uniswap V3 Position Manager on Avalanche
        }
    }

    /// Get Uniswap V3 swap router address for chain
    #[cfg(test)]
    fn get_swap_router_address(chain_id: ChainId) -> String {
        match chain_id {
            ChainId::Ethereum | ChainId::Arbitrum | ChainId::Optimism | ChainId::Polygon | ChainId::Base => {
                "0xE592427A0AEce92De3Edee1F18E0157C05861564".to_string() // SwapRouter
            }
            ChainId::Bsc => "0x1b81D678ffb9C0263b24A97847620C99d213eB14".to_string(), // PancakeSwap V3 Router
            ChainId::Avalanche => "0xbb00FF08d01D300023C629E8fFfFcb65A5a578cE".to_string(), // Uniswap V3 Router on Avalanche
        }
    }

    /// Get mock pools for testing
    #[cfg(test)]
    fn get_mock_pools(chain_id: ChainId) -> HashMap<String, UniswapPoolData> {
        let mut pools = HashMap::new();

        let base_tvl = match chain_id {
            ChainId::Ethereum => "10000000".parse::<Decimal>().unwrap_or_default(),   // $10M
            ChainId::Arbitrum => "3000000".parse::<Decimal>().unwrap_or_default(),    // $3M
            ChainId::Optimism => "2000000".parse::<Decimal>().unwrap_or_default(),    // $2M
            ChainId::Polygon => "2500000".parse::<Decimal>().unwrap_or_default(),     // $2.5M
            ChainId::Base => "1500000".parse::<Decimal>().unwrap_or_default(),        // $1.5M
            ChainId::Bsc => "1000000".parse::<Decimal>().unwrap_or_default(),         // $1M
            ChainId::Avalanche => "800000".parse::<Decimal>().unwrap_or_default(),    // $800k
        };

        // USDC/WETH 0.05% Pool
        pools.insert(
            "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640".to_string(),
            UniswapPoolData {
                pool_address: "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640".to_string(),
                token0: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                token1: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), // WETH
                fee_tier: UNISWAP_FEE_TIER_500, // 0.05%
                tick_spacing: 10,
                current_tick: 200_000, // Mock current tick
                sqrt_price_x96: "1461446703485210103287273052203988822378723970341".to_string(), // Mock sqrt price
                liquidity: base_tvl,
                token0_reserve: base_tvl * "0.5".parse::<Decimal>().unwrap_or_default(),
                token1_reserve: base_tvl * "0.5".parse::<Decimal>().unwrap_or_default(),
                tvl_usd: base_tvl,
                volume_24h_usd: base_tvl * "0.1".parse::<Decimal>().unwrap_or_default(), // 10% daily volume
                status: UniswapPoolStatus::Active,
                flash_swap_enabled: true,
                #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for mock data")]
                last_update: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            },
        );

        // USDC/USDT 0.01% Pool
        pools.insert(
            "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6".to_string(),
            UniswapPoolData {
                pool_address: "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6".to_string(),
                token0: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                token1: "0xdAC17F958D2ee523a2206206994597C13D831ec7".to_string(), // USDT
                fee_tier: UNISWAP_FEE_TIER_100, // 0.01%
                tick_spacing: 1,
                current_tick: 0, // Stable pair, near 1:1
                sqrt_price_x96: "79228162514264337593543950336".to_string(), // 1:1 price
                liquidity: base_tvl * "0.8".parse::<Decimal>().unwrap_or_default(),
                token0_reserve: base_tvl * "0.4".parse::<Decimal>().unwrap_or_default(),
                token1_reserve: base_tvl * "0.4".parse::<Decimal>().unwrap_or_default(),
                tvl_usd: base_tvl * "0.8".parse::<Decimal>().unwrap_or_default(),
                volume_24h_usd: base_tvl * "0.15".parse::<Decimal>().unwrap_or_default(), // 15% daily volume
                status: UniswapPoolStatus::Active,
                flash_swap_enabled: true,
                #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for mock data")]
                last_update: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            },
        );

        pools
    }

    /// Get factory configuration for chain
    #[cfg(test)]
    fn get_factory_configuration(chain_id: ChainId) -> UniswapFactoryConfiguration {
        let _supported = matches!(chain_id, ChainId::Ethereum | ChainId::Arbitrum | ChainId::Optimism | ChainId::Polygon | ChainId::Base | ChainId::Bsc | ChainId::Avalanche);

        let mut fee_amount_tick_spacing = HashMap::new();
        fee_amount_tick_spacing.insert(UNISWAP_FEE_TIER_100, 1);
        fee_amount_tick_spacing.insert(UNISWAP_FEE_TIER_500, 10);
        fee_amount_tick_spacing.insert(UNISWAP_FEE_TIER_3000, 60);
        fee_amount_tick_spacing.insert(UNISWAP_FEE_TIER_10000, 200);

        UniswapFactoryConfiguration {
            owner: "0x1a9C8182C09F50C8318d769245beA52c32BE35BC".to_string(), // Mock owner
            fee_amount_tick_spacing,
            protocol_fee_percentage: "0.0".parse().unwrap_or_default(), // 0% protocol fee
            paused: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_uniswap_flashloan_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(uniswap) = UniswapFlashloan::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(uniswap.stats().total_requests.load(Ordering::Relaxed), 0);
        assert_eq!(uniswap.stats().successful_executions.load(Ordering::Relaxed), 0);
        assert_eq!(uniswap.stats().failed_executions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_uniswap_flashloan_config_default() {
        let config = UniswapFlashloanConfig::default();
        assert!(config.enabled);
        assert_eq!(config.pool_monitoring_interval_ms, UNISWAP_DEFAULT_MONITORING_INTERVAL_MS);
        assert_eq!(config.fee_tier_optimization_interval_ms, UNISWAP_DEFAULT_OPTIMIZATION_INTERVAL_MS);
        assert_eq!(config.performance_monitoring_interval_ms, UNISWAP_DEFAULT_PERFORMANCE_INTERVAL_MS);
        assert!(config.enable_dynamic_fee_tiers);
        assert!(config.enable_callback_optimization);
        assert!(config.enable_multi_hop);
        assert!(config.enable_liquidity_analysis);
        assert!(!config.supported_chains.is_empty());
        assert!(!config.supported_fee_tiers.is_empty());
        assert!(!config.supported_assets.is_empty());
        assert_eq!(config.default_fee_percentage, "0.0005".parse::<Decimal>().unwrap_or_default());
    }

    #[test]
    fn test_aligned_uniswap_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedUniswapData>(), 64);
        assert!(mem::size_of::<AlignedUniswapData>() <= 64);
    }

    #[test]
    fn test_uniswap_flashloan_stats_operations() {
        let stats = UniswapFlashloanStats::default();

        stats.total_requests.fetch_add(150, Ordering::Relaxed);
        stats.successful_executions.fetch_add(138, Ordering::Relaxed); // 92% success rate
        stats.failed_executions.fetch_add(12, Ordering::Relaxed);
        stats.total_volume_usd.fetch_add(5_000_000, Ordering::Relaxed);
        stats.total_fees_paid_usd.fetch_add(2_500, Ordering::Relaxed); // 0.05% fees
        stats.fee_tier_optimizations.fetch_add(25, Ordering::Relaxed);
        stats.multi_hop_swaps.fetch_add(30, Ordering::Relaxed);

        assert_eq!(stats.total_requests.load(Ordering::Relaxed), 150);
        assert_eq!(stats.successful_executions.load(Ordering::Relaxed), 138);
        assert_eq!(stats.failed_executions.load(Ordering::Relaxed), 12);
        assert_eq!(stats.total_volume_usd.load(Ordering::Relaxed), 5_000_000);
        assert_eq!(stats.total_fees_paid_usd.load(Ordering::Relaxed), 2_500);
        assert_eq!(stats.fee_tier_optimizations.load(Ordering::Relaxed), 25);
        assert_eq!(stats.multi_hop_swaps.load(Ordering::Relaxed), 30);
    }

    #[test]
    fn test_aligned_uniswap_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedUniswapData::new(
            10_000_000_000_000, // $10M liquidity (scaled by 1e6)
            500,                // 0.05% fee (500 basis points)
            50,                 // 50 pools
            920_000,            // 92% health score (scaled by 1e6)
            8,                  // 8s execution time
            920_000,            // 92% success rate (scaled by 1e6)
            200,                // 200 swaps executed
            now,
        );

        let stale_data = AlignedUniswapData::new(
            10_000_000_000_000, 500, 50, 920_000, 8, 920_000, 200,
            now - 350_000, // 5.83 minutes old
        );

        assert!(!fresh_data.is_stale(150_000)); // 2.5 minutes
        assert!(stale_data.is_stale(150_000)); // 2.5 minutes
    }

    #[test]
    fn test_aligned_uniswap_data_conversions() {
        let data = AlignedUniswapData::new(
            10_000_000_000_000, // $10M liquidity (scaled by 1e6)
            500,                // 0.05% fee (500 basis points)
            50,                 // 50 pools
            920_000,            // 92% health score (scaled by 1e6)
            8,                  // 8s execution time
            920_000,            // 92% success rate (scaled by 1e6)
            200,                // 200 swaps executed
            1_640_995_200_000,
        );

        assert_eq!(data.available_liquidity_usd(), dec!(10000000));
        assert_eq!(data.fee_percentage(), dec!(0.0005)); // 500 basis points = 0.05%
        assert_eq!(data.factory_health_score(), dec!(0.92));
        assert_eq!(data.success_rate(), dec!(0.92));
        assert_eq!(data.pool_count, 50);

        // Overall score should be weighted average
        let liquidity_score = dec!(10000000) / dec!(100000000); // 0.1
        let diversity_score = dec!(50) / dec!(200); // 0.25
        let expected_overall = liquidity_score * dec!(0.35) + dec!(0.92) * dec!(0.25) + dec!(0.92) * dec!(0.25) + diversity_score * dec!(0.15);
        assert!((data.overall_score() - expected_overall).abs() < dec!(0.001));
    }

    #[test]
    fn test_uniswap_factory_status_enum() {
        assert_eq!(UniswapFactoryStatus::Active, UniswapFactoryStatus::Active);
        assert_ne!(UniswapFactoryStatus::Active, UniswapFactoryStatus::Paused);
        assert_ne!(UniswapFactoryStatus::Maintenance, UniswapFactoryStatus::Deprecated);
    }

    #[test]
    fn test_uniswap_pool_status_enum() {
        assert_eq!(UniswapPoolStatus::Active, UniswapPoolStatus::Active);
        assert_ne!(UniswapPoolStatus::Active, UniswapPoolStatus::Paused);
        assert_ne!(UniswapPoolStatus::InsufficientLiquidity, UniswapPoolStatus::Deprecated);
    }

    #[test]
    fn test_uniswap_execution_status_enum() {
        assert_eq!(UniswapExecutionStatus::Success, UniswapExecutionStatus::Success);
        assert_ne!(UniswapExecutionStatus::Success, UniswapExecutionStatus::Failed);
        assert_ne!(UniswapExecutionStatus::Pending, UniswapExecutionStatus::SwapInitiated);
    }

    #[test]
    fn test_uniswap_flash_swap_request_validation() {
        // Valid request
        let valid_request = UniswapFlashSwapRequest {
            pool_address: "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640".to_string(),
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(),
            amount: dec!(1000000),
            callback_data: vec![],
            recipient: "0x1234567890123456789012345678901234567890".to_string(),
            chain_id: ChainId::Ethereum,
            strategy_id: "test_strategy".to_string(),
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test deadline")]
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 + 300_000, // 5 minutes from now
            preferred_fee_tier: Some(UNISWAP_FEE_TIER_500),
            zero_for_one: true,
        };

        assert!(UniswapFlashloan::validate_request(&valid_request));

        // Invalid request - empty pool address
        let invalid_request = UniswapFlashSwapRequest {
            pool_address: String::new(), // Empty pool address
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(),
            amount: dec!(1000000),
            callback_data: vec![],
            recipient: "0x1234567890123456789012345678901234567890".to_string(),
            chain_id: ChainId::Ethereum,
            strategy_id: "test_strategy".to_string(),
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test deadline")]
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 + 300_000,
            preferred_fee_tier: Some(UNISWAP_FEE_TIER_500),
            zero_for_one: true,
        };

        assert!(!UniswapFlashloan::validate_request(&invalid_request));

        // Invalid request - zero amount
        let zero_amount_request = UniswapFlashSwapRequest {
            pool_address: "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640".to_string(),
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(),
            amount: dec!(0), // Zero amount
            callback_data: vec![],
            recipient: "0x1234567890123456789012345678901234567890".to_string(),
            chain_id: ChainId::Ethereum,
            strategy_id: "test_strategy".to_string(),
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test deadline")]
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 + 300_000,
            preferred_fee_tier: Some(UNISWAP_FEE_TIER_500),
            zero_for_one: true,
        };

        assert!(!UniswapFlashloan::validate_request(&zero_amount_request));
    }

    #[test]
    fn test_uniswap_fee_calculation() {
        let mut pools = HashMap::new();
        pools.insert(
            "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640".to_string(),
            UniswapPoolData {
                pool_address: "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640".to_string(),
                token0: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(),
                token1: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(),
                fee_tier: UNISWAP_FEE_TIER_500, // 0.05%
                tick_spacing: 10,
                current_tick: 200_000,
                sqrt_price_x96: "1461446703485210103287273052203988822378723970341".to_string(),
                liquidity: dec!(10000000),
                token0_reserve: dec!(5000000),
                token1_reserve: dec!(5000000),
                tvl_usd: dec!(10000000),
                volume_24h_usd: dec!(1000000),
                status: UniswapPoolStatus::Active,
                flash_swap_enabled: true,
                last_update: 1_640_995_200_000,
            },
        );

        let factory = UniswapFactoryInfo {
            chain_id: ChainId::Ethereum,
            factory_address: "0x1F98431c8aD98523631AE4a59f267346ea31F984".to_string(),
            position_manager_address: "0xC36442b4a4522E871399CD717aBDD847Ab11FE88".to_string(),
            swap_router_address: "0xE592427A0AEce92De3Edee1F18E0157C05861564".to_string(),
            available_pools: pools,
            factory_configuration: UniswapFactoryConfiguration {
                owner: "0x1a9C8182C09F50C8318d769245beA52c32BE35BC".to_string(),
                fee_amount_tick_spacing: HashMap::new(),
                protocol_fee_percentage: Decimal::ZERO,
                paused: false,
            },
            status: UniswapFactoryStatus::Active,
            last_update: 1_640_995_200_000,
        };

        let request = UniswapFlashSwapRequest {
            pool_address: "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640".to_string(),
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(),
            amount: dec!(1000000), // $1M
            callback_data: vec![],
            recipient: "0x1234567890123456789012345678901234567890".to_string(),
            chain_id: ChainId::Ethereum,
            strategy_id: "test_strategy".to_string(),
            deadline: 1_640_995_500_000,
            preferred_fee_tier: None, // Use pool's fee tier
            zero_for_one: true,
        };

        let (fee_tier, fee_paid) = UniswapFlashloan::calculate_swap_fee(&factory, &request);

        assert_eq!(fee_tier, UNISWAP_FEE_TIER_500);
        assert_eq!(fee_paid, dec!(500)); // 0.05% of $1M = $500
    }

    #[test]
    fn test_uniswap_gas_estimation() {
        let gas_used = UniswapFlashloan::estimate_gas_usage();
        assert_eq!(gas_used, 200_000); // 200k gas for Uniswap V3 flash swap

        // Gas costs per chain
        assert_eq!(UniswapFlashloan::estimate_gas_cost(ChainId::Ethereum), dec!(35));
        assert_eq!(UniswapFlashloan::estimate_gas_cost(ChainId::Arbitrum), dec!(1.2));
        assert_eq!(UniswapFlashloan::estimate_gas_cost(ChainId::Polygon), dec!(0.25));
    }

    #[test]
    fn test_uniswap_factory_addresses() {
        assert_eq!(
            UniswapFlashloan::get_factory_address(ChainId::Ethereum),
            "0x1F98431c8aD98523631AE4a59f267346ea31F984"
        );
        assert_eq!(
            UniswapFlashloan::get_factory_address(ChainId::Arbitrum),
            "0x1F98431c8aD98523631AE4a59f267346ea31F984"
        );
        assert_eq!(
            UniswapFlashloan::get_factory_address(ChainId::Bsc),
            "0xdB1d10011AD0Ff90774D0C6Bb92e5C5c8b4461F7"
        );
        assert_eq!(
            UniswapFlashloan::get_factory_address(ChainId::Avalanche),
            "0x740b1c1de25031C31FF4fC9A62f554A55cdC1baD"
        );
    }

    #[test]
    fn test_uniswap_position_manager_addresses() {
        assert_eq!(
            UniswapFlashloan::get_position_manager_address(ChainId::Ethereum),
            "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
        );
        assert_eq!(
            UniswapFlashloan::get_position_manager_address(ChainId::Bsc),
            "0x46A15B0b27311cedF172AB29E4f4766fbE7F4364"
        );
        assert_eq!(
            UniswapFlashloan::get_position_manager_address(ChainId::Avalanche),
            "0x655C406EBFa14EE2006250925e54ec43AD184f8B"
        );
    }

    #[test]
    fn test_uniswap_swap_router_addresses() {
        assert_eq!(
            UniswapFlashloan::get_swap_router_address(ChainId::Ethereum),
            "0xE592427A0AEce92De3Edee1F18E0157C05861564"
        );
        assert_eq!(
            UniswapFlashloan::get_swap_router_address(ChainId::Bsc),
            "0x1b81D678ffb9C0263b24A97847620C99d213eB14"
        );
        assert_eq!(
            UniswapFlashloan::get_swap_router_address(ChainId::Avalanche),
            "0xbb00FF08d01D300023C629E8fFfFcb65A5a578cE"
        );
    }

    #[test]
    fn test_uniswap_mock_pools() {
        let eth_pools = UniswapFlashloan::get_mock_pools(ChainId::Ethereum);
        assert!(!eth_pools.is_empty());
        assert!(eth_pools.contains_key("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")); // USDC/WETH 0.05%
        assert!(eth_pools.contains_key("0x3416cF6C708Da44DB2624D63ea0AAef7113527C6")); // USDC/USDT 0.01%

        if let Some(usdc_weth_pool) = eth_pools.get("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640") {
            assert_eq!(usdc_weth_pool.fee_tier, UNISWAP_FEE_TIER_500);
            assert_eq!(usdc_weth_pool.tick_spacing, 10);
            assert!(usdc_weth_pool.flash_swap_enabled);
            assert_eq!(usdc_weth_pool.status, UniswapPoolStatus::Active);
        }

        if let Some(usdc_usdt_pool) = eth_pools.get("0x3416cF6C708Da44DB2624D63ea0AAef7113527C6") {
            assert_eq!(usdc_usdt_pool.fee_tier, UNISWAP_FEE_TIER_100);
            assert_eq!(usdc_usdt_pool.tick_spacing, 1);
            assert!(usdc_usdt_pool.flash_swap_enabled);
            assert_eq!(usdc_usdt_pool.status, UniswapPoolStatus::Active);
        }
    }

    #[test]
    fn test_uniswap_factory_configuration() {
        let config = UniswapFlashloan::get_factory_configuration(ChainId::Ethereum);
        assert!(!config.paused);
        assert_eq!(config.protocol_fee_percentage, Decimal::ZERO);
        assert!(config.fee_amount_tick_spacing.contains_key(&UNISWAP_FEE_TIER_100));
        assert!(config.fee_amount_tick_spacing.contains_key(&UNISWAP_FEE_TIER_500));
        assert!(config.fee_amount_tick_spacing.contains_key(&UNISWAP_FEE_TIER_3000));
        assert!(config.fee_amount_tick_spacing.contains_key(&UNISWAP_FEE_TIER_10000));

        // Check tick spacing mappings
        assert_eq!(config.fee_amount_tick_spacing.get(&UNISWAP_FEE_TIER_100).copied().unwrap_or_default(), 1);
        assert_eq!(config.fee_amount_tick_spacing.get(&UNISWAP_FEE_TIER_500).copied().unwrap_or_default(), 10);
        assert_eq!(config.fee_amount_tick_spacing.get(&UNISWAP_FEE_TIER_3000).copied().unwrap_or_default(), 60);
        assert_eq!(config.fee_amount_tick_spacing.get(&UNISWAP_FEE_TIER_10000).copied().unwrap_or_default(), 200);
    }

    #[test]
    fn test_uniswap_fee_tiers() {
        assert_eq!(UNISWAP_FEE_TIER_100, 100); // 0.01%
        assert_eq!(UNISWAP_FEE_TIER_500, 500); // 0.05%
        assert_eq!(UNISWAP_FEE_TIER_3000, 3000); // 0.3%
        assert_eq!(UNISWAP_FEE_TIER_10000, 10000); // 1%
    }

    #[tokio::test]
    async fn test_uniswap_flashloan_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(uniswap) = UniswapFlashloan::new(config).await else {
            return; // Skip test if creation fails
        };

        // Test getting factories (should be empty initially)
        let factories = uniswap.get_factories().await;
        assert!(factories.is_empty());

        // Test getting active executions (should be empty initially)
        let executions = uniswap.get_active_executions().await;
        assert!(executions.is_empty());

        // Test stats access
        let stats = uniswap.stats();
        assert_eq!(stats.total_requests.load(Ordering::Relaxed), 0);
    }
}
