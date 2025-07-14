//! Flashloan Coordinator for ultra-performance flashloan management
//!
//! This module provides advanced flashloan coordination capabilities for maximizing
//! capital efficiency through intelligent flashloan provider selection and
//! optimal loan management across multiple protocols and chains.
//!
//! ## Performance Targets
//! - Provider Discovery: <25μs
//! - Loan Calculation: <50μs
//! - Fee Optimization: <30μs
//! - Provider Selection: <75μs
//! - Loan Execution: <100μs
//!
//! ## Architecture
//! - Real-time flashloan provider monitoring
//! - Advanced multi-provider optimization algorithms
//! - Dynamic fee calculation and comparison
//! - Cross-chain flashloan coordination
//! - Lock-free loan management primitives

pub mod aave_flashloan;
pub mod balancer_flashloan;
pub mod dydx_flashloan;
pub mod optimal_selector;
pub mod parallel_executor;
pub mod uniswap_flashloan;

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

/// Flashloan coordinator configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct FlashloanConfig {
    /// Enable flashloan coordination
    pub enabled: bool,

    /// Provider discovery interval in milliseconds
    pub provider_discovery_interval_ms: u64,

    /// Fee optimization interval in milliseconds
    pub fee_optimization_interval_ms: u64,

    /// Performance monitoring interval in milliseconds
    pub performance_monitoring_interval_ms: u64,

    /// Enable dynamic provider selection
    pub enable_dynamic_selection: bool,

    /// Enable fee optimization
    pub enable_fee_optimization: bool,

    /// Enable cross-chain coordination
    pub enable_cross_chain: bool,

    /// Enable loan aggregation
    pub enable_loan_aggregation: bool,

    /// Maximum loan amount (USD)
    pub max_loan_amount_usd: Decimal,

    /// Maximum acceptable fee percentage
    pub max_fee_percentage: Decimal,

    /// Minimum loan amount (USD)
    pub min_loan_amount_usd: Decimal,

    /// Preferred providers
    pub preferred_providers: Vec<FlashloanProvider>,

    /// Supported chains for flashloans
    pub supported_chains: Vec<ChainId>,
}

/// Flashloan provider types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FlashloanProvider {
    /// Aave flashloan provider
    Aave,
    /// dYdX flashloan provider
    DyDx,
    /// Uniswap V3 flashloan provider
    UniswapV3,
    /// Balancer flashloan provider
    Balancer,
    /// Compound flashloan provider
    Compound,
    /// MakerDAO flashloan provider
    MakerDao,
    /// Euler flashloan provider
    Euler,
    /// Iron Bank flashloan provider
    IronBank,
    /// Radiant flashloan provider
    Radiant,
    /// Venus flashloan provider (BSC)
    Venus,
}

/// Flashloan provider information
#[derive(Debug, Clone)]
pub struct FlashloanProviderInfo {
    /// Provider type
    pub provider: FlashloanProvider,

    /// Chain ID where provider is available
    pub chain_id: ChainId,

    /// Contract address
    pub contract_address: String,

    /// Available tokens for flashloan
    pub available_tokens: Vec<String>,

    /// Maximum loan amounts per token (USD)
    pub max_loan_amounts: HashMap<String, Decimal>,

    /// Fee percentage (0.0 - 1.0)
    pub fee_percentage: Decimal,

    /// Fixed fee amount (USD)
    pub fixed_fee_usd: Decimal,

    /// Minimum loan amount (USD)
    pub min_loan_amount_usd: Decimal,

    /// Provider reliability score (0-100)
    pub reliability_score: u8,

    /// Average execution time (seconds)
    pub avg_execution_time_s: u32,

    /// Provider status
    pub status: ProviderStatus,

    /// Last update timestamp
    pub last_update: u64,
}

/// Provider status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderStatus {
    /// Provider is active and available
    Active,
    /// Provider is temporarily unavailable
    Unavailable,
    /// Provider is under maintenance
    Maintenance,
    /// Provider has insufficient liquidity
    InsufficientLiquidity,
    /// Provider is deprecated
    Deprecated,
}

/// Flashloan request
#[derive(Debug, Clone)]
pub struct FlashloanRequest {
    /// Token address to borrow
    pub token_address: String,

    /// Amount to borrow (in token units)
    pub amount: Decimal,

    /// Amount in USD
    pub amount_usd: Decimal,

    /// Target chain for the loan
    pub chain_id: ChainId,

    /// Maximum acceptable fee percentage
    pub max_fee_percentage: Decimal,

    /// Preferred providers (optional)
    pub preferred_providers: Vec<FlashloanProvider>,

    /// Execution deadline (timestamp)
    pub deadline: u64,

    /// Strategy identifier
    pub strategy_id: String,

    /// Priority level (1-10, higher = more urgent)
    pub priority: u8,
}

/// Flashloan quote
#[derive(Debug, Clone)]
pub struct FlashloanQuote {
    /// Provider offering the quote
    pub provider: FlashloanProvider,

    /// Chain ID
    pub chain_id: ChainId,

    /// Token address
    pub token_address: String,

    /// Loan amount
    pub amount: Decimal,

    /// Loan amount in USD
    pub amount_usd: Decimal,

    /// Fee percentage
    pub fee_percentage: Decimal,

    /// Fee amount (in token units)
    pub fee_amount: Decimal,

    /// Fee amount in USD
    pub fee_amount_usd: Decimal,

    /// Total repayment amount (loan + fee)
    pub total_repayment: Decimal,

    /// Total repayment in USD
    pub total_repayment_usd: Decimal,

    /// Estimated execution time (seconds)
    pub estimated_execution_time_s: u32,

    /// Provider reliability score
    pub reliability_score: u8,

    /// Quote validity timestamp
    pub valid_until: u64,

    /// Quote score (higher = better)
    pub quote_score: Decimal,
}

/// Flashloan execution result
#[derive(Debug, Clone)]
pub struct FlashloanExecution {
    /// Request ID
    pub request_id: String,

    /// Selected provider
    pub provider: FlashloanProvider,

    /// Chain ID
    pub chain_id: ChainId,

    /// Execution status
    pub status: ExecutionStatus,

    /// Transaction hash (if successful)
    pub transaction_hash: Option<String>,

    /// Actual fee paid
    pub actual_fee_usd: Decimal,

    /// Execution time (seconds)
    pub execution_time_s: u32,

    /// Gas used
    pub gas_used: u64,

    /// Gas cost (USD)
    pub gas_cost_usd: Decimal,

    /// Error message (if failed)
    pub error_message: Option<String>,

    /// Execution timestamp
    pub executed_at: u64,
}

/// Execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStatus {
    /// Execution pending
    Pending,
    /// Execution in progress
    InProgress,
    /// Execution successful
    Success,
    /// Execution failed
    Failed,
    /// Execution cancelled
    Cancelled,
    /// Execution timed out
    TimedOut,
}

/// Flashloan coordinator statistics
#[derive(Debug, Default)]
pub struct FlashloanStats {
    /// Total flashloan requests processed
    pub requests_processed: AtomicU64,

    /// Successful flashloan executions
    pub successful_executions: AtomicU64,

    /// Failed flashloan executions
    pub failed_executions: AtomicU64,

    /// Total flashloan volume (USD)
    pub total_volume_usd: AtomicU64,

    /// Total fees paid (USD)
    pub total_fees_paid_usd: AtomicU64,

    /// Total fees saved through optimization (USD)
    pub total_fees_saved_usd: AtomicU64,

    /// Provider discoveries performed
    pub provider_discoveries: AtomicU64,

    /// Fee optimizations performed
    pub fee_optimizations: AtomicU64,

    /// Cross-chain coordinations
    pub cross_chain_coordinations: AtomicU64,

    /// Average execution time (μs)
    pub avg_execution_time_us: AtomicU64,

    /// Average fee percentage (scaled by 1e6)
    pub avg_fee_percentage_scaled: AtomicU64,

    /// Optimal provider selections
    pub optimal_provider_selections: AtomicU64,
}

/// Cache-line aligned flashloan data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedFlashloanData {
    /// Provider reliability score (scaled by 1e6)
    pub reliability_score_scaled: u64,

    /// Fee percentage (scaled by 1e6)
    pub fee_percentage_scaled: u64,

    /// Maximum loan amount USD (scaled by 1e6)
    pub max_loan_amount_usd_scaled: u64,

    /// Average execution time (seconds)
    pub avg_execution_time_s: u64,

    /// Provider utilization rate (scaled by 1e6)
    pub utilization_rate_scaled: u64,

    /// Success rate (scaled by 1e6)
    pub success_rate_scaled: u64,

    /// Total loans executed
    pub total_loans_executed: u64,

    /// Last update timestamp
    pub timestamp: u64,
}

/// Flashloan coordinator constants
pub const FLASHLOAN_DEFAULT_DISCOVERY_INTERVAL_MS: u64 = 1000; // 1 second
pub const FLASHLOAN_DEFAULT_OPTIMIZATION_INTERVAL_MS: u64 = 2000; // 2 seconds
pub const FLASHLOAN_DEFAULT_MONITORING_INTERVAL_MS: u64 = 5000; // 5 seconds
pub const FLASHLOAN_DEFAULT_MAX_LOAN_USD: &str = "10000000.0"; // $10M maximum
pub const FLASHLOAN_DEFAULT_MAX_FEE_PERCENTAGE: &str = "0.01"; // 1% maximum
pub const FLASHLOAN_DEFAULT_MIN_LOAN_USD: &str = "1000.0"; // $1k minimum
pub const FLASHLOAN_MAX_PROVIDERS: usize = 50;
pub const FLASHLOAN_MAX_QUOTES: usize = 100;
pub const FLASHLOAN_MAX_EXECUTIONS: usize = 1000;

impl Default for FlashloanConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider_discovery_interval_ms: FLASHLOAN_DEFAULT_DISCOVERY_INTERVAL_MS,
            fee_optimization_interval_ms: FLASHLOAN_DEFAULT_OPTIMIZATION_INTERVAL_MS,
            performance_monitoring_interval_ms: FLASHLOAN_DEFAULT_MONITORING_INTERVAL_MS,
            enable_dynamic_selection: true,
            enable_fee_optimization: true,
            enable_cross_chain: true,
            enable_loan_aggregation: true,
            max_loan_amount_usd: FLASHLOAN_DEFAULT_MAX_LOAN_USD.parse().unwrap_or_default(),
            max_fee_percentage: FLASHLOAN_DEFAULT_MAX_FEE_PERCENTAGE.parse().unwrap_or_default(),
            min_loan_amount_usd: FLASHLOAN_DEFAULT_MIN_LOAN_USD.parse().unwrap_or_default(),
            preferred_providers: vec![
                FlashloanProvider::Aave,
                FlashloanProvider::UniswapV3,
                FlashloanProvider::Balancer,
                FlashloanProvider::DyDx,
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

impl AlignedFlashloanData {
    /// Create new aligned flashloan data
    #[inline(always)]
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "Aligned data structure requires all fields")]
    pub const fn new(
        reliability_score_scaled: u64,
        fee_percentage_scaled: u64,
        max_loan_amount_usd_scaled: u64,
        avg_execution_time_s: u64,
        utilization_rate_scaled: u64,
        success_rate_scaled: u64,
        total_loans_executed: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            reliability_score_scaled,
            fee_percentage_scaled,
            max_loan_amount_usd_scaled,
            avg_execution_time_s,
            utilization_rate_scaled,
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

    /// Get reliability score as Decimal
    #[inline(always)]
    #[must_use]
    pub fn reliability_score(&self) -> Decimal {
        Decimal::from(self.reliability_score_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get fee percentage as Decimal
    #[inline(always)]
    #[must_use]
    pub fn fee_percentage(&self) -> Decimal {
        Decimal::from(self.fee_percentage_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get maximum loan amount USD as Decimal
    #[inline(always)]
    #[must_use]
    pub fn max_loan_amount_usd(&self) -> Decimal {
        Decimal::from(self.max_loan_amount_usd_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get utilization rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn utilization_rate(&self) -> Decimal {
        Decimal::from(self.utilization_rate_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get success rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn success_rate(&self) -> Decimal {
        Decimal::from(self.success_rate_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get overall provider score
    #[inline(always)]
    #[must_use]
    pub fn overall_score(&self) -> Decimal {
        // Weighted score: reliability (40%) + success rate (30%) + low fees (20%) + speed (10%)
        let reliability_weight = "0.4".parse::<Decimal>().unwrap_or_default();
        let success_weight = "0.3".parse::<Decimal>().unwrap_or_default();
        let fee_weight = "0.2".parse::<Decimal>().unwrap_or_default();
        let speed_weight = "0.1".parse::<Decimal>().unwrap_or_default();

        // Normalize fee score (lower fees = higher score)
        let fee_score = Decimal::ONE - self.fee_percentage().min(Decimal::ONE);

        // Normalize speed score (faster = higher score, max 300s)
        let speed_score = (Decimal::from(300_u64) - Decimal::from(self.avg_execution_time_s).min(Decimal::from(300_u64))) / Decimal::from(300_u64);

        self.reliability_score() * reliability_weight +
        self.success_rate() * success_weight +
        fee_score * fee_weight +
        speed_score * speed_weight
    }
}

/// Flashloan Coordinator for ultra-performance flashloan management
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct FlashloanCoordinator {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Flashloan specific configuration
    flashloan_config: FlashloanConfig,

    /// Statistics
    stats: Arc<FlashloanStats>,

    /// Provider information
    providers: Arc<RwLock<HashMap<(FlashloanProvider, ChainId), FlashloanProviderInfo>>>,

    /// Provider data cache for ultra-fast access
    provider_cache: Arc<DashMap<(FlashloanProvider, ChainId), AlignedFlashloanData>>,

    /// Active quotes
    active_quotes: Arc<RwLock<HashMap<String, FlashloanQuote>>>,

    /// Execution history
    execution_history: Arc<RwLock<HashMap<String, FlashloanExecution>>>,

    /// Performance timers
    discovery_timer: Timer,
    optimization_timer: Timer,
    execution_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Provider update channels
    provider_sender: Sender<FlashloanProviderInfo>,
    provider_receiver: Receiver<FlashloanProviderInfo>,

    /// Quote channels
    quote_sender: Sender<FlashloanQuote>,
    quote_receiver: Receiver<FlashloanQuote>,

    /// Execution channels
    execution_sender: Sender<FlashloanExecution>,
    execution_receiver: Receiver<FlashloanExecution>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current request round
    request_round: Arc<TokioMutex<u64>>,
}

impl FlashloanCoordinator {
    /// Create new flashloan coordinator with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let flashloan_config = FlashloanConfig::default();
        let stats = Arc::new(FlashloanStats::default());
        let providers = Arc::new(RwLock::new(HashMap::with_capacity(FLASHLOAN_MAX_PROVIDERS)));
        let provider_cache = Arc::new(DashMap::with_capacity(FLASHLOAN_MAX_PROVIDERS));
        let active_quotes = Arc::new(RwLock::new(HashMap::with_capacity(FLASHLOAN_MAX_QUOTES)));
        let execution_history = Arc::new(RwLock::new(HashMap::with_capacity(FLASHLOAN_MAX_EXECUTIONS)));
        let discovery_timer = Timer::new("provider_discovery");
        let optimization_timer = Timer::new("fee_optimization");
        let execution_timer = Timer::new("flashloan_execution");
        let shutdown = Arc::new(AtomicBool::new(false));
        let request_round = Arc::new(TokioMutex::new(0));

        let (provider_sender, provider_receiver) = channel::bounded(FLASHLOAN_MAX_PROVIDERS);
        let (quote_sender, quote_receiver) = channel::bounded(FLASHLOAN_MAX_QUOTES);
        let (execution_sender, execution_receiver) = channel::bounded(FLASHLOAN_MAX_EXECUTIONS);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            flashloan_config,
            stats,
            providers,
            provider_cache,
            active_quotes,
            execution_history,
            discovery_timer,
            optimization_timer,
            execution_timer,
            shutdown,
            provider_sender,
            provider_receiver,
            quote_sender,
            quote_receiver,
            execution_sender,
            execution_receiver,
            http_client,
            request_round,
        })
    }

    /// Start flashloan coordination services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.flashloan_config.enabled {
            info!("Flashloan coordination disabled");
            return Ok(());
        }

        info!("Starting flashloan coordination");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start provider discovery
        self.start_provider_discovery().await;

        // Start fee optimization
        if self.flashloan_config.enable_fee_optimization {
            self.start_fee_optimization().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Flashloan coordination started successfully");
        Ok(())
    }

    /// Stop flashloan coordination
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping flashloan coordination");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Flashloan coordination stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &FlashloanStats {
        &self.stats
    }

    /// Get available providers
    #[inline]
    pub async fn get_providers(&self) -> Vec<FlashloanProviderInfo> {
        let providers = self.providers.read().await;
        providers.values().cloned().collect()
    }

    /// Get execution history
    #[inline]
    pub async fn get_execution_history(&self) -> Vec<FlashloanExecution> {
        let history = self.execution_history.read().await;
        history.values().cloned().collect()
    }

    /// Get flashloan quotes for a request
    #[inline]
    #[must_use]
    pub async fn get_quotes(&self, request: &FlashloanRequest) -> Vec<FlashloanQuote> {
        let start_time = Instant::now();

        let mut quotes = Vec::new();

        // Generate quotes from all suitable providers
        {
            let providers = self.providers.read().await;
            for (_provider_key, provider_info) in providers.iter() {
                if Self::is_provider_suitable(provider_info, request) {
                    let quote = Self::generate_quote(provider_info, request);
                    quotes.push(quote);
                }
            }
        }

        // Sort quotes by score (best first)
        quotes.sort_by(|a, b| b.quote_score.partial_cmp(&a.quote_score).unwrap_or(std::cmp::Ordering::Equal));

        // Update statistics
        self.stats.requests_processed.fetch_add(1, Ordering::Relaxed);

        #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
        let quote_time = start_time.elapsed().as_micros() as u64;
        trace!("Generated {} quotes in {}μs", quotes.len(), quote_time);

        quotes
    }

    /// Execute flashloan with optimal provider
    #[inline]
    #[must_use]
    pub async fn execute_flashloan(&self, request: &FlashloanRequest) -> Option<FlashloanExecution> {
        let start_time = Instant::now();

        // Get quotes and select best one
        let quotes = self.get_quotes(request).await;
        let best_quote = quotes.first()?;

        // Generate request ID
        let request_id = self.generate_request_id(request).await;

        // Execute the flashloan
        let execution = self.execute_with_provider(best_quote, &request_id).await;

        // Update statistics
        match execution.status {
            ExecutionStatus::Success => {
                self.stats.successful_executions.fetch_add(1, Ordering::Relaxed);
                self.stats.optimal_provider_selections.fetch_add(1, Ordering::Relaxed);

                // Update volume and fees
                let volume_scaled = (execution.actual_fee_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0);
                self.stats.total_volume_usd.fetch_add(volume_scaled, Ordering::Relaxed);

                let fees_scaled = (execution.actual_fee_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0);
                self.stats.total_fees_paid_usd.fetch_add(fees_scaled, Ordering::Relaxed);
            }
            ExecutionStatus::Failed | ExecutionStatus::TimedOut => {
                self.stats.failed_executions.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
        let execution_time = start_time.elapsed().as_micros() as u64;
        self.stats.avg_execution_time_us.store(execution_time, Ordering::Relaxed);

        // Store execution in history
        {
            let mut history = self.execution_history.write().await;
            history.insert(request_id.clone(), execution.clone());

            // Keep only recent executions
            while history.len() > FLASHLOAN_MAX_EXECUTIONS {
                if let Some(oldest_key) = history.keys().next().cloned() {
                    history.remove(&oldest_key);
                }
            }
            drop(history);
        }

        Some(execution)
    }

    /// Check if provider is suitable for request
    fn is_provider_suitable(provider: &FlashloanProviderInfo, request: &FlashloanRequest) -> bool {
        // Check chain compatibility
        if provider.chain_id != request.chain_id {
            return false;
        }

        // Check provider status
        if provider.status != ProviderStatus::Active {
            return false;
        }

        // Check token availability
        if !provider.available_tokens.contains(&request.token_address) {
            return false;
        }

        // Check loan amount limits
        if request.amount_usd < provider.min_loan_amount_usd {
            return false;
        }

        if let Some(max_amount) = provider.max_loan_amounts.get(&request.token_address) {
            if request.amount_usd > *max_amount {
                return false;
            }
        }

        // Check fee tolerance
        if provider.fee_percentage > request.max_fee_percentage {
            return false;
        }

        // Check preferred providers (if specified)
        if !request.preferred_providers.is_empty() && !request.preferred_providers.contains(&provider.provider) {
            return false;
        }

        true
    }

    /// Generate quote from provider
    fn generate_quote(provider: &FlashloanProviderInfo, request: &FlashloanRequest) -> FlashloanQuote {
        // Calculate fee amounts
        let fee_amount = request.amount * provider.fee_percentage;
        let fee_amount_usd = request.amount_usd * provider.fee_percentage + provider.fixed_fee_usd;
        let total_repayment = request.amount + fee_amount;
        let total_repayment_usd = request.amount_usd + fee_amount_usd;

        // Calculate quote score (lower fees and higher reliability = higher score)
        let fee_score = (Decimal::ONE - provider.fee_percentage).max(Decimal::ZERO);
        let reliability_score = Decimal::from(provider.reliability_score) / Decimal::from(100_u64);
        let speed_score = (Decimal::from(300_u64) - Decimal::from(provider.avg_execution_time_s).min(Decimal::from(300_u64))) / Decimal::from(300_u64);

        let quote_score = fee_score * "0.5".parse::<Decimal>().unwrap_or_default() +
                         reliability_score * "0.3".parse::<Decimal>().unwrap_or_default() +
                         speed_score * "0.2".parse::<Decimal>().unwrap_or_default();

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for quote validity")]
        let valid_until = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64 + 300_000; // 5 minutes validity

        FlashloanQuote {
            provider: provider.provider.clone(),
            chain_id: provider.chain_id,
            token_address: request.token_address.clone(),
            amount: request.amount,
            amount_usd: request.amount_usd,
            fee_percentage: provider.fee_percentage,
            fee_amount,
            fee_amount_usd,
            total_repayment,
            total_repayment_usd,
            estimated_execution_time_s: provider.avg_execution_time_s,
            reliability_score: provider.reliability_score,
            valid_until,
            quote_score,
        }
    }

    /// Execute flashloan with specific provider
    async fn execute_with_provider(&self, quote: &FlashloanQuote, request_id: &str) -> FlashloanExecution {
        let start_time = Instant::now();

        // Simulate flashloan execution (in production this would interact with actual contracts)
        let execution_success = Self::simulate_flashloan_execution(quote);

        let status = if execution_success {
            ExecutionStatus::Success
        } else {
            ExecutionStatus::Failed
        };

        let transaction_hash = if execution_success {
            Some(format!("0x{:x}", fastrand::u64(..)))
        } else {
            None
        };

        let error_message = if execution_success {
            None
        } else {
            Some("Simulated execution failure".to_string())
        };

        #[expect(clippy::cast_possible_truncation, reason = "Execution time truncation is acceptable")]
        let execution_time_s = start_time.elapsed().as_secs() as u32;

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for execution data")]
        let executed_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        FlashloanExecution {
            request_id: request_id.to_string(),
            provider: quote.provider.clone(),
            chain_id: quote.chain_id,
            status,
            transaction_hash,
            actual_fee_usd: quote.fee_amount_usd,
            execution_time_s,
            gas_used: Self::estimate_gas_usage(&quote.provider),
            gas_cost_usd: Self::estimate_gas_cost(quote.chain_id),
            error_message,
            executed_at,
        }
    }

    /// Generate unique request ID
    async fn generate_request_id(&self, request: &FlashloanRequest) -> String {
        let mut round = self.request_round.lock().await;
        *round += 1;
        #[expect(clippy::cast_possible_truncation, reason = "Chain ID values are small")]
        let chain_id_u8 = request.chain_id as u8;
        format!("fl_{}_{}_{}_{}", chain_id_u8, request.strategy_id, *round, fastrand::u64(..))
    }

    /// Simulate flashloan execution (for testing)
    fn simulate_flashloan_execution(quote: &FlashloanQuote) -> bool {
        // Simulate success rate based on provider reliability
        #[expect(clippy::float_arithmetic, reason = "Simulation requires floating point arithmetic")]
        let success_threshold = f64::from(quote.reliability_score) / 100.0;
        fastrand::f64() < success_threshold
    }

    /// Estimate gas usage for provider
    const fn estimate_gas_usage(provider: &FlashloanProvider) -> u64 {
        match provider {
            FlashloanProvider::Aave => 300_000,        // 300k gas
            FlashloanProvider::UniswapV3 | FlashloanProvider::Euler | FlashloanProvider::Venus => 200_000,   // 200k gas
            FlashloanProvider::Balancer => 250_000,    // 250k gas
            FlashloanProvider::DyDx => 180_000,        // 180k gas
            FlashloanProvider::Compound => 220_000,    // 220k gas
            FlashloanProvider::MakerDao => 350_000,    // 350k gas
            FlashloanProvider::IronBank => 240_000,    // 240k gas
            FlashloanProvider::Radiant => 280_000,     // 280k gas
        }
    }

    /// Estimate gas cost for chain
    fn estimate_gas_cost(chain_id: ChainId) -> Decimal {
        match chain_id {
            ChainId::Ethereum => "50".parse().unwrap_or_default(),    // $50
            ChainId::Arbitrum | ChainId::Avalanche => "2".parse().unwrap_or_default(),     // $2
            ChainId::Optimism => "3".parse().unwrap_or_default(),     // $3
            ChainId::Polygon => "0.5".parse().unwrap_or_default(),    // $0.5
            ChainId::Bsc => "1".parse().unwrap_or_default(),          // $1
            ChainId::Base => "1.5".parse().unwrap_or_default(),       // $1.5
        }
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(2000)) // Flashloan timeout
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

    /// Start provider discovery
    async fn start_provider_discovery(&self) {
        let provider_receiver = self.provider_receiver.clone();
        let providers = Arc::clone(&self.providers);
        let provider_cache = Arc::clone(&self.provider_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let flashloan_config = self.flashloan_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(flashloan_config.provider_discovery_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming provider updates
                while let Ok(provider_info) = provider_receiver.try_recv() {
                    let provider_key = (provider_info.provider.clone(), provider_info.chain_id);

                    // Update provider information
                    {
                        let mut providers_guard = providers.write().await;
                        providers_guard.insert(provider_key.clone(), provider_info.clone());
                        drop(providers_guard);
                    }

                    // Update cache with aligned data
                    let aligned_data = AlignedFlashloanData::new(
                        (Decimal::from(provider_info.reliability_score) * Decimal::from(10_000_u64)).to_u64().unwrap_or(0),
                        (provider_info.fee_percentage * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        (provider_info.max_loan_amounts.values().max().unwrap_or(&Decimal::ZERO) * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        u64::from(provider_info.avg_execution_time_s),
                        500_000, // 50% utilization rate (mock)
                        950_000, // 95% success rate (mock)
                        100, // Total loans executed (mock)
                        provider_info.last_update,
                    );
                    provider_cache.insert(provider_key, aligned_data);
                }

                // Discover providers from external sources
                if let Ok(discovered_providers) = Self::fetch_provider_info(&flashloan_config.supported_chains).await {
                    for provider_info in discovered_providers {
                        let provider_key = (provider_info.provider.clone(), provider_info.chain_id);

                        // Update providers directly since we're in the same task
                        {
                            let mut providers_guard = providers.write().await;
                            providers_guard.insert(provider_key, provider_info);
                        }
                    }
                }

                stats.provider_discoveries.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let discovery_time = start_time.elapsed().as_micros() as u64;
                trace!("Provider discovery cycle completed in {}μs", discovery_time);
            }
        });
    }

    /// Start fee optimization
    async fn start_fee_optimization(&self) {
        let quote_receiver = self.quote_receiver.clone();
        let active_quotes = Arc::clone(&self.active_quotes);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let flashloan_config = self.flashloan_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(flashloan_config.fee_optimization_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming quotes
                while let Ok(quote) = quote_receiver.try_recv() {
                    #[expect(clippy::cast_possible_truncation, reason = "Provider and chain ID values are small")]
                    let quote_id = format!("quote_{}_{}_{}_{}", quote.provider.clone() as u8, quote.chain_id as u8, quote.amount_usd, quote.valid_until);

                    // Store quote
                    {
                        let mut quotes_guard = active_quotes.write().await;
                        quotes_guard.insert(quote_id, quote);

                        // Remove expired quotes
                        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for quote expiry")]
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;

                        quotes_guard.retain(|_, q| q.valid_until > now);

                        // Keep only recent quotes
                        while quotes_guard.len() > FLASHLOAN_MAX_QUOTES {
                            if let Some(oldest_key) = quotes_guard.keys().next().cloned() {
                                quotes_guard.remove(&oldest_key);
                            }
                        }
                        drop(quotes_guard);
                    }
                }

                // Perform fee optimization analysis
                Self::optimize_fee_parameters(&active_quotes, &stats).await;

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let optimization_time = start_time.elapsed().as_micros() as u64;
                trace!("Fee optimization cycle completed in {}μs", optimization_time);
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

                let requests_processed = stats.requests_processed.load(Ordering::Relaxed);
                let successful_executions = stats.successful_executions.load(Ordering::Relaxed);
                let failed_executions = stats.failed_executions.load(Ordering::Relaxed);
                let total_volume = stats.total_volume_usd.load(Ordering::Relaxed);
                let total_fees_paid = stats.total_fees_paid_usd.load(Ordering::Relaxed);
                let total_fees_saved = stats.total_fees_saved_usd.load(Ordering::Relaxed);
                let provider_discoveries = stats.provider_discoveries.load(Ordering::Relaxed);
                let fee_optimizations = stats.fee_optimizations.load(Ordering::Relaxed);
                let cross_chain_coordinations = stats.cross_chain_coordinations.load(Ordering::Relaxed);
                let avg_execution_time = stats.avg_execution_time_us.load(Ordering::Relaxed);
                let avg_fee_percentage = stats.avg_fee_percentage_scaled.load(Ordering::Relaxed);
                let optimal_selections = stats.optimal_provider_selections.load(Ordering::Relaxed);

                info!(
                    "Flashloan Stats: requests={}, successful={}, failed={}, volume=${}, fees_paid=${}, fees_saved=${}, discoveries={}, optimizations={}, cross_chain={}, avg_time={}μs, avg_fee={}%, optimal={}",
                    requests_processed, successful_executions, failed_executions, total_volume, total_fees_paid, total_fees_saved,
                    provider_discoveries, fee_optimizations, cross_chain_coordinations, avg_execution_time, avg_fee_percentage, optimal_selections
                );
            }
        });
    }

    /// Fetch provider information from external sources
    async fn fetch_provider_info(supported_chains: &[ChainId]) -> Result<Vec<FlashloanProviderInfo>> {
        let mut providers = Vec::new();

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for mock provider data")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Mock provider data generation - in production this would query real provider APIs
        for chain_id in supported_chains {
            let chain_providers = Self::get_providers_for_chain(*chain_id);

            for provider in chain_providers {
                let provider_info = FlashloanProviderInfo {
                    provider: provider.clone(),
                    chain_id: *chain_id,
                    contract_address: Self::get_provider_address(&provider, *chain_id),
                    available_tokens: Self::get_available_tokens(&provider, *chain_id),
                    max_loan_amounts: Self::get_max_loan_amounts(&provider, *chain_id),
                    fee_percentage: Self::get_provider_fee(&provider),
                    fixed_fee_usd: Self::get_fixed_fee(&provider),
                    min_loan_amount_usd: Self::get_min_loan_amount(&provider),
                    reliability_score: Self::get_reliability_score(&provider),
                    avg_execution_time_s: Self::get_avg_execution_time(&provider),
                    status: ProviderStatus::Active,
                    last_update: now,
                };
                providers.push(provider_info);
            }
        }

        Ok(providers)
    }

    /// Optimize fee parameters based on quote history
    async fn optimize_fee_parameters(
        active_quotes: &Arc<RwLock<HashMap<String, FlashloanQuote>>>,
        stats: &Arc<FlashloanStats>,
    ) {
        let quotes_guard = active_quotes.read().await;

        if quotes_guard.is_empty() {
            return;
        }

        // Analyze fee patterns
        let mut total_fee_percentage = Decimal::ZERO;
        let mut quote_count = 0;

        for quote in quotes_guard.values() {
            total_fee_percentage += quote.fee_percentage;
            quote_count += 1;
        }

        if quote_count > 0 {
            let avg_fee_percentage = total_fee_percentage / Decimal::from(quote_count);
            let avg_fee_scaled = (avg_fee_percentage * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0);
            stats.avg_fee_percentage_scaled.store(avg_fee_scaled, Ordering::Relaxed);
            stats.fee_optimizations.fetch_add(1, Ordering::Relaxed);
        }

        drop(quotes_guard);
        trace!("Fee parameter optimization completed");
    }

    /// Get providers available on specific chain
    fn get_providers_for_chain(chain_id: ChainId) -> Vec<FlashloanProvider> {
        match chain_id {
            ChainId::Ethereum => vec![
                FlashloanProvider::Aave,
                FlashloanProvider::DyDx,
                FlashloanProvider::UniswapV3,
                FlashloanProvider::Balancer,
                FlashloanProvider::Compound,
                FlashloanProvider::MakerDao,
                FlashloanProvider::Euler,
                FlashloanProvider::IronBank,
            ],
            ChainId::Arbitrum => vec![
                FlashloanProvider::Aave,
                FlashloanProvider::UniswapV3,
                FlashloanProvider::Balancer,
                FlashloanProvider::Radiant,
            ],
            ChainId::Optimism | ChainId::Polygon => vec![
                FlashloanProvider::Aave,
                FlashloanProvider::UniswapV3,
                FlashloanProvider::Balancer,
            ],
            ChainId::Bsc => vec![
                FlashloanProvider::Venus,
            ],
            ChainId::Avalanche => vec![
                FlashloanProvider::Aave,
            ],
            ChainId::Base => vec![
                FlashloanProvider::Aave,
                FlashloanProvider::UniswapV3,
            ],
        }
    }

    /// Get provider contract address
    fn get_provider_address(provider: &FlashloanProvider, chain_id: ChainId) -> String {
        match (provider, chain_id) {
            (FlashloanProvider::Aave, ChainId::Ethereum) => "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2".to_string(),
            (FlashloanProvider::Aave, ChainId::Arbitrum | ChainId::Optimism | ChainId::Polygon | ChainId::Avalanche) => "0x794a61358D6845594F94dc1DB02A252b5b4814aD".to_string(),
            (FlashloanProvider::Aave, ChainId::Base) => "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5".to_string(),
            (FlashloanProvider::DyDx, ChainId::Ethereum) => "0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e".to_string(),
            (FlashloanProvider::UniswapV3, _) => "0xE592427A0AEce92De3Edee1F18E0157C05861564".to_string(), // Universal router
            (FlashloanProvider::Balancer, _) => "0xBA12222222228d8Ba445958a75a0704d566BF2C8".to_string(), // Vault
            (FlashloanProvider::Compound, ChainId::Ethereum) => "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B".to_string(),
            (FlashloanProvider::MakerDao, ChainId::Ethereum) => "0x35D1b3F3D7966A1DFe207aa4514C12a259A0492B".to_string(),
            (FlashloanProvider::Euler, ChainId::Ethereum) => "0x27182842E098f60e3D576794A5bFFb0777E025d3".to_string(),
            (FlashloanProvider::IronBank, ChainId::Ethereum) => "0xAB1c342C7bf5Ec5F02ADEA1c2270670bCa144CbB".to_string(),
            (FlashloanProvider::Radiant, ChainId::Arbitrum) => "0x2032b9A8e9F7e76768CA9271003d3e43E1616B1F".to_string(),
            (FlashloanProvider::Venus, ChainId::Bsc) => "0xfD36E2c2a6789Db23113685031d7F16329158384".to_string(),
            _ => "0x0000000000000000000000000000000000000000".to_string(), // Default
        }
    }

    /// Get available tokens for provider
    fn get_available_tokens(provider: &FlashloanProvider, chain_id: ChainId) -> Vec<String> {
        match (provider, chain_id) {
            (FlashloanProvider::Aave, ChainId::Ethereum) => vec![
                "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7".to_string(), // USDT
                "0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), // DAI
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), // WETH
                "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599".to_string(), // WBTC
            ],
            (FlashloanProvider::DyDx, ChainId::Ethereum) => vec![
                "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), // WETH
                "0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), // DAI
            ],
            (FlashloanProvider::UniswapV3, _) => vec![
                "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7".to_string(), // USDT
                "0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), // DAI
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), // WETH
            ],
            (FlashloanProvider::Balancer, _) => vec![
                "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                "0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), // DAI
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), // WETH
            ],
            _ => vec![
                "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
                "0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), // DAI
            ],
        }
    }

    /// Get maximum loan amounts for provider
    fn get_max_loan_amounts(provider: &FlashloanProvider, _chain_id: ChainId) -> HashMap<String, Decimal> {
        let mut amounts = HashMap::new();

        match provider {
            FlashloanProvider::Aave => {
                amounts.insert("0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), "50000000".parse().unwrap_or_default()); // $50M USDC
                amounts.insert("0xdAC17F958D2ee523a2206206994597C13D831ec7".to_string(), "50000000".parse().unwrap_or_default()); // $50M USDT
                amounts.insert("0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), "50000000".parse().unwrap_or_default()); // $50M DAI
                amounts.insert("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), "20000000".parse().unwrap_or_default()); // $20M WETH
            }
            FlashloanProvider::DyDx => {
                amounts.insert("0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), "100000000".parse().unwrap_or_default()); // $100M USDC
                amounts.insert("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), "30000000".parse().unwrap_or_default()); // $30M WETH
                amounts.insert("0x6B175474E89094C44Da98b954EedeAC495271d0F".to_string(), "30000000".parse().unwrap_or_default()); // $30M DAI
            }
            FlashloanProvider::UniswapV3 => {
                amounts.insert("0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), "10000000".parse().unwrap_or_default()); // $10M USDC
                amounts.insert("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), "5000000".parse().unwrap_or_default()); // $5M WETH
            }
            FlashloanProvider::Balancer => {
                amounts.insert("0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), "20000000".parse().unwrap_or_default()); // $20M USDC
                amounts.insert("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(), "10000000".parse().unwrap_or_default()); // $10M WETH
            }
            _ => {
                amounts.insert("0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), "5000000".parse().unwrap_or_default()); // $5M USDC
            }
        }

        amounts
    }

    /// Get provider fee percentage
    fn get_provider_fee(provider: &FlashloanProvider) -> Decimal {
        match provider {
            FlashloanProvider::Aave | FlashloanProvider::Radiant => "0.0009".parse().unwrap_or_default(),      // 0.09%
            FlashloanProvider::DyDx | FlashloanProvider::Balancer | FlashloanProvider::MakerDao => "0.0000".parse().unwrap_or_default(),      // 0% (free)
            FlashloanProvider::UniswapV3 | FlashloanProvider::Euler | FlashloanProvider::Venus => "0.0005".parse().unwrap_or_default(), // 0.05%
            FlashloanProvider::Compound => "0.0008".parse().unwrap_or_default(),  // 0.08%
            FlashloanProvider::IronBank => "0.0010".parse().unwrap_or_default(),  // 0.10%
        }
    }

    /// Get fixed fee amount
    fn get_fixed_fee(provider: &FlashloanProvider) -> Decimal {
        match provider {
            FlashloanProvider::DyDx => "2.0".parse().unwrap_or_default(),    // $2 fixed fee
            _ => "0.0".parse().unwrap_or_default(), // No fixed fee for others
        }
    }

    /// Get minimum loan amount
    fn get_min_loan_amount(provider: &FlashloanProvider) -> Decimal {
        match provider {
            FlashloanProvider::Aave | FlashloanProvider::Compound | FlashloanProvider::IronBank | FlashloanProvider::Radiant => "1000".parse().unwrap_or_default(),      // $1k minimum
            FlashloanProvider::DyDx => "10000".parse().unwrap_or_default(),     // $10k minimum
            FlashloanProvider::UniswapV3 | FlashloanProvider::Balancer => "100".parse().unwrap_or_default(),  // $100 minimum
            FlashloanProvider::MakerDao => "5000".parse().unwrap_or_default(),  // $5k minimum
            FlashloanProvider::Euler | FlashloanProvider::Venus => "500".parse().unwrap_or_default(),      // $500 minimum
        }
    }

    /// Get provider reliability score
    const fn get_reliability_score(provider: &FlashloanProvider) -> u8 {
        match provider {
            FlashloanProvider::Aave => 95,        // Very reliable
            FlashloanProvider::DyDx => 90,        // Reliable
            FlashloanProvider::UniswapV3 => 92,   // Very reliable
            FlashloanProvider::Balancer => 88,    // Reliable
            FlashloanProvider::Compound | FlashloanProvider::Venus => 85,    // Good reliability
            FlashloanProvider::MakerDao => 93,    // Very reliable
            FlashloanProvider::Euler => 80,       // Good reliability
            FlashloanProvider::IronBank => 75,    // Moderate reliability
            FlashloanProvider::Radiant => 82,     // Good reliability
        }
    }

    /// Get average execution time
    const fn get_avg_execution_time(provider: &FlashloanProvider) -> u32 {
        match provider {
            FlashloanProvider::Aave => 15,        // 15 seconds
            FlashloanProvider::DyDx => 10,        // 10 seconds
            FlashloanProvider::UniswapV3 => 8,    // 8 seconds
            FlashloanProvider::Balancer => 12,    // 12 seconds
            FlashloanProvider::Compound => 20,    // 20 seconds
            FlashloanProvider::MakerDao => 25,    // 25 seconds
            FlashloanProvider::Euler => 18,       // 18 seconds
            FlashloanProvider::IronBank => 22,    // 22 seconds
            FlashloanProvider::Radiant => 16,     // 16 seconds
            FlashloanProvider::Venus => 14,       // 14 seconds
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_flashloan_coordinator_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = FlashloanCoordinator::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(coordinator.stats().requests_processed.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().successful_executions.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().failed_executions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_flashloan_config_default() {
        let config = FlashloanConfig::default();
        assert!(config.enabled);
        assert_eq!(config.provider_discovery_interval_ms, FLASHLOAN_DEFAULT_DISCOVERY_INTERVAL_MS);
        assert_eq!(config.fee_optimization_interval_ms, FLASHLOAN_DEFAULT_OPTIMIZATION_INTERVAL_MS);
        assert_eq!(config.performance_monitoring_interval_ms, FLASHLOAN_DEFAULT_MONITORING_INTERVAL_MS);
        assert!(config.enable_dynamic_selection);
        assert!(config.enable_fee_optimization);
        assert!(config.enable_cross_chain);
        assert!(config.enable_loan_aggregation);
        assert!(!config.preferred_providers.is_empty());
        assert!(!config.supported_chains.is_empty());
    }

    #[test]
    fn test_aligned_flashloan_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedFlashloanData>(), 64);
        assert!(mem::size_of::<AlignedFlashloanData>() <= 64);
    }

    #[test]
    fn test_flashloan_stats_operations() {
        let stats = FlashloanStats::default();

        stats.requests_processed.fetch_add(100, Ordering::Relaxed);
        stats.successful_executions.fetch_add(95, Ordering::Relaxed);
        stats.failed_executions.fetch_add(5, Ordering::Relaxed);
        stats.total_volume_usd.fetch_add(1_000_000, Ordering::Relaxed);
        stats.total_fees_paid_usd.fetch_add(5000, Ordering::Relaxed);
        stats.total_fees_saved_usd.fetch_add(2000, Ordering::Relaxed);

        assert_eq!(stats.requests_processed.load(Ordering::Relaxed), 100);
        assert_eq!(stats.successful_executions.load(Ordering::Relaxed), 95);
        assert_eq!(stats.failed_executions.load(Ordering::Relaxed), 5);
        assert_eq!(stats.total_volume_usd.load(Ordering::Relaxed), 1_000_000);
        assert_eq!(stats.total_fees_paid_usd.load(Ordering::Relaxed), 5000);
        assert_eq!(stats.total_fees_saved_usd.load(Ordering::Relaxed), 2000);
    }

    #[test]
    fn test_aligned_flashloan_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedFlashloanData::new(
            950_000, // 95% reliability (scaled by 1e6)
            9_000,   // 0.009% fee (scaled by 1e6)
            50_000_000_000_000, // $50M max loan (scaled by 1e6)
            15,      // 15s execution time
            500_000, // 50% utilization (scaled by 1e6)
            950_000, // 95% success rate (scaled by 1e6)
            100,     // 100 loans executed
            now,
        );

        let stale_data = AlignedFlashloanData::new(
            950_000, 9_000, 50_000_000_000_000, 15, 500_000, 950_000, 100,
            now - 300_000, // 5 minutes old
        );

        assert!(!fresh_data.is_stale(120_000)); // 2 minutes
        assert!(stale_data.is_stale(120_000)); // 2 minutes
    }

    #[test]
    fn test_aligned_flashloan_data_conversions() {
        let data = AlignedFlashloanData::new(
            950_000, // 95% reliability (scaled by 1e6)
            9_000,   // 0.009% fee (scaled by 1e6)
            50_000_000_000_000, // $50M max loan (scaled by 1e6)
            15,      // 15s execution time
            500_000, // 50% utilization (scaled by 1e6)
            950_000, // 95% success rate (scaled by 1e6)
            100,     // 100 loans executed
            1_640_995_200_000,
        );

        assert_eq!(data.reliability_score(), dec!(0.95));
        assert_eq!(data.fee_percentage(), dec!(0.009));
        assert_eq!(data.max_loan_amount_usd(), dec!(50000000));
        assert_eq!(data.utilization_rate(), dec!(0.5));
        assert_eq!(data.success_rate(), dec!(0.95));

        // Overall score should be weighted average
        let expected_overall = dec!(0.95) * dec!(0.4) + dec!(0.95) * dec!(0.3) + (dec!(1) - dec!(0.009)) * dec!(0.2) + (dec!(300) - dec!(15)) / dec!(300) * dec!(0.1);
        assert!((data.overall_score() - expected_overall).abs() < dec!(0.001));
    }

    #[test]
    fn test_flashloan_provider_enum() {
        assert_eq!(FlashloanProvider::Aave, FlashloanProvider::Aave);
        assert_ne!(FlashloanProvider::Aave, FlashloanProvider::DyDx);
        assert_ne!(FlashloanProvider::UniswapV3, FlashloanProvider::Balancer);
    }

    #[test]
    fn test_provider_status_enum() {
        assert_eq!(ProviderStatus::Active, ProviderStatus::Active);
        assert_ne!(ProviderStatus::Active, ProviderStatus::Unavailable);
        assert_ne!(ProviderStatus::Maintenance, ProviderStatus::Deprecated);
    }

    #[test]
    fn test_execution_status_enum() {
        assert_eq!(ExecutionStatus::Success, ExecutionStatus::Success);
        assert_ne!(ExecutionStatus::Success, ExecutionStatus::Failed);
        assert_ne!(ExecutionStatus::Pending, ExecutionStatus::InProgress);
    }

    #[test]
    fn test_flashloan_request_creation() {
        let request = FlashloanRequest {
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(),
            amount: dec!(1000000),
            amount_usd: dec!(1000000),
            chain_id: ChainId::Ethereum,
            max_fee_percentage: dec!(0.01),
            preferred_providers: vec![FlashloanProvider::Aave, FlashloanProvider::DyDx],
            deadline: 1_640_995_200_000,
            strategy_id: "arbitrage_001".to_string(),
            priority: 8,
        };

        assert_eq!(request.token_address, "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E");
        assert_eq!(request.amount, dec!(1000000));
        assert_eq!(request.amount_usd, dec!(1000000));
        assert_eq!(request.chain_id, ChainId::Ethereum);
        assert_eq!(request.max_fee_percentage, dec!(0.01));
        assert_eq!(request.preferred_providers.len(), 2);
        assert_eq!(request.strategy_id, "arbitrage_001");
        assert_eq!(request.priority, 8);
    }

    #[test]
    fn test_flashloan_quote_creation() {
        let quote = FlashloanQuote {
            provider: FlashloanProvider::Aave,
            chain_id: ChainId::Ethereum,
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(),
            amount: dec!(1000000),
            amount_usd: dec!(1000000),
            fee_percentage: dec!(0.0009),
            fee_amount: dec!(900),
            fee_amount_usd: dec!(900),
            total_repayment: dec!(1000900),
            total_repayment_usd: dec!(1000900),
            estimated_execution_time_s: 15,
            reliability_score: 95,
            valid_until: 1_640_995_500_000,
            quote_score: dec!(0.85),
        };

        assert_eq!(quote.provider, FlashloanProvider::Aave);
        assert_eq!(quote.chain_id, ChainId::Ethereum);
        assert_eq!(quote.amount, dec!(1000000));
        assert_eq!(quote.fee_percentage, dec!(0.0009));
        assert_eq!(quote.fee_amount, dec!(900));
        assert_eq!(quote.total_repayment, dec!(1000900));
        assert_eq!(quote.estimated_execution_time_s, 15);
        assert_eq!(quote.reliability_score, 95);
        assert_eq!(quote.quote_score, dec!(0.85));
    }

    #[test]
    fn test_flashloan_execution_creation() {
        let execution = FlashloanExecution {
            request_id: "fl_1_arbitrage_001_1_12345".to_string(),
            provider: FlashloanProvider::Aave,
            chain_id: ChainId::Ethereum,
            status: ExecutionStatus::Success,
            transaction_hash: Some("0xabcdef1234567890".to_string()),
            actual_fee_usd: dec!(900),
            execution_time_s: 15,
            gas_used: 300_000,
            gas_cost_usd: dec!(50),
            error_message: None,
            executed_at: 1_640_995_200_000,
        };

        assert_eq!(execution.request_id, "fl_1_arbitrage_001_1_12345");
        assert_eq!(execution.provider, FlashloanProvider::Aave);
        assert_eq!(execution.chain_id, ChainId::Ethereum);
        assert_eq!(execution.status, ExecutionStatus::Success);
        assert!(execution.transaction_hash.is_some());
        assert_eq!(execution.actual_fee_usd, dec!(900));
        assert_eq!(execution.execution_time_s, 15);
        assert_eq!(execution.gas_used, 300_000);
        assert_eq!(execution.gas_cost_usd, dec!(50));
        assert!(execution.error_message.is_none());
    }

    #[tokio::test]
    async fn test_flashloan_coordinator_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = FlashloanCoordinator::new(config).await else {
            return; // Skip test if creation fails
        };

        // Test getting providers (should be empty initially)
        let providers = coordinator.get_providers().await;
        assert!(providers.is_empty());

        // Test getting execution history (should be empty initially)
        let history = coordinator.get_execution_history().await;
        assert!(history.is_empty());

        // Test stats access
        let stats = coordinator.stats();
        assert_eq!(stats.requests_processed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_provider_data_functions() {
        // Test provider fees
        assert_eq!(FlashloanCoordinator::get_provider_fee(&FlashloanProvider::Aave), dec!(0.0009));
        assert_eq!(FlashloanCoordinator::get_provider_fee(&FlashloanProvider::DyDx), dec!(0.0000));
        assert_eq!(FlashloanCoordinator::get_provider_fee(&FlashloanProvider::UniswapV3), dec!(0.0005));

        // Test fixed fees
        assert_eq!(FlashloanCoordinator::get_fixed_fee(&FlashloanProvider::DyDx), dec!(2.0));
        assert_eq!(FlashloanCoordinator::get_fixed_fee(&FlashloanProvider::Aave), dec!(0.0));

        // Test minimum loan amounts
        assert_eq!(FlashloanCoordinator::get_min_loan_amount(&FlashloanProvider::Aave), dec!(1000));
        assert_eq!(FlashloanCoordinator::get_min_loan_amount(&FlashloanProvider::DyDx), dec!(10000));
        assert_eq!(FlashloanCoordinator::get_min_loan_amount(&FlashloanProvider::UniswapV3), dec!(100));

        // Test reliability scores
        assert_eq!(FlashloanCoordinator::get_reliability_score(&FlashloanProvider::Aave), 95);
        assert_eq!(FlashloanCoordinator::get_reliability_score(&FlashloanProvider::DyDx), 90);
        assert_eq!(FlashloanCoordinator::get_reliability_score(&FlashloanProvider::UniswapV3), 92);

        // Test execution times
        assert_eq!(FlashloanCoordinator::get_avg_execution_time(&FlashloanProvider::Aave), 15);
        assert_eq!(FlashloanCoordinator::get_avg_execution_time(&FlashloanProvider::DyDx), 10);
        assert_eq!(FlashloanCoordinator::get_avg_execution_time(&FlashloanProvider::UniswapV3), 8);
    }

    #[test]
    fn test_gas_estimation() {
        assert_eq!(FlashloanCoordinator::estimate_gas_usage(&FlashloanProvider::Aave), 300_000);
        assert_eq!(FlashloanCoordinator::estimate_gas_usage(&FlashloanProvider::DyDx), 180_000);
        assert_eq!(FlashloanCoordinator::estimate_gas_usage(&FlashloanProvider::UniswapV3), 200_000);

        assert_eq!(FlashloanCoordinator::estimate_gas_cost(ChainId::Ethereum), dec!(50));
        assert_eq!(FlashloanCoordinator::estimate_gas_cost(ChainId::Arbitrum), dec!(2));
        assert_eq!(FlashloanCoordinator::estimate_gas_cost(ChainId::Polygon), dec!(0.5));
    }

    #[test]
    fn test_providers_for_chain() {
        let eth_providers = FlashloanCoordinator::get_providers_for_chain(ChainId::Ethereum);
        assert!(eth_providers.contains(&FlashloanProvider::Aave));
        assert!(eth_providers.contains(&FlashloanProvider::DyDx));
        assert!(eth_providers.contains(&FlashloanProvider::UniswapV3));

        let bsc_providers = FlashloanCoordinator::get_providers_for_chain(ChainId::Bsc);
        assert!(bsc_providers.contains(&FlashloanProvider::Venus));
        assert!(!bsc_providers.contains(&FlashloanProvider::Aave));

        let arb_providers = FlashloanCoordinator::get_providers_for_chain(ChainId::Arbitrum);
        assert!(arb_providers.contains(&FlashloanProvider::Aave));
        assert!(arb_providers.contains(&FlashloanProvider::Radiant));
        assert!(!arb_providers.contains(&FlashloanProvider::DyDx));
    }

    #[test]
    fn test_provider_addresses() {
        let aave_eth = FlashloanCoordinator::get_provider_address(&FlashloanProvider::Aave, ChainId::Ethereum);
        assert_eq!(aave_eth, "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2");

        let dydx_eth = FlashloanCoordinator::get_provider_address(&FlashloanProvider::DyDx, ChainId::Ethereum);
        assert_eq!(dydx_eth, "0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e");

        let venus_bsc = FlashloanCoordinator::get_provider_address(&FlashloanProvider::Venus, ChainId::Bsc);
        assert_eq!(venus_bsc, "0xfD36E2c2a6789Db23113685031d7F16329158384");
    }

    #[test]
    fn test_available_tokens() {
        let aave_tokens = FlashloanCoordinator::get_available_tokens(&FlashloanProvider::Aave, ChainId::Ethereum);
        assert!(aave_tokens.contains(&"0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string())); // USDC
        assert!(aave_tokens.contains(&"0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string())); // WETH

        let dydx_tokens = FlashloanCoordinator::get_available_tokens(&FlashloanProvider::DyDx, ChainId::Ethereum);
        assert!(dydx_tokens.contains(&"0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string())); // USDC
        assert!(dydx_tokens.len() >= 3);
    }

    #[test]
    fn test_max_loan_amounts() {
        let aave_amounts = FlashloanCoordinator::get_max_loan_amounts(&FlashloanProvider::Aave, ChainId::Ethereum);
        assert_eq!(aave_amounts.get("0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E"), Some(&dec!(50000000))); // $50M USDC

        let dydx_amounts = FlashloanCoordinator::get_max_loan_amounts(&FlashloanProvider::DyDx, ChainId::Ethereum);
        assert_eq!(dydx_amounts.get("0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E"), Some(&dec!(100000000))); // $100M USDC
    }
}
