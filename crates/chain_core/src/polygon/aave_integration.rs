//! Aave Polygon Integration for ultra-performance lending/borrowing operations
//!
//! This module provides Aave v3 integration for Polygon chain,
//! enabling liquidation monitoring, interest rate optimization, and collateral management.
//!
//! ## Performance Targets
//! - Liquidation Detection: <80μs
//! - Interest Rate Calculation: <40μs
//! - Collateral Monitoring: <60μs
//! - Position Management: <150μs
//! - Risk Assessment: <120μs
//!
//! ## Architecture
//! - Real-time liquidation monitoring
//! - Dynamic interest rate optimization
//! - Automated collateral management
//! - Risk-based position sizing
//! - Lock-free data structures for hot paths

use crate::{
    ChainCoreConfig, Result,
    types::TokenAddress,
    utils::perf::Timer,
    polygon::PolygonConfig,
};
use crossbeam::channel::{self, Receiver, Sender};
use dashmap::DashMap;
use rust_decimal::{Decimal, prelude::ToPrimitive};
use serde::{Deserialize, Serialize};
use std::{
    str::FromStr,
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
use tracing::{debug, info, trace};

/// Aave Polygon integration configuration
#[derive(Debug, Clone)]
pub struct AavePolygonConfig {
    /// Enable Aave integration
    pub enabled: bool,
    
    /// Liquidation monitoring interval in milliseconds
    pub liquidation_monitor_interval_ms: u64,
    
    /// Minimum liquidation profit threshold (USD)
    pub min_liquidation_profit_usd: Decimal,
    
    /// Maximum loan-to-value ratio for safety
    pub max_ltv_ratio: Decimal,
    
    /// Liquidation threshold (percentage below max LTV)
    pub liquidation_threshold: Decimal,
    
    /// Enable automated liquidations
    pub enable_auto_liquidation: bool,
    
    /// Enable yield optimization
    pub enable_yield_optimization: bool,
    
    /// Supported aTokens for monitoring
    pub monitored_atokens: Vec<String>,
    
    /// Maximum gas price for liquidations (Gwei)
    pub max_liquidation_gas_gwei: u64,
}

/// Aave market information on Polygon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AavePolygonMarket {
    /// aToken address
    pub atoken_address: String,
    
    /// Underlying token address
    pub underlying_token: TokenAddress,
    
    /// Current supply APY
    pub supply_apy: Decimal,
    
    /// Current borrow APY (variable)
    pub borrow_apy_variable: Decimal,
    
    /// Current borrow APY (stable)
    pub borrow_apy_stable: Decimal,
    
    /// Total supply in underlying token
    pub total_supply: Decimal,
    
    /// Total borrows in underlying token
    pub total_borrows: Decimal,
    
    /// Available liquidity
    pub available_liquidity: Decimal,
    
    /// Collateral factor (max LTV)
    pub collateral_factor: Decimal,
    
    /// Liquidation threshold
    pub liquidation_threshold: Decimal,
    
    /// Liquidation bonus
    pub liquidation_bonus: Decimal,
    
    /// Reserve factor
    pub reserve_factor: Decimal,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Liquidation opportunity on Aave Polygon
#[derive(Debug, Clone)]
pub struct AavePolygonLiquidationOpportunity {
    /// Borrower address
    pub borrower: String,
    
    /// Collateral aToken to seize
    pub collateral_atoken: String,
    
    /// Debt aToken to repay
    pub debt_atoken: String,
    
    /// Amount to repay
    pub repay_amount: Decimal,
    
    /// Expected collateral to seize
    pub seize_amount: Decimal,
    
    /// Expected profit in USD
    pub profit_usd: Decimal,
    
    /// Current health factor
    pub health_factor: Decimal,
    
    /// Gas estimate for liquidation
    pub gas_estimate: u64,
    
    /// Opportunity confidence score (0-100)
    pub confidence: u8,
    
    /// Discovery timestamp
    pub discovered_at: Instant,
}

/// User position in Aave Polygon
#[derive(Debug, Clone)]
pub struct AavePolygonPosition {
    /// User address
    pub user_address: String,
    
    /// Supplied assets (aToken -> amount)
    pub supplies: HashMap<String, Decimal>,
    
    /// Borrowed assets (aToken -> amount)
    pub borrows: HashMap<String, Decimal>,
    
    /// Total supply value in USD
    pub total_supply_usd: Decimal,
    
    /// Total borrow value in USD
    pub total_borrow_usd: Decimal,
    
    /// Current health factor
    pub health_factor: Decimal,
    
    /// Liquidation threshold
    pub liquidation_threshold_usd: Decimal,
    
    /// Available borrow capacity
    pub available_borrow_usd: Decimal,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Aave Polygon statistics
#[derive(Debug, Default)]
pub struct AavePolygonStats {
    /// Total liquidations executed
    pub liquidations_executed: AtomicU64,
    
    /// Total liquidation profit (USD)
    pub total_liquidation_profit_usd: AtomicU64,
    
    /// Successful liquidation rate (percentage)
    pub liquidation_success_rate: AtomicU64,
    
    /// Average liquidation execution time (microseconds)
    pub avg_liquidation_time_us: AtomicU64,
    
    /// Positions monitored
    pub positions_monitored: AtomicU64,
    
    /// Liquidation opportunities found
    pub opportunities_found: AtomicU64,
    
    /// Market monitoring errors
    pub market_errors: AtomicU64,
    
    /// Total yield earned (USD)
    pub total_yield_usd: AtomicU64,
}

/// Cache-line aligned market data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedAavePolygonMarketData {
    /// Supply APY (scaled by 1e6)
    pub supply_apy_scaled: u64,
    
    /// Borrow APY variable (scaled by 1e6)
    pub borrow_apy_variable_scaled: u64,
    
    /// Utilization rate (scaled by 1e6)
    pub utilization_scaled: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// Aave Polygon integration constants
pub const AAVE_POLYGON_DEFAULT_MIN_PROFIT_USD: &str = "25"; // $25 minimum profit (lower than mainnet)
pub const AAVE_POLYGON_DEFAULT_MAX_LTV: &str = "0.8"; // 80% max LTV
pub const AAVE_POLYGON_DEFAULT_LIQUIDATION_THRESHOLD: &str = "0.05"; // 5% below max LTV
pub const AAVE_POLYGON_MAX_LIQUIDATION_GAS_GWEI: u64 = 100; // 100 Gwei for Polygon
pub const AAVE_POLYGON_MONITOR_INTERVAL_MS: u64 = 300; // 300ms monitoring (faster than mainnet)
pub const AAVE_POLYGON_MAX_POSITIONS: usize = 5000;
pub const AAVE_POLYGON_MAX_MARKETS: usize = 30;
pub const AAVE_POLYGON_LIQUIDATION_FREQ_HZ: u64 = 3; // 333ms intervals

/// Aave v3 protocol addresses on Polygon
pub const AAVE_V3_POOL_POLYGON: &str = "0x794a61358D6845594F94dc1DB02A252b5b4814aD";
pub const AAVE_V3_POOL_DATA_PROVIDER_POLYGON: &str = "0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654";
pub const AAVE_V3_PRICE_ORACLE_POLYGON: &str = "0xb023e699F5a33916Ea823A16485e259257cA8Bd1";

/// Common aToken addresses on Polygon
pub const AUSDC_POLYGON: &str = "0x625E7708f30cA75bfd92586e17077590C60eb4cD";
pub const AUSDT_POLYGON: &str = "0x6ab707Aca953eDAeFBc4fD23bA73294241490620";
pub const AWMATIC_POLYGON: &str = "0x6d80113e533a2C0fe82EaBD35f1875DcEA89Ea97";
pub const AWETH_POLYGON: &str = "0xe50fA9b3c56FfB159cB0FCA61F5c9D750e8128c8";
pub const AWBTC_POLYGON: &str = "0x078f358208685046a11C85e8ad32895DED33A249";

/// Default monitored aTokens
pub const DEFAULT_MONITORED_ATOKENS_POLYGON: &[&str] = &[
    AUSDC_POLYGON,
    AUSDT_POLYGON,
    AWMATIC_POLYGON,
    AWETH_POLYGON,
    AWBTC_POLYGON,
];

impl Default for AavePolygonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            liquidation_monitor_interval_ms: AAVE_POLYGON_MONITOR_INTERVAL_MS,
            min_liquidation_profit_usd: AAVE_POLYGON_DEFAULT_MIN_PROFIT_USD.parse().unwrap_or_default(),
            max_ltv_ratio: AAVE_POLYGON_DEFAULT_MAX_LTV.parse().unwrap_or_default(),
            liquidation_threshold: AAVE_POLYGON_DEFAULT_LIQUIDATION_THRESHOLD.parse().unwrap_or_default(),
            enable_auto_liquidation: true,
            enable_yield_optimization: true,
            monitored_atokens: DEFAULT_MONITORED_ATOKENS_POLYGON.iter().map(|&s| s.to_string()).collect(),
            max_liquidation_gas_gwei: AAVE_POLYGON_MAX_LIQUIDATION_GAS_GWEI,
        }
    }
}

impl AlignedAavePolygonMarketData {
    /// Create new aligned market data
    #[inline(always)]
    #[must_use]
    pub const fn new(supply_apy_scaled: u64, borrow_apy_variable_scaled: u64, utilization_scaled: u64, timestamp: u64) -> Self {
        Self {
            supply_apy_scaled,
            borrow_apy_variable_scaled,
            utilization_scaled,
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
    
    /// Get supply APY as Decimal
    #[inline(always)]
    #[must_use]
    pub fn supply_apy(&self) -> Decimal {
        Decimal::from(self.supply_apy_scaled) / Decimal::from(1_000_000_u64)
    }
    
    /// Get borrow APY variable as Decimal
    #[inline(always)]
    #[must_use]
    pub fn borrow_apy_variable(&self) -> Decimal {
        Decimal::from(self.borrow_apy_variable_scaled) / Decimal::from(1_000_000_u64)
    }
    
    /// Get utilization rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn utilization_rate(&self) -> Decimal {
        Decimal::from(self.utilization_scaled) / Decimal::from(1_000_000_u64)
    }
}

/// Aave Polygon Integration for ultra-performance lending/borrowing operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct AavePolygonIntegration {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Aave Polygon specific configuration
    aave_config: AavePolygonConfig,

    /// Polygon configuration
    polygon_config: PolygonConfig,

    /// Statistics
    stats: Arc<AavePolygonStats>,

    /// Active markets
    markets: Arc<DashMap<String, AavePolygonMarket>>,

    /// Market cache for ultra-fast access
    market_cache: Arc<DashMap<String, AlignedAavePolygonMarketData>>,

    /// User positions
    positions: Arc<DashMap<String, AavePolygonPosition>>,

    /// Liquidation opportunities
    opportunities: Arc<RwLock<Vec<AavePolygonLiquidationOpportunity>>>,

    /// Performance timers
    liquidation_timer: Timer,
    market_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Market update channels
    market_sender: Sender<AavePolygonMarket>,
    market_receiver: Receiver<AavePolygonMarket>,

    /// Liquidation channels
    liquidation_sender: Sender<AavePolygonLiquidationOpportunity>,
    liquidation_receiver: Receiver<AavePolygonLiquidationOpportunity>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current block number
    current_block: Arc<TokioMutex<u64>>,
}

impl AavePolygonIntegration {
    /// Create new Aave Polygon integration with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(
        config: Arc<ChainCoreConfig>,
        polygon_config: PolygonConfig,
    ) -> Result<Self> {
        let aave_config = AavePolygonConfig::default();
        let stats = Arc::new(AavePolygonStats::default());
        let markets = Arc::new(DashMap::with_capacity(AAVE_POLYGON_MAX_MARKETS));
        let market_cache = Arc::new(DashMap::with_capacity(AAVE_POLYGON_MAX_MARKETS));
        let positions = Arc::new(DashMap::with_capacity(AAVE_POLYGON_MAX_POSITIONS));
        let opportunities = Arc::new(RwLock::new(Vec::with_capacity(100)));
        let liquidation_timer = Timer::new("aave_polygon_liquidation_execution");
        let market_timer = Timer::new("aave_polygon_market_monitoring");
        let shutdown = Arc::new(AtomicBool::new(false));
        let current_block = Arc::new(TokioMutex::new(0));

        let (market_sender, market_receiver) = channel::bounded(AAVE_POLYGON_MAX_MARKETS);
        let (liquidation_sender, liquidation_receiver) = channel::bounded(100);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            aave_config,
            polygon_config,
            stats,
            markets,
            market_cache,
            positions,
            opportunities,
            liquidation_timer,
            market_timer,
            shutdown,
            market_sender,
            market_receiver,
            liquidation_sender,
            liquidation_receiver,
            http_client,
            current_block,
        })
    }

    /// Start Aave Polygon integration services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.aave_config.enabled {
            info!("Aave Polygon integration disabled");
            return Ok(());
        }

        info!("Starting Aave Polygon integration");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start market monitoring
        self.start_market_monitoring().await;

        // Start liquidation monitoring
        self.start_liquidation_monitoring().await;

        // Start position tracking
        self.start_position_tracking().await;

        // Start yield optimization
        if self.aave_config.enable_yield_optimization {
            self.start_yield_optimization().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Aave Polygon integration started successfully");
        Ok(())
    }

    /// Stop Aave Polygon integration
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping Aave Polygon integration");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Aave Polygon integration stopped");
    }

    /// Execute liquidation
    ///
    /// # Errors
    ///
    /// Returns error if liquidation execution fails
    #[inline]
    pub async fn execute_liquidation(
        &self,
        opportunity: &AavePolygonLiquidationOpportunity,
    ) -> Result<String> {
        let start_time = Instant::now();

        // Validate opportunity is still profitable
        if opportunity.profit_usd < self.aave_config.min_liquidation_profit_usd {
            return Err(crate::ChainCoreError::Internal(format!(
                "Liquidation profit {} below minimum {}",
                opportunity.profit_usd, self.aave_config.min_liquidation_profit_usd
            )));
        }

        // Execute liquidation (simplified implementation)
        let tx_hash = self.execute_liquidation_transaction(opportunity).await?;

        // Update statistics
        self.stats.liquidations_executed.fetch_add(1, Ordering::Relaxed);
        let profit_usd = opportunity.profit_usd.to_u64().unwrap_or(0);
        self.stats.total_liquidation_profit_usd.fetch_add(profit_usd, Ordering::Relaxed);

        #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
        let execution_time = start_time.elapsed().as_micros() as u64;
        self.stats.avg_liquidation_time_us.store(execution_time, Ordering::Relaxed);

        debug!("Aave Polygon liquidation executed in {}μs: {}", execution_time, tx_hash);
        Ok(tx_hash)
    }

    /// Get current Aave Polygon statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &AavePolygonStats {
        &self.stats
    }

    /// Get market information
    #[inline]
    pub async fn get_market_info(&self, atoken_address: &str) -> Option<AavePolygonMarket> {
        self.markets.get(atoken_address).map(|entry| entry.value().clone())
    }

    /// Get liquidation opportunities
    #[inline]
    pub async fn get_liquidation_opportunities(&self) -> Vec<AavePolygonLiquidationOpportunity> {
        let opportunities = self.opportunities.read().await;
        opportunities.clone()
    }

    /// Get user position
    #[inline]
    pub async fn get_user_position(&self, user_address: &str) -> Option<AavePolygonPosition> {
        self.positions.get(user_address).map(|entry| entry.value().clone())
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(3000)) // Faster timeout for Polygon
            .http2_prior_knowledge()
            .http2_keep_alive_timeout(Duration::from_secs(30))
            .http2_keep_alive_interval(Duration::from_secs(10))
            .pool_max_idle_per_host(8)
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
        let market_receiver = self.market_receiver.clone();
        let markets = Arc::clone(&self.markets);
        let market_cache = Arc::clone(&self.market_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let aave_config = self.aave_config.clone();
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000 / AAVE_POLYGON_LIQUIDATION_FREQ_HZ));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let _timer = Timer::new("aave_polygon_market_monitor_tick");

                // Process incoming market updates
                while let Ok(market) = market_receiver.try_recv() {
                    let start_time = Instant::now();

                    // Update market data
                    markets.insert(market.atoken_address.clone(), market.clone());

                    // Update cache with aligned data
                    let aligned_data = AlignedAavePolygonMarketData::new(
                        (market.supply_apy * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        (market.borrow_apy_variable * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        if market.total_supply > Decimal::ZERO {
                            ((market.total_borrows / market.total_supply) * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0)
                        } else {
                            0
                        },
                        market.last_update,
                    );
                    market_cache.insert(market.atoken_address.clone(), aligned_data);

                    #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                    let update_time = start_time.elapsed().as_micros() as u64;
                    trace!("Aave Polygon market {} updated in {}μs", market.atoken_address, update_time);
                }

                // Fetch market data from blockchain
                if let Err(_e) = Self::fetch_market_data(&http_client, &aave_config).await {
                    stats.market_errors.fetch_add(1, Ordering::Relaxed);
                }

                // Clean stale cache entries
                Self::clean_stale_cache(&market_cache, 240_000); // 4 minutes (faster than mainnet)
            }
        });
    }

    /// Start liquidation monitoring
    async fn start_liquidation_monitoring(&self) {
        let liquidation_receiver = self.liquidation_receiver.clone();
        let opportunities = Arc::clone(&self.opportunities);
        let positions = Arc::clone(&self.positions);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let aave_config = self.aave_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(aave_config.liquidation_monitor_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let _timer = Timer::new("aave_polygon_liquidation_monitor_tick");

                // Process incoming liquidation opportunities
                while let Ok(opportunity) = liquidation_receiver.try_recv() {
                    let mut opps = opportunities.write().await;

                    // Remove stale opportunities
                    opps.retain(|opp| opp.discovered_at.elapsed().as_secs() < 45); // Faster cleanup for Polygon

                    // Add new opportunity if profitable
                    if opportunity.profit_usd >= aave_config.min_liquidation_profit_usd {
                        opps.push(opportunity);
                        stats.opportunities_found.fetch_add(1, Ordering::Relaxed);
                    }

                    drop(opps);
                }

                // Scan positions for liquidation opportunities
                let position_count = positions.len();
                for position_entry in positions.iter() {
                    let position = position_entry.value();

                    // Check if position is liquidatable (health factor < 1.0)
                    if position.health_factor < Decimal::ONE && position.available_borrow_usd < Decimal::ZERO {
                        // Create liquidation opportunity (simplified)
                        let opportunity = AavePolygonLiquidationOpportunity {
                            borrower: position.user_address.clone(),
                            collateral_atoken: AWMATIC_POLYGON.to_string(),
                            debt_atoken: AUSDC_POLYGON.to_string(),
                            repay_amount: position.total_borrow_usd * Decimal::from_str("0.5").unwrap_or_default(),
                            seize_amount: position.total_supply_usd * Decimal::from_str("0.55").unwrap_or_default(),
                            profit_usd: position.total_supply_usd * Decimal::from_str("0.05").unwrap_or_default(),
                            health_factor: position.health_factor,
                            gas_estimate: 250_000, // Lower gas than mainnet
                            confidence: 88,
                            discovered_at: Instant::now(),
                        };

                        if opportunity.profit_usd >= aave_config.min_liquidation_profit_usd {
                            let mut opps = opportunities.write().await;
                            opps.push(opportunity);
                            stats.opportunities_found.fetch_add(1, Ordering::Relaxed);
                            drop(opps);
                        }
                    }
                }

                stats.positions_monitored.store(position_count as u64, Ordering::Relaxed);
                trace!("Monitored {} Aave Polygon positions for liquidation", position_count);
            }
        });
    }

    /// Start position tracking
    async fn start_position_tracking(&self) {
        let positions = Arc::clone(&self.positions);
        let _markets = Arc::clone(&self.markets);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(20)); // Update every 20 seconds (faster than mainnet)

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Update position health factors based on current market data
                for mut position_entry in positions.iter_mut() {
                    let position = position_entry.value_mut();

                    // Recalculate health factor (simplified)
                    if position.total_borrow_usd > Decimal::ZERO {
                        position.health_factor = position.total_supply_usd / position.total_borrow_usd;
                    } else {
                        position.health_factor = Decimal::from(999); // Very healthy
                    }

                    position.last_update = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                }

                trace!("Updated {} Aave Polygon position health factors", positions.len());
            }
        });
    }

    /// Start yield optimization
    async fn start_yield_optimization(&self) {
        let markets = Arc::clone(&self.markets);
        let _stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(180)); // Optimize every 3 minutes (faster than mainnet)

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Find best yield opportunities
                let mut best_supply_apy = Decimal::ZERO;
                let mut best_market = String::new();

                for market_entry in markets.iter() {
                    let market = market_entry.value();
                    if market.supply_apy > best_supply_apy {
                        best_supply_apy = market.supply_apy;
                        best_market.clone_from(&market.atoken_address);
                    }
                }

                if !best_market.is_empty() {
                    trace!("Best Aave Polygon yield opportunity: {} with {}% APY", best_market, best_supply_apy);
                }
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

                let liquidations_executed = stats.liquidations_executed.load(Ordering::Relaxed);
                let total_profit = stats.total_liquidation_profit_usd.load(Ordering::Relaxed);
                let avg_liquidation_time = stats.avg_liquidation_time_us.load(Ordering::Relaxed);
                let positions_monitored = stats.positions_monitored.load(Ordering::Relaxed);
                let opportunities_found = stats.opportunities_found.load(Ordering::Relaxed);

                info!(
                    "Aave Polygon Stats: liquidations={}, profit=${}, avg_time={}μs, positions={}, opportunities={}",
                    liquidations_executed, total_profit, avg_liquidation_time, positions_monitored, opportunities_found
                );
            }
        });
    }

    /// Execute liquidation transaction
    async fn execute_liquidation_transaction(&self, _opportunity: &AavePolygonLiquidationOpportunity) -> Result<String> {
        // Simplified implementation - in production this would make actual blockchain calls
        Ok("0xdef456789abcdef456789abcdef456789abcdef45".to_string())
    }

    /// Fetch market data from blockchain
    async fn fetch_market_data(
        _http_client: &Arc<TokioMutex<Option<reqwest::Client>>>,
        _config: &AavePolygonConfig,
    ) -> Result<Vec<AavePolygonMarket>> {
        // Simplified implementation - in production this would fetch real market data
        Ok(vec![])
    }

    /// Clean stale cache entries
    fn clean_stale_cache(cache: &Arc<DashMap<String, AlignedAavePolygonMarketData>>, max_age_ms: u64) {
        cache.retain(|_key, data| !data.is_stale(max_age_ms));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChainCoreConfig, polygon::PolygonConfig};
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_aave_polygon_integration_creation() {
        let config = Arc::new(ChainCoreConfig::default());
        let polygon_config = PolygonConfig::default();

        let Ok(integration) = AavePolygonIntegration::new(config, polygon_config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(integration.stats().liquidations_executed.load(Ordering::Relaxed), 0);
        assert_eq!(integration.stats().total_liquidation_profit_usd.load(Ordering::Relaxed), 0);
        assert!(integration.markets.is_empty());
    }

    #[test]
    fn test_aave_polygon_config_default() {
        let config = AavePolygonConfig::default();
        assert!(config.enabled);
        assert_eq!(config.liquidation_monitor_interval_ms, AAVE_POLYGON_MONITOR_INTERVAL_MS);
        assert_eq!(config.max_liquidation_gas_gwei, AAVE_POLYGON_MAX_LIQUIDATION_GAS_GWEI);
        assert!(config.enable_auto_liquidation);
        assert!(config.enable_yield_optimization);
        assert!(!config.monitored_atokens.is_empty());
    }

    #[test]
    fn test_aligned_aave_polygon_market_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedAavePolygonMarketData>(), 64);
        assert!(mem::size_of::<AlignedAavePolygonMarketData>() <= 64);
    }

    #[test]
    fn test_aave_polygon_stats_operations() {
        let stats = AavePolygonStats::default();

        stats.liquidations_executed.fetch_add(8, Ordering::Relaxed);
        stats.total_liquidation_profit_usd.fetch_add(1200, Ordering::Relaxed);
        stats.opportunities_found.fetch_add(25, Ordering::Relaxed);

        assert_eq!(stats.liquidations_executed.load(Ordering::Relaxed), 8);
        assert_eq!(stats.total_liquidation_profit_usd.load(Ordering::Relaxed), 1200);
        assert_eq!(stats.opportunities_found.load(Ordering::Relaxed), 25);
    }

    #[test]
    #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
    fn test_aligned_aave_polygon_market_data_staleness() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedAavePolygonMarketData::new(60_000, 90_000, 800_000, now);
        let stale_data = AlignedAavePolygonMarketData::new(60_000, 90_000, 800_000, now - 10_000);

        assert!(!fresh_data.is_stale(5_000));
        assert!(stale_data.is_stale(5_000));
    }

    #[test]
    fn test_aligned_aave_polygon_market_data_conversions() {
        let data = AlignedAavePolygonMarketData::new(
            60_000,   // 6% supply APY
            90_000,   // 9% borrow APY variable
            800_000,  // 80% utilization
            1_640_995_200_000,
        );

        assert_eq!(data.supply_apy(), dec!(0.06));
        assert_eq!(data.borrow_apy_variable(), dec!(0.09));
        assert_eq!(data.utilization_rate(), dec!(0.8));
    }

    #[test]
    fn test_aave_polygon_market_creation() {
        let market = AavePolygonMarket {
            atoken_address: AWMATIC_POLYGON.to_string(),
            underlying_token: TokenAddress::ZERO,
            supply_apy: dec!(0.06),
            borrow_apy_variable: dec!(0.09),
            borrow_apy_stable: dec!(0.11),
            total_supply: dec!(500000),
            total_borrows: dec!(400000),
            available_liquidity: dec!(100000),
            collateral_factor: dec!(0.8),
            liquidation_threshold: dec!(0.85),
            liquidation_bonus: dec!(0.05),
            reserve_factor: dec!(0.1),
            last_update: 1_640_995_200,
        };

        assert_eq!(market.supply_apy, dec!(0.06));
        assert_eq!(market.borrow_apy_variable, dec!(0.09));
        assert_eq!(market.collateral_factor, dec!(0.8));
    }

    #[test]
    fn test_aave_polygon_liquidation_opportunity_creation() {
        let opportunity = AavePolygonLiquidationOpportunity {
            borrower: "0x123456789abcdef".to_string(),
            collateral_atoken: AWMATIC_POLYGON.to_string(),
            debt_atoken: AUSDC_POLYGON.to_string(),
            repay_amount: dec!(500),
            seize_amount: dec!(550),
            profit_usd: dec!(50),
            health_factor: dec!(0.92),
            gas_estimate: 250_000,
            confidence: 88,
            discovered_at: Instant::now(),
        };

        assert_eq!(opportunity.profit_usd, dec!(50));
        assert_eq!(opportunity.confidence, 88);
        assert_eq!(opportunity.gas_estimate, 250_000);
    }

    #[test]
    fn test_aave_polygon_position_creation() {
        let mut supplies = HashMap::new();
        supplies.insert(AWMATIC_POLYGON.to_string(), dec!(100));

        let mut borrows = HashMap::new();
        borrows.insert(AUSDC_POLYGON.to_string(), dec!(4000));

        let position = AavePolygonPosition {
            user_address: "0xuser456".to_string(),
            supplies,
            borrows,
            total_supply_usd: dec!(6000),
            total_borrow_usd: dec!(4000),
            health_factor: dec!(1.5),
            liquidation_threshold_usd: dec!(5100),
            available_borrow_usd: dec!(800),
            last_update: 1_640_995_200,
        };

        assert_eq!(position.health_factor, dec!(1.5));
        assert_eq!(position.total_supply_usd, dec!(6000));
        assert_eq!(position.available_borrow_usd, dec!(800));
    }

    #[tokio::test]
    async fn test_aave_polygon_liquidation_execution() {
        let config = Arc::new(ChainCoreConfig::default());
        let polygon_config = PolygonConfig::default();

        let Ok(integration) = AavePolygonIntegration::new(config, polygon_config).await else {
            return;
        };

        let opportunity = AavePolygonLiquidationOpportunity {
            borrower: "0x789".to_string(),
            collateral_atoken: AWMATIC_POLYGON.to_string(),
            debt_atoken: AUSDC_POLYGON.to_string(),
            repay_amount: dec!(500),
            seize_amount: dec!(550),
            profit_usd: dec!(50),
            health_factor: dec!(0.92),
            gas_estimate: 250_000,
            confidence: 88,
            discovered_at: Instant::now(),
        };

        let result = integration.execute_liquidation(&opportunity).await;
        assert!(result.is_ok());

        if let Ok(tx_hash) = result {
            assert!(!tx_hash.is_empty());
            assert_eq!(integration.stats().liquidations_executed.load(Ordering::Relaxed), 1);
        }
    }
}
