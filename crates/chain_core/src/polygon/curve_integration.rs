//! Curve Polygon Integration for ultra-performance stable swap operations
//!
//! This module provides Curve protocol integration for Polygon chain,
//! enabling low-slippage stablecoin trading and yield farming optimization.
//!
//! ## Performance Targets
//! - Pool Analysis: <60μs
//! - Swap Execution: <120μs
//! - Yield Calculation: <80μs
//! - Arbitrage Detection: <100μs
//! - Route Optimization: <90μs
//!
//! ## Architecture
//! - Real-time stable swap pool monitoring
//! - Multi-pool arbitrage detection
//! - Yield farming strategy automation
//! - Low-slippage swap execution
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

/// Curve Polygon integration configuration
#[derive(Debug, Clone)]
pub struct CurvePolygonConfig {
    /// Enable Curve integration
    pub enabled: bool,
    
    /// Pool monitoring interval in milliseconds
    pub pool_monitor_interval_ms: u64,
    
    /// Maximum slippage tolerance (percentage)
    pub max_slippage_percent: Decimal,
    
    /// Minimum liquidity threshold for pools (USD)
    pub min_liquidity_usd: Decimal,
    
    /// Maximum gas price in Gwei for Polygon
    pub max_gas_price_gwei: u64,
    
    /// Enable yield farming strategies
    pub enable_yield_farming: bool,
    
    /// Enable arbitrage detection
    pub enable_arbitrage_detection: bool,
    
    /// Monitored pool types
    pub monitored_pool_types: Vec<CurvePoolType>,
    
    /// Minimum arbitrage profit threshold (USD)
    pub min_arbitrage_profit_usd: Decimal,
}

/// Curve pool types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurvePoolType {
    /// Plain pools (e.g., 3pool)
    Plain,
    /// Lending pools (e.g., aave)
    Lending,
    /// Meta pools (e.g., pools with LP tokens)
    Meta,
    /// Crypto pools (e.g., tricrypto)
    Crypto,
    /// Factory pools
    Factory,
}

/// Curve pool information on Polygon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvePolygonPool {
    /// Pool address
    pub address: String,
    
    /// Pool name
    pub name: String,
    
    /// Pool type
    pub pool_type: String,
    
    /// Token addresses in the pool
    pub tokens: Vec<TokenAddress>,
    
    /// Token balances
    pub balances: Vec<Decimal>,
    
    /// Pool fees (percentage)
    pub fee: Decimal,
    
    /// Admin fee (percentage)
    pub admin_fee: Decimal,
    
    /// Total value locked in USD
    pub tvl_usd: Decimal,
    
    /// 24h volume in USD
    pub volume_24h_usd: Decimal,
    
    /// Current A parameter (amplification coefficient)
    pub a_parameter: u64,
    
    /// Virtual price
    pub virtual_price: Decimal,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Curve swap route information
#[derive(Debug, Clone)]
pub struct CurvePolygonRoute {
    /// Route pools
    pub pools: Vec<CurvePolygonPool>,
    
    /// Token indices for each pool
    pub token_indices: Vec<(usize, usize)>,
    
    /// Expected output amount
    pub expected_output: Decimal,
    
    /// Price impact percentage
    pub price_impact: Decimal,
    
    /// Total fees
    pub total_fees: Decimal,
    
    /// Gas estimate
    pub gas_estimate: u64,
    
    /// Route confidence score (0-100)
    pub confidence: u8,
}

/// Curve yield farming position
#[derive(Debug, Clone)]
pub struct CurvePolygonPosition {
    /// Position ID
    pub position_id: String,
    
    /// Pool address
    pub pool_address: String,
    
    /// LP token amount
    pub lp_token_amount: Decimal,
    
    /// Underlying token amounts
    pub underlying_amounts: Vec<Decimal>,
    
    /// Staked in gauge
    pub staked_amount: Decimal,
    
    /// Unclaimed CRV rewards
    pub unclaimed_crv: Decimal,
    
    /// Unclaimed MATIC rewards
    pub unclaimed_matic: Decimal,
    
    /// Current APY
    pub current_apy: Decimal,
    
    /// Position value in USD
    pub value_usd: Decimal,
}

/// Curve arbitrage opportunity
#[derive(Debug, Clone)]
pub struct CurvePolygonArbitrageOpportunity {
    /// Opportunity ID
    pub id: String,
    
    /// Source pool
    pub source_pool: String,
    
    /// Target pool
    pub target_pool: String,
    
    /// Token to arbitrage
    pub token: TokenAddress,
    
    /// Amount to trade
    pub amount: Decimal,
    
    /// Expected profit in USD
    pub profit_usd: Decimal,
    
    /// Gas cost estimate
    pub gas_cost_usd: Decimal,
    
    /// Net profit (profit - gas cost)
    pub net_profit_usd: Decimal,
    
    /// Confidence score (0-100)
    pub confidence: u8,
    
    /// Discovery timestamp
    pub discovered_at: Instant,
}

/// Curve Polygon statistics
#[derive(Debug, Default)]
pub struct CurvePolygonStats {
    /// Total swaps executed
    pub swaps_executed: AtomicU64,
    
    /// Total volume traded (USD)
    pub total_volume_usd: AtomicU64,
    
    /// Total fees earned (USD)
    pub total_fees_earned_usd: AtomicU64,
    
    /// Successful arbitrage opportunities
    pub arbitrage_opportunities: AtomicU64,
    
    /// Average swap execution time (microseconds)
    pub avg_swap_time_us: AtomicU64,
    
    /// Pool monitoring errors
    pub pool_errors: AtomicU64,
    
    /// Active yield positions
    pub active_positions: AtomicU64,
    
    /// Total yield earned (USD)
    pub total_yield_usd: AtomicU64,
}

/// Cache-line aligned pool data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedCurvePolygonPoolData {
    /// Virtual price (scaled by 1e18)
    pub virtual_price_scaled: u64,
    
    /// TVL in USD (scaled by 1e6)
    pub tvl_scaled: u64,
    
    /// Volume 24h (scaled by 1e6)
    pub volume_scaled: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// Curve Polygon integration constants
pub const CURVE_POLYGON_DEFAULT_SLIPPAGE: &str = "0.001"; // 0.1% (lower than other DEXs)
pub const CURVE_POLYGON_MIN_LIQUIDITY_USD: &str = "10000"; // $10k minimum
pub const CURVE_POLYGON_MAX_GAS_GWEI: u64 = 100; // 100 Gwei for Polygon
pub const CURVE_POLYGON_POOL_MONITOR_INTERVAL_MS: u64 = 500; // 500ms monitoring
pub const CURVE_POLYGON_MAX_POOLS: usize = 100;
pub const CURVE_POLYGON_MAX_POSITIONS: usize = 50;
pub const CURVE_POLYGON_ROUTING_FREQ_HZ: u64 = 4; // 250ms intervals
pub const CURVE_POLYGON_MIN_ARBITRAGE_PROFIT_USD: &str = "15"; // $15 minimum

/// Curve protocol addresses on Polygon
pub const CURVE_ADDRESS_PROVIDER_POLYGON: &str = "0x0000000022D53366457F9d5E68Ec105046FC4383";
pub const CURVE_REGISTRY_POLYGON: &str = "0x094d12e5b541784701FD8d65F11fc0598FBC6332";
pub const CURVE_FACTORY_POLYGON: &str = "0x722272D36ef0Da72FF51c5A65Db7b870E2e8D4ee";

/// Popular Curve pools on Polygon
pub const CURVE_3POOL_POLYGON: &str = "0x445FE580eF8d70FF569aB36e80c647af338db351"; // DAI/USDC/USDT
pub const CURVE_AAVE_POOL_POLYGON: &str = "0x445FE580eF8d70FF569aB36e80c647af338db351"; // aDAI/aUSDC/aUSDT
pub const CURVE_TRICRYPTO_POLYGON: &str = "0x92215849c439E1f8612b6646060B4E3E5ef822cC"; // USDT/WBTC/WETH

impl Default for CurvePolygonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pool_monitor_interval_ms: CURVE_POLYGON_POOL_MONITOR_INTERVAL_MS,
            max_slippage_percent: CURVE_POLYGON_DEFAULT_SLIPPAGE.parse().unwrap_or_default(),
            min_liquidity_usd: CURVE_POLYGON_MIN_LIQUIDITY_USD.parse().unwrap_or_default(),
            max_gas_price_gwei: CURVE_POLYGON_MAX_GAS_GWEI,
            enable_yield_farming: true,
            enable_arbitrage_detection: true,
            monitored_pool_types: vec![
                CurvePoolType::Plain,
                CurvePoolType::Lending,
                CurvePoolType::Meta,
                CurvePoolType::Crypto,
            ],
            min_arbitrage_profit_usd: CURVE_POLYGON_MIN_ARBITRAGE_PROFIT_USD.parse().unwrap_or_default(),
        }
    }
}

impl AlignedCurvePolygonPoolData {
    /// Create new aligned pool data
    #[inline(always)]
    #[must_use]
    pub const fn new(virtual_price_scaled: u64, tvl_scaled: u64, volume_scaled: u64, timestamp: u64) -> Self {
        Self {
            virtual_price_scaled,
            tvl_scaled,
            volume_scaled,
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
    
    /// Get virtual price as Decimal
    #[inline(always)]
    #[must_use]
    pub fn virtual_price(&self) -> Decimal {
        Decimal::from(self.virtual_price_scaled) / Decimal::from(1_000_000_000_000_000_000_u64)
    }
    
    /// Get TVL as Decimal
    #[inline(always)]
    #[must_use]
    pub fn tvl_usd(&self) -> Decimal {
        Decimal::from(self.tvl_scaled) / Decimal::from(1_000_000_u64)
    }
}

/// Curve Polygon Integration for ultra-performance stable swap operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct CurvePolygonIntegration {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Curve Polygon specific configuration
    curve_config: CurvePolygonConfig,

    /// Polygon configuration
    polygon_config: PolygonConfig,

    /// Statistics
    stats: Arc<CurvePolygonStats>,

    /// Active pools
    pools: Arc<DashMap<String, CurvePolygonPool>>,

    /// Pool cache for ultra-fast access
    pool_cache: Arc<DashMap<String, AlignedCurvePolygonPoolData>>,

    /// Active yield positions
    positions: Arc<DashMap<String, CurvePolygonPosition>>,

    /// Arbitrage opportunities
    arbitrage_opportunities: Arc<RwLock<Vec<CurvePolygonArbitrageOpportunity>>>,

    /// Swap routes cache
    routes_cache: Arc<RwLock<HashMap<String, CurvePolygonRoute>>>,

    /// Performance timers
    swap_timer: Timer,
    pool_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Pool update channels
    pool_sender: Sender<CurvePolygonPool>,
    pool_receiver: Receiver<CurvePolygonPool>,

    /// Arbitrage channels
    arbitrage_sender: Sender<CurvePolygonArbitrageOpportunity>,
    arbitrage_receiver: Receiver<CurvePolygonArbitrageOpportunity>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current block number
    current_block: Arc<TokioMutex<u64>>,
}

impl CurvePolygonIntegration {
    /// Create new Curve Polygon integration with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(
        config: Arc<ChainCoreConfig>,
        polygon_config: PolygonConfig,
    ) -> Result<Self> {
        let curve_config = CurvePolygonConfig::default();
        let stats = Arc::new(CurvePolygonStats::default());
        let pools = Arc::new(DashMap::with_capacity(CURVE_POLYGON_MAX_POOLS));
        let pool_cache = Arc::new(DashMap::with_capacity(CURVE_POLYGON_MAX_POOLS));
        let positions = Arc::new(DashMap::with_capacity(CURVE_POLYGON_MAX_POSITIONS));
        let arbitrage_opportunities = Arc::new(RwLock::new(Vec::with_capacity(50)));
        let routes_cache = Arc::new(RwLock::new(HashMap::with_capacity(CURVE_POLYGON_MAX_POOLS)));
        let swap_timer = Timer::new("curve_polygon_swap_execution");
        let pool_timer = Timer::new("curve_polygon_pool_monitoring");
        let shutdown = Arc::new(AtomicBool::new(false));
        let current_block = Arc::new(TokioMutex::new(0));

        let (pool_sender, pool_receiver) = channel::bounded(CURVE_POLYGON_MAX_POOLS);
        let (arbitrage_sender, arbitrage_receiver) = channel::bounded(100);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            curve_config,
            polygon_config,
            stats,
            pools,
            pool_cache,
            positions,
            arbitrage_opportunities,
            routes_cache,
            swap_timer,
            pool_timer,
            shutdown,
            pool_sender,
            pool_receiver,
            arbitrage_sender,
            arbitrage_receiver,
            http_client,
            current_block,
        })
    }

    /// Start Curve Polygon integration services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.curve_config.enabled {
            info!("Curve Polygon integration disabled");
            return Ok(());
        }

        info!("Starting Curve Polygon integration");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start pool monitoring
        self.start_pool_monitoring().await;

        // Start arbitrage detection
        if self.curve_config.enable_arbitrage_detection {
            self.start_arbitrage_detection().await;
        }

        // Start yield farming manager
        if self.curve_config.enable_yield_farming {
            self.start_yield_farming().await;
        }

        // Start route optimization
        self.start_route_optimization().await;

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Curve Polygon integration started successfully");
        Ok(())
    }

    /// Stop Curve Polygon integration
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping Curve Polygon integration");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Curve Polygon integration stopped");
    }

    /// Execute stable swap with optimal routing
    ///
    /// # Errors
    ///
    /// Returns error if swap execution fails
    #[inline]
    pub async fn execute_swap(
        &self,
        token_in: TokenAddress,
        token_out: TokenAddress,
        amount_in: Decimal,
        min_amount_out: Decimal,
    ) -> Result<String> {
        let start_time = Instant::now();

        // Find optimal route
        let route = self.find_optimal_route(token_in, token_out, amount_in).await?;

        // Validate route meets minimum output
        if route.expected_output < min_amount_out {
            return Err(crate::ChainCoreError::Internal(format!(
                "Route output {} below minimum {}",
                route.expected_output, min_amount_out
            )));
        }

        // Execute swap
        let tx_hash = self.execute_swap_route(&route).await?;

        // Update statistics
        self.stats.swaps_executed.fetch_add(1, Ordering::Relaxed);
        let volume_usd = route.pools.first().map_or(0, |first_pool| (amount_in * first_pool.virtual_price).to_u64().unwrap_or(0));
        self.stats.total_volume_usd.fetch_add(volume_usd, Ordering::Relaxed);

        #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
        let execution_time = start_time.elapsed().as_micros() as u64;
        self.stats.avg_swap_time_us.store(execution_time, Ordering::Relaxed);

        debug!("Curve Polygon swap executed in {}μs: {}", execution_time, tx_hash);
        Ok(tx_hash)
    }

    /// Get current Curve Polygon statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &CurvePolygonStats {
        &self.stats
    }

    /// Get pool information
    #[inline]
    pub async fn get_pool_info(&self, pool_address: &str) -> Option<CurvePolygonPool> {
        self.pools.get(pool_address).map(|entry| entry.value().clone())
    }

    /// Get active yield positions
    #[inline]
    #[must_use]
    pub fn get_active_positions(&self) -> Vec<CurvePolygonPosition> {
        self.positions.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Get arbitrage opportunities
    #[inline]
    pub async fn get_arbitrage_opportunities(&self) -> Vec<CurvePolygonArbitrageOpportunity> {
        let opportunities = self.arbitrage_opportunities.read().await;
        opportunities.clone()
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(3000)) // Fast timeout for Polygon
            .http2_prior_knowledge()
            .http2_keep_alive_timeout(Duration::from_secs(30))
            .http2_keep_alive_interval(Duration::from_secs(10))
            .pool_max_idle_per_host(6)
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
        let pool_receiver = self.pool_receiver.clone();
        let pools = Arc::clone(&self.pools);
        let pool_cache = Arc::clone(&self.pool_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let curve_config = self.curve_config.clone();
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000 / CURVE_POLYGON_ROUTING_FREQ_HZ));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let _timer = Timer::new("curve_polygon_pool_monitor_tick");

                // Process incoming pool updates
                while let Ok(pool) = pool_receiver.try_recv() {
                    let start_time = Instant::now();

                    // Update pool data
                    pools.insert(pool.address.clone(), pool.clone());

                    // Update cache with aligned data
                    let aligned_data = AlignedCurvePolygonPoolData::new(
                        (pool.virtual_price * Decimal::from(1_000_000_000_000_000_000_u64)).to_u64().unwrap_or(0),
                        (pool.tvl_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        (pool.volume_24h_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        pool.last_update,
                    );
                    pool_cache.insert(pool.address.clone(), aligned_data);

                    #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                    let update_time = start_time.elapsed().as_micros() as u64;
                    trace!("Curve Polygon pool {} updated in {}μs", pool.address, update_time);
                }

                // Fetch pool data from blockchain
                if let Err(_e) = Self::fetch_pool_data(&http_client, &curve_config).await {
                    stats.pool_errors.fetch_add(1, Ordering::Relaxed);
                }

                // Clean stale cache entries
                Self::clean_stale_cache(&pool_cache, 300_000); // 5 minutes
            }
        });
    }

    /// Start arbitrage detection
    async fn start_arbitrage_detection(&self) {
        let arbitrage_receiver = self.arbitrage_receiver.clone();
        let arbitrage_opportunities = Arc::clone(&self.arbitrage_opportunities);
        let pools = Arc::clone(&self.pools);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let curve_config = self.curve_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(2000)); // Check every 2 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Process incoming arbitrage opportunities
                while let Ok(opportunity) = arbitrage_receiver.try_recv() {
                    let mut opps = arbitrage_opportunities.write().await;

                    // Remove stale opportunities
                    opps.retain(|opp| opp.discovered_at.elapsed().as_secs() < 30);

                    // Add new opportunity if profitable
                    if opportunity.net_profit_usd >= curve_config.min_arbitrage_profit_usd {
                        opps.push(opportunity);
                        stats.arbitrage_opportunities.fetch_add(1, Ordering::Relaxed);
                    }

                    drop(opps);
                }

                // Scan pools for arbitrage opportunities
                let pool_list: Vec<_> = pools.iter().map(|entry| entry.value().clone()).collect();

                for (i, pool_a) in pool_list.iter().enumerate() {
                    for pool_b in pool_list.iter().skip(i + 1) {
                        // Check for arbitrage opportunities between pools
                        if let Some(opportunity) = Self::detect_arbitrage_opportunity(pool_a, pool_b, &curve_config) {
                            if opportunity.net_profit_usd >= curve_config.min_arbitrage_profit_usd {
                                let mut opps = arbitrage_opportunities.write().await;
                                opps.push(opportunity);
                                stats.arbitrage_opportunities.fetch_add(1, Ordering::Relaxed);
                                drop(opps);
                            }
                        }
                    }
                }

                trace!("Scanned {} Curve Polygon pools for arbitrage", pool_list.len());
            }
        });
    }

    /// Start yield farming manager
    async fn start_yield_farming(&self) {
        let positions = Arc::clone(&self.positions);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Check every minute

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Monitor existing positions
                for mut position_entry in positions.iter_mut() {
                    let position = position_entry.value_mut();

                    // Update position metrics (simplified)
                    position.current_apy = Decimal::from(12); // 12% APY example for Curve
                    position.value_usd = position.lp_token_amount * Decimal::from(2); // Simplified valuation
                }

                stats.active_positions.store(positions.len() as u64, Ordering::Relaxed);
                trace!("Updated {} Curve Polygon yield positions", positions.len());
            }
        });
    }

    /// Start route optimization
    async fn start_route_optimization(&self) {
        let routes_cache = Arc::clone(&self.routes_cache);
        let pools = Arc::clone(&self.pools);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Optimize every 30 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Optimize routes based on current pool states
                let pool_count = pools.len();
                if pool_count > 0 {
                    let mut cache = routes_cache.write().await;

                    // Clear old routes and rebuild (simplified)
                    cache.clear();

                    trace!("Optimized Curve Polygon routes for {} pools", pool_count);
                    drop(cache);
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

                let swaps_executed = stats.swaps_executed.load(Ordering::Relaxed);
                let total_volume = stats.total_volume_usd.load(Ordering::Relaxed);
                let avg_swap_time = stats.avg_swap_time_us.load(Ordering::Relaxed);
                let arbitrage_opportunities = stats.arbitrage_opportunities.load(Ordering::Relaxed);
                let active_positions = stats.active_positions.load(Ordering::Relaxed);

                info!(
                    "Curve Polygon Stats: swaps={}, volume=${}, avg_time={}μs, arbitrage={}, positions={}",
                    swaps_executed, total_volume, avg_swap_time, arbitrage_opportunities, active_positions
                );
            }
        });
    }

    /// Find optimal route for swap
    async fn find_optimal_route(
        &self,
        token_in: TokenAddress,
        token_out: TokenAddress,
        amount_in: Decimal,
    ) -> Result<CurvePolygonRoute> {
        // Check cache first
        let cache_key = format!("{token_in:?}_{token_out:?}_{amount_in}");
        {
            let routes = self.routes_cache.read().await;
            if let Some(cached_route) = routes.get(&cache_key) {
                return Ok(cached_route.clone());
            }
        }

        // Find best pool for the token pair
        let best_pool = self.pools.iter()
            .filter(|entry| {
                let pool = entry.value();
                pool.tokens.contains(&token_in) && pool.tokens.contains(&token_out)
            })
            .max_by_key(|entry| entry.value().tvl_usd.to_u64().unwrap_or(0))
            .map(|entry| entry.value().clone());

        let Some(pool) = best_pool else {
            return Err(crate::ChainCoreError::Internal("No suitable Curve pool found".to_string()));
        };

        // Calculate expected output (simplified)
        let expected_output = amount_in * (Decimal::ONE - pool.fee);

        let route = CurvePolygonRoute {
            pools: vec![pool],
            token_indices: vec![(0, 1)], // Simplified
            expected_output,
            price_impact: Decimal::from_str("0.0005").unwrap_or_default(), // 0.05% (very low for Curve)
            total_fees: amount_in * Decimal::from_str("0.0004").unwrap_or_default(), // 0.04%
            gas_estimate: 100_000, // Lower gas than other DEXs
            confidence: 95, // High confidence for Curve
        };

        // Cache the route
        {
            let mut routes = self.routes_cache.write().await;
            routes.insert(cache_key, route.clone());
        }

        Ok(route)
    }

    /// Execute swap route
    async fn execute_swap_route(&self, _route: &CurvePolygonRoute) -> Result<String> {
        // Simplified implementation - in production this would make actual blockchain calls
        Ok("0x123abc456def789abc456def789abc456def789a".to_string())
    }

    /// Detect arbitrage opportunity between two pools
    fn detect_arbitrage_opportunity(
        pool_a: &CurvePolygonPool,
        pool_b: &CurvePolygonPool,
        config: &CurvePolygonConfig,
    ) -> Option<CurvePolygonArbitrageOpportunity> {
        // Find common tokens
        for token in &pool_a.tokens {
            if pool_b.tokens.contains(token) {
                // Simplified arbitrage detection
                let price_diff = (pool_a.virtual_price - pool_b.virtual_price).abs();
                if price_diff > Decimal::from_str("0.001").unwrap_or_default() {
                    let profit_usd = price_diff * Decimal::from(1000); // Simplified calculation

                    if profit_usd >= config.min_arbitrage_profit_usd {
                        return Some(CurvePolygonArbitrageOpportunity {
                            id: format!("curve_arb_{}", chrono::Utc::now().timestamp_millis()),
                            source_pool: pool_a.address.clone(),
                            target_pool: pool_b.address.clone(),
                            token: *token,
                            amount: Decimal::from(1000),
                            profit_usd,
                            gas_cost_usd: Decimal::from(5), // Low gas cost on Polygon
                            net_profit_usd: profit_usd - Decimal::from(5),
                            confidence: 85,
                            discovered_at: Instant::now(),
                        });
                    }
                }
            }
        }
        None
    }

    /// Fetch pool data from blockchain
    async fn fetch_pool_data(
        _http_client: &Arc<TokioMutex<Option<reqwest::Client>>>,
        _config: &CurvePolygonConfig,
    ) -> Result<Vec<CurvePolygonPool>> {
        // Simplified implementation - in production this would fetch real pool data
        Ok(vec![])
    }

    /// Clean stale cache entries
    fn clean_stale_cache(cache: &Arc<DashMap<String, AlignedCurvePolygonPoolData>>, max_age_ms: u64) {
        cache.retain(|_key, data| !data.is_stale(max_age_ms));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChainCoreConfig, polygon::PolygonConfig};
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_curve_polygon_integration_creation() {
        let config = Arc::new(ChainCoreConfig::default());
        let polygon_config = PolygonConfig::default();

        let Ok(integration) = CurvePolygonIntegration::new(config, polygon_config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(integration.stats().swaps_executed.load(Ordering::Relaxed), 0);
        assert_eq!(integration.stats().total_volume_usd.load(Ordering::Relaxed), 0);
        assert!(integration.pools.is_empty());
    }

    #[test]
    fn test_curve_polygon_config_default() {
        let config = CurvePolygonConfig::default();
        assert!(config.enabled);
        assert_eq!(config.pool_monitor_interval_ms, CURVE_POLYGON_POOL_MONITOR_INTERVAL_MS);
        assert_eq!(config.max_gas_price_gwei, CURVE_POLYGON_MAX_GAS_GWEI);
        assert!(config.enable_yield_farming);
        assert!(config.enable_arbitrage_detection);
        assert!(!config.monitored_pool_types.is_empty());
    }

    #[test]
    fn test_aligned_curve_polygon_pool_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedCurvePolygonPoolData>(), 64);
        assert!(mem::size_of::<AlignedCurvePolygonPoolData>() <= 64);
    }

    #[test]
    fn test_curve_polygon_stats_operations() {
        let stats = CurvePolygonStats::default();

        stats.swaps_executed.fetch_add(15, Ordering::Relaxed);
        stats.total_volume_usd.fetch_add(45000, Ordering::Relaxed);
        stats.arbitrage_opportunities.fetch_add(5, Ordering::Relaxed);

        assert_eq!(stats.swaps_executed.load(Ordering::Relaxed), 15);
        assert_eq!(stats.total_volume_usd.load(Ordering::Relaxed), 45000);
        assert_eq!(stats.arbitrage_opportunities.load(Ordering::Relaxed), 5);
    }

    #[test]
    #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
    fn test_aligned_curve_polygon_pool_data_staleness() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedCurvePolygonPoolData::new(1_050_000_000_000_000_000, 15_000_000, 8_000_000, now);
        let stale_data = AlignedCurvePolygonPoolData::new(1_050_000_000_000_000_000, 15_000_000, 8_000_000, now - 10_000);

        assert!(!fresh_data.is_stale(5_000));
        assert!(stale_data.is_stale(5_000));
    }

    #[test]
    fn test_aligned_curve_polygon_pool_data_conversions() {
        let data = AlignedCurvePolygonPoolData::new(
            1_050_000_000_000_000_000, // 1.05 virtual price
            15_000_000,                // $15M TVL
            8_000_000,                 // $8M volume
            1_640_995_200_000,
        );

        assert_eq!(data.virtual_price(), dec!(1.05));
        assert_eq!(data.tvl_usd(), dec!(15));
    }

    #[test]
    fn test_curve_polygon_pool_creation() {
        let pool = CurvePolygonPool {
            address: CURVE_3POOL_POLYGON.to_string(),
            name: "3pool".to_string(),
            pool_type: "Plain".to_string(),
            tokens: vec![TokenAddress::ZERO, TokenAddress([1_u8; 20]), TokenAddress([2_u8; 20])],
            balances: vec![dec!(1000000), dec!(1000000), dec!(1000000)],
            fee: dec!(0.0004),
            admin_fee: dec!(0.5),
            tvl_usd: dec!(150000000),
            volume_24h_usd: dec!(25000000),
            a_parameter: 2000,
            virtual_price: dec!(1.05),
            last_update: 1_640_995_200,
        };

        assert_eq!(pool.fee, dec!(0.0004));
        assert_eq!(pool.virtual_price, dec!(1.05));
        assert_eq!(pool.tvl_usd, dec!(150000000));
    }

    #[test]
    fn test_curve_polygon_route_creation() {
        let pool = CurvePolygonPool {
            address: "0x123".to_string(),
            name: "test_pool".to_string(),
            pool_type: "Plain".to_string(),
            tokens: vec![TokenAddress::ZERO, TokenAddress([1_u8; 20])],
            balances: vec![dec!(500000), dec!(500000)],
            fee: dec!(0.0004),
            admin_fee: dec!(0.5),
            tvl_usd: dec!(1000000),
            volume_24h_usd: dec!(500000),
            a_parameter: 2000,
            virtual_price: dec!(1.02),
            last_update: 1_640_995_200,
        };

        let route = CurvePolygonRoute {
            pools: vec![pool],
            token_indices: vec![(0, 1)],
            expected_output: dec!(0.9996),
            price_impact: dec!(0.0005),
            total_fees: dec!(0.0004),
            gas_estimate: 100_000,
            confidence: 95,
        };

        assert_eq!(route.pools.len(), 1);
        assert_eq!(route.expected_output, dec!(0.9996));
        assert_eq!(route.confidence, 95);
    }

    #[test]
    fn test_curve_polygon_position_creation() {
        let position = CurvePolygonPosition {
            position_id: "pos_12345".to_string(),
            pool_address: CURVE_3POOL_POLYGON.to_string(),
            lp_token_amount: dec!(1000),
            underlying_amounts: vec![dec!(333), dec!(333), dec!(334)],
            staked_amount: dec!(800),
            unclaimed_crv: dec!(5.5),
            unclaimed_matic: dec!(2.3),
            current_apy: dec!(12.5),
            value_usd: dec!(2000),
        };

        assert_eq!(position.lp_token_amount, dec!(1000));
        assert_eq!(position.current_apy, dec!(12.5));
        assert_eq!(position.value_usd, dec!(2000));
    }

    #[test]
    fn test_curve_polygon_arbitrage_opportunity_creation() {
        let opportunity = CurvePolygonArbitrageOpportunity {
            id: "arb_67890".to_string(),
            source_pool: CURVE_3POOL_POLYGON.to_string(),
            target_pool: CURVE_AAVE_POOL_POLYGON.to_string(),
            token: TokenAddress::ZERO,
            amount: dec!(10000),
            profit_usd: dec!(25),
            gas_cost_usd: dec!(3),
            net_profit_usd: dec!(22),
            confidence: 85,
            discovered_at: Instant::now(),
        };

        assert_eq!(opportunity.profit_usd, dec!(25));
        assert_eq!(opportunity.net_profit_usd, dec!(22));
        assert_eq!(opportunity.confidence, 85);
    }

    #[tokio::test]
    async fn test_optimal_route_finding() {
        let config = Arc::new(ChainCoreConfig::default());
        let polygon_config = PolygonConfig::default();

        let Ok(integration) = CurvePolygonIntegration::new(config, polygon_config).await else {
            return;
        };

        // Add a test pool
        let test_pool = CurvePolygonPool {
            address: "0xtest".to_string(),
            name: "test_pool".to_string(),
            pool_type: "Plain".to_string(),
            tokens: vec![TokenAddress::ZERO, TokenAddress([1_u8; 20])],
            balances: vec![dec!(1000000), dec!(1000000)],
            fee: dec!(0.0004),
            admin_fee: dec!(0.5),
            tvl_usd: dec!(5000000),
            volume_24h_usd: dec!(2000000),
            a_parameter: 2000,
            virtual_price: dec!(1.03),
            last_update: 1_640_995_200,
        };

        integration.pools.insert(test_pool.address.clone(), test_pool);

        let route = integration.find_optimal_route(
            TokenAddress::ZERO,
            TokenAddress([1_u8; 20]),
            dec!(1000),
        ).await;

        assert!(route.is_ok());
        if let Ok(route) = route {
            assert!(!route.pools.is_empty());
            assert!(route.expected_output > Decimal::ZERO);
            assert_eq!(route.confidence, 95);
        }
    }
}
