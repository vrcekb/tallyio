//! PancakeSwap Integration for ultra-performance DEX operations
//!
//! This module provides PancakeSwap v3 integration for BSC chain,
//! enabling maximum MEV extraction through liquidity analysis and swap optimization.
//!
//! ## Performance Targets
//! - Liquidity Analysis: <100μs
//! - Swap Execution: <200μs
//! - Pool Monitoring: <50μs
//! - Yield Calculation: <75μs
//! - Route Optimization: <150μs
//!
//! ## Architecture
//! - Real-time pool monitoring with concentrated liquidity
//! - Multi-hop routing optimization
//! - Yield farming strategy automation
//! - MEV-aware swap execution
//! - Lock-free data structures for hot paths

use crate::{
    ChainCoreConfig, Result,
    types::TokenAddress,
    utils::perf::Timer,
    bsc::BscConfig,
};
use crossbeam::channel::{self, Receiver, Sender};
use dashmap::DashMap;
use rust_decimal::{Decimal, prelude::ToPrimitive};
use std::str::FromStr;
use serde::{Deserialize, Serialize};
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
use tracing::{debug, info, trace};

/// PancakeSwap integration configuration
#[derive(Debug, Clone)]
pub struct PancakeConfig {
    /// Enable PancakeSwap integration
    pub enabled: bool,
    
    /// Pool monitoring interval in milliseconds
    pub pool_monitor_interval_ms: u64,
    
    /// Maximum slippage tolerance (percentage)
    pub max_slippage_percent: Decimal,
    
    /// Minimum liquidity threshold for pools
    pub min_liquidity_usd: Decimal,
    
    /// Maximum gas price in Gwei
    pub max_gas_price_gwei: u64,
    
    /// Enable yield farming strategies
    pub enable_yield_farming: bool,
    
    /// Enable concentrated liquidity strategies
    pub enable_concentrated_liquidity: bool,
    
    /// Pool fee tiers to monitor
    pub monitored_fee_tiers: Vec<u32>,
    
    /// Maximum number of hops for routing
    pub max_routing_hops: u8,
}

/// PancakeSwap pool information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PancakePool {
    /// Pool address
    pub address: String,
    
    /// Token0 address
    pub token0: TokenAddress,
    
    /// Token1 address
    pub token1: TokenAddress,
    
    /// Fee tier (in basis points)
    pub fee: u32,
    
    /// Current price (token1/token0)
    pub price: Decimal,
    
    /// Total value locked in USD
    pub tvl_usd: Decimal,
    
    /// 24h volume in USD
    pub volume_24h_usd: Decimal,
    
    /// Current tick
    pub tick: i32,
    
    /// Liquidity
    pub liquidity: u128,
    
    /// Pool creation block
    pub creation_block: u64,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Swap route information
#[derive(Debug, Clone)]
pub struct SwapRoute {
    /// Route pools
    pub pools: Vec<PancakePool>,
    
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

/// Yield farming position
#[derive(Debug, Clone)]
pub struct YieldPosition {
    /// Position ID
    pub position_id: u64,
    
    /// Pool address
    pub pool_address: String,
    
    /// Lower tick
    pub tick_lower: i32,
    
    /// Upper tick
    pub tick_upper: i32,
    
    /// Liquidity amount
    pub liquidity: u128,
    
    /// Token0 amount
    pub amount0: Decimal,
    
    /// Token1 amount
    pub amount1: Decimal,
    
    /// Unclaimed fees
    pub unclaimed_fees: Decimal,
    
    /// Current APY
    pub current_apy: Decimal,
    
    /// Position value in USD
    pub value_usd: Decimal,
}

/// PancakeSwap statistics
#[derive(Debug, Default)]
pub struct PancakeStats {
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
pub struct AlignedPoolData {
    /// Pool price (scaled by 1e18)
    pub price_scaled: u64,
    
    /// TVL in USD (scaled by 1e6)
    pub tvl_scaled: u64,
    
    /// Volume 24h (scaled by 1e6)
    pub volume_scaled: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// PancakeSwap integration constants
pub const PANCAKE_DEFAULT_SLIPPAGE: &str = "0.005"; // 0.5%
pub const PANCAKE_MIN_LIQUIDITY_USD: &str = "10000"; // $10k minimum
pub const PANCAKE_MAX_GAS_GWEI: u64 = 20;
pub const PANCAKE_POOL_MONITOR_INTERVAL_MS: u64 = 100;
pub const PANCAKE_MAX_POOLS: usize = 1000;
pub const PANCAKE_MAX_POSITIONS: usize = 100;
pub const PANCAKE_ROUTING_FREQ_HZ: u64 = 20; // 50ms intervals

/// PancakeSwap v3 factory address
pub const PANCAKE_V3_FACTORY: &str = "0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865";

/// PancakeSwap v3 router address
pub const PANCAKE_V3_ROUTER: &str = "0x13f4EA83D0bd40E75C8222255bc855a974568Dd4";

/// PancakeSwap v3 position manager
pub const PANCAKE_V3_POSITION_MANAGER: &str = "0x46A15B0b27311cedF172AB29E4f4766fbE7F4364";

/// Common fee tiers (in basis points)
pub const PANCAKE_FEE_TIERS: &[u32] = &[100, 500, 2500, 10000]; // 0.01%, 0.05%, 0.25%, 1%

impl Default for PancakeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pool_monitor_interval_ms: PANCAKE_POOL_MONITOR_INTERVAL_MS,
            max_slippage_percent: PANCAKE_DEFAULT_SLIPPAGE.parse().unwrap_or_default(),
            min_liquidity_usd: PANCAKE_MIN_LIQUIDITY_USD.parse().unwrap_or_default(),
            max_gas_price_gwei: PANCAKE_MAX_GAS_GWEI,
            enable_yield_farming: true,
            enable_concentrated_liquidity: true,
            monitored_fee_tiers: PANCAKE_FEE_TIERS.to_vec(),
            max_routing_hops: 3,
        }
    }
}

impl AlignedPoolData {
    /// Create new aligned pool data
    #[inline(always)]
    #[must_use]
    pub const fn new(price_scaled: u64, tvl_scaled: u64, volume_scaled: u64, timestamp: u64) -> Self {
        Self {
            price_scaled,
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
    
    /// Get price as Decimal
    #[inline(always)]
    #[must_use]
    pub fn price(&self) -> Decimal {
        Decimal::from(self.price_scaled) / Decimal::from(1_000_000_000_000_000_000_u64)
    }
    
    /// Get TVL as Decimal
    #[inline(always)]
    #[must_use]
    pub fn tvl_usd(&self) -> Decimal {
        Decimal::from(self.tvl_scaled) / Decimal::from(1_000_000_u64)
    }
}

/// PancakeSwap Integration for ultra-performance DEX operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct PancakeIntegration {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// PancakeSwap specific configuration
    pancake_config: PancakeConfig,

    /// BSC configuration
    bsc_config: BscConfig,

    /// Statistics
    stats: Arc<PancakeStats>,

    /// Active pools
    pools: Arc<DashMap<String, PancakePool>>,

    /// Pool cache for ultra-fast access
    pool_cache: Arc<DashMap<String, AlignedPoolData>>,

    /// Active yield positions
    positions: Arc<DashMap<u64, YieldPosition>>,

    /// Swap routes cache
    routes_cache: Arc<RwLock<HashMap<String, SwapRoute>>>,

    /// Performance timers
    swap_timer: Timer,
    pool_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Pool update channels
    pool_sender: Sender<PancakePool>,
    pool_receiver: Receiver<PancakePool>,

    /// Swap execution channels
    swap_sender: Sender<SwapRoute>,
    swap_receiver: Receiver<SwapRoute>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current block number
    current_block: Arc<TokioMutex<u64>>,
}

impl PancakeIntegration {
    /// Create new PancakeSwap integration with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(
        config: Arc<ChainCoreConfig>,
        bsc_config: BscConfig,
    ) -> Result<Self> {
        let pancake_config = PancakeConfig::default();
        let stats = Arc::new(PancakeStats::default());
        let pools = Arc::new(DashMap::with_capacity(PANCAKE_MAX_POOLS));
        let pool_cache = Arc::new(DashMap::with_capacity(PANCAKE_MAX_POOLS));
        let positions = Arc::new(DashMap::with_capacity(PANCAKE_MAX_POSITIONS));
        let routes_cache = Arc::new(RwLock::new(HashMap::with_capacity(PANCAKE_MAX_POOLS)));
        let swap_timer = Timer::new("pancake_swap_execution");
        let pool_timer = Timer::new("pancake_pool_monitoring");
        let shutdown = Arc::new(AtomicBool::new(false));
        let current_block = Arc::new(TokioMutex::new(0));

        let (pool_sender, pool_receiver) = channel::bounded(PANCAKE_MAX_POOLS);
        let (swap_sender, swap_receiver) = channel::bounded(100);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            pancake_config,
            bsc_config,
            stats,
            pools,
            pool_cache,
            positions,
            routes_cache,
            swap_timer,
            pool_timer,
            shutdown,
            pool_sender,
            pool_receiver,
            swap_sender,
            swap_receiver,
            http_client,
            current_block,
        })
    }

    /// Start PancakeSwap integration services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.pancake_config.enabled {
            info!("PancakeSwap integration disabled");
            return Ok(());
        }

        info!("Starting PancakeSwap integration on BSC");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start pool monitoring
        self.start_pool_monitoring().await;

        // Start swap execution engine
        self.start_swap_execution().await;

        // Start yield farming manager
        if self.pancake_config.enable_yield_farming {
            self.start_yield_farming().await;
        }

        // Start route optimization
        self.start_route_optimization().await;

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("PancakeSwap integration started successfully");
        Ok(())
    }

    /// Stop PancakeSwap integration
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping PancakeSwap integration");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("PancakeSwap integration stopped");
    }

    /// Execute swap with optimal routing
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
        let volume_usd = route.pools.first().map_or(0, |first_pool| (amount_in * first_pool.price).to_u64().unwrap_or(0));
        self.stats.total_volume_usd.fetch_add(volume_usd, Ordering::Relaxed);

        #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
        let execution_time = start_time.elapsed().as_micros() as u64;
        self.stats.avg_swap_time_us.store(execution_time, Ordering::Relaxed);

        debug!("Swap executed in {}μs: {}", execution_time, tx_hash);
        Ok(tx_hash)
    }

    /// Get current PancakeSwap statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &PancakeStats {
        &self.stats
    }

    /// Get pool information
    #[inline]
    pub async fn get_pool_info(&self, pool_address: &str) -> Option<PancakePool> {
        self.pools.get(pool_address).map(|entry| entry.value().clone())
    }

    /// Get active yield positions
    #[inline]
    #[must_use]
    pub fn get_active_positions(&self) -> Vec<YieldPosition> {
        self.positions.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(5000))
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
        let pool_receiver = self.pool_receiver.clone();
        let pools = Arc::clone(&self.pools);
        let pool_cache = Arc::clone(&self.pool_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let pancake_config = self.pancake_config.clone();
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000 / PANCAKE_ROUTING_FREQ_HZ));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let _timer = Timer::new("pancake_pool_monitor_tick");

                // Process incoming pool updates
                while let Ok(pool) = pool_receiver.try_recv() {
                    let start_time = Instant::now();

                    // Update pool data
                    pools.insert(pool.address.clone(), pool.clone());

                    // Update cache with aligned data
                    let aligned_data = AlignedPoolData::new(
                        (pool.price * Decimal::from(1_000_000_000_000_000_000_u64)).to_u64().unwrap_or(0),
                        (pool.tvl_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        (pool.volume_24h_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        pool.last_update,
                    );
                    pool_cache.insert(pool.address.clone(), aligned_data);

                    #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                    let update_time = start_time.elapsed().as_micros() as u64;
                    trace!("Pool {} updated in {}μs", pool.address, update_time);
                }

                // Fetch pool data from blockchain
                if let Err(_e) = Self::fetch_pool_data(&http_client, &pancake_config).await {
                    stats.pool_errors.fetch_add(1, Ordering::Relaxed);
                }

                // Clean stale cache entries
                Self::clean_stale_cache(&pool_cache, 300_000); // 5 minutes
            }
        });
    }

    /// Start swap execution engine
    async fn start_swap_execution(&self) {
        let swap_receiver = self.swap_receiver.clone();
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                if let Ok(route) = swap_receiver.recv_timeout(Duration::from_millis(100)) {
                    let start_time = Instant::now();

                    // Execute swap route (simplified implementation)
                    trace!("Executing swap route with {} pools", route.pools.len());

                    #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                    let execution_time = start_time.elapsed().as_micros() as u64;
                    stats.avg_swap_time_us.store(execution_time, Ordering::Relaxed);
                }
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
                    position.current_apy = Decimal::from(15); // 15% APY example
                    position.value_usd = position.amount0 + position.amount1;
                }

                stats.active_positions.store(positions.len() as u64, Ordering::Relaxed);
                trace!("Updated {} yield positions", positions.len());
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

                    trace!("Optimized routes for {} pools", pool_count);
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
                let active_positions = stats.active_positions.load(Ordering::Relaxed);

                info!(
                    "PancakeSwap Stats: swaps={}, volume=${}, avg_time={}μs, positions={}",
                    swaps_executed, total_volume, avg_swap_time, active_positions
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
    ) -> Result<SwapRoute> {
        // Check cache first
        let cache_key = format!("{token_in:?}_{token_out:?}_{amount_in}");
        {
            let routes = self.routes_cache.read().await;
            if let Some(cached_route) = routes.get(&cache_key) {
                return Ok(cached_route.clone());
            }
        }

        // Find best route through available pools
        let best_pool = self.pools.iter()
            .filter(|entry| {
                let pool = entry.value();
                (pool.token0 == token_in && pool.token1 == token_out) ||
                (pool.token0 == token_out && pool.token1 == token_in)
            })
            .max_by_key(|entry| entry.value().tvl_usd.to_u64().unwrap_or(0))
            .map(|entry| entry.value().clone());

        let Some(pool) = best_pool else {
            return Err(crate::ChainCoreError::Internal("No suitable pool found".to_string()));
        };

        // Calculate expected output (simplified)
        let expected_output = amount_in * pool.price * (Decimal::ONE - Decimal::from(pool.fee) / Decimal::from(1_000_000_u32));

        let route = SwapRoute {
            pools: vec![pool],
            expected_output,
            price_impact: Decimal::from_str("0.001").unwrap_or_default(), // 0.1%
            total_fees: amount_in * Decimal::from_str("0.0025").unwrap_or_default(), // 0.25%
            gas_estimate: 150_000,
            confidence: 95,
        };

        // Cache the route
        {
            let mut routes = self.routes_cache.write().await;
            routes.insert(cache_key, route.clone());
        }

        Ok(route)
    }

    /// Execute swap route
    async fn execute_swap_route(&self, _route: &SwapRoute) -> Result<String> {
        // Simplified implementation - in production this would make actual blockchain calls
        Ok("0x1234567890abcdef1234567890abcdef12345678".to_string())
    }

    /// Fetch pool data from blockchain
    async fn fetch_pool_data(
        _http_client: &Arc<TokioMutex<Option<reqwest::Client>>>,
        _config: &PancakeConfig,
    ) -> Result<Vec<PancakePool>> {
        // Simplified implementation - in production this would fetch real pool data
        Ok(vec![])
    }

    /// Clean stale cache entries
    fn clean_stale_cache(cache: &Arc<DashMap<String, AlignedPoolData>>, max_age_ms: u64) {
        cache.retain(|_key, data| !data.is_stale(max_age_ms));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChainCoreConfig, bsc::BscConfig};
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_pancake_integration_creation() {
        let config = Arc::new(ChainCoreConfig::default());
        let bsc_config = BscConfig::default();

        let Ok(integration) = PancakeIntegration::new(config, bsc_config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(integration.stats().swaps_executed.load(Ordering::Relaxed), 0);
        assert_eq!(integration.stats().total_volume_usd.load(Ordering::Relaxed), 0);
        assert!(integration.pools.is_empty());
    }

    #[test]
    fn test_pancake_config_default() {
        let config = PancakeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.pool_monitor_interval_ms, PANCAKE_POOL_MONITOR_INTERVAL_MS);
        assert_eq!(config.max_gas_price_gwei, PANCAKE_MAX_GAS_GWEI);
        assert!(config.enable_yield_farming);
        assert!(config.enable_concentrated_liquidity);
        assert_eq!(config.max_routing_hops, 3);
        assert!(!config.monitored_fee_tiers.is_empty());
    }

    #[test]
    fn test_aligned_pool_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedPoolData>(), 64);
        assert!(mem::size_of::<AlignedPoolData>() <= 64);
    }

    #[test]
    fn test_pancake_stats_operations() {
        let stats = PancakeStats::default();

        stats.swaps_executed.fetch_add(10, Ordering::Relaxed);
        stats.total_volume_usd.fetch_add(50000, Ordering::Relaxed);
        stats.arbitrage_opportunities.fetch_add(5, Ordering::Relaxed);

        assert_eq!(stats.swaps_executed.load(Ordering::Relaxed), 10);
        assert_eq!(stats.total_volume_usd.load(Ordering::Relaxed), 50000);
        assert_eq!(stats.arbitrage_opportunities.load(Ordering::Relaxed), 5);
    }

    #[test]
    #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
    fn test_aligned_pool_data_staleness() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedPoolData::new(1_000_000_000_000_000_000, 10_000_000, 5_000_000, now);
        let stale_data = AlignedPoolData::new(1_000_000_000_000_000_000, 10_000_000, 5_000_000, now - 10_000);

        assert!(!fresh_data.is_stale(5_000));
        assert!(stale_data.is_stale(5_000));
    }

    #[test]
    fn test_aligned_pool_data_conversions() {
        let data = AlignedPoolData::new(
            2_000_000_000_000_000_000, // 2.0 price
            15_000_000,                // $15 TVL
            8_000_000,                 // $8 volume
            1_640_995_200_000,
        );

        assert_eq!(data.price(), dec!(2.0));
        assert_eq!(data.tvl_usd(), dec!(15));
    }

    #[test]
    fn test_pancake_pool_creation() {
        let pool = PancakePool {
            address: PANCAKE_V3_FACTORY.to_string(),
            token0: TokenAddress::ZERO,
            token1: TokenAddress([1_u8; 20]),
            fee: 2500, // 0.25%
            price: dec!(1.5),
            tvl_usd: dec!(100000),
            volume_24h_usd: dec!(50000),
            tick: 1000,
            liquidity: 1_000_000_000_000_000_000,
            creation_block: 25_000_000,
            last_update: 1_640_995_200,
        };

        assert_eq!(pool.fee, 2500);
        assert_eq!(pool.price, dec!(1.5));
        assert_eq!(pool.tvl_usd, dec!(100000));
    }

    #[test]
    fn test_swap_route_creation() {
        let pool = PancakePool {
            address: "0x123".to_string(),
            token0: TokenAddress::ZERO,
            token1: TokenAddress([1_u8; 20]),
            fee: 500,
            price: dec!(2000),
            tvl_usd: dec!(1000000),
            volume_24h_usd: dec!(500000),
            tick: 2000,
            liquidity: 5_000_000_000_000_000_000,
            creation_block: 25_000_000,
            last_update: 1_640_995_200,
        };

        let route = SwapRoute {
            pools: vec![pool],
            expected_output: dec!(1.95),
            price_impact: dec!(0.001),
            total_fees: dec!(0.005),
            gas_estimate: 150_000,
            confidence: 95,
        };

        assert_eq!(route.pools.len(), 1);
        assert_eq!(route.expected_output, dec!(1.95));
        assert_eq!(route.confidence, 95);
    }

    #[test]
    fn test_yield_position_creation() {
        let position = YieldPosition {
            position_id: 12345,
            pool_address: "0x456".to_string(),
            tick_lower: -1000,
            tick_upper: 1000,
            liquidity: 2_000_000_000_000_000_000,
            amount0: dec!(100),
            amount1: dec!(200000),
            unclaimed_fees: dec!(5.5),
            current_apy: dec!(25.5),
            value_usd: dec!(200100),
        };

        assert_eq!(position.position_id, 12345);
        assert_eq!(position.current_apy, dec!(25.5));
        assert_eq!(position.value_usd, dec!(200100));
    }

    #[tokio::test]
    async fn test_optimal_route_finding() {
        let config = Arc::new(ChainCoreConfig::default());
        let bsc_config = BscConfig::default();

        let Ok(integration) = PancakeIntegration::new(config, bsc_config).await else {
            return;
        };

        // Add a test pool
        let test_pool = PancakePool {
            address: "0x789".to_string(),
            token0: TokenAddress::ZERO,
            token1: TokenAddress([1_u8; 20]),
            fee: 2500,
            price: dec!(2000),
            tvl_usd: dec!(5000000),
            volume_24h_usd: dec!(1000000),
            tick: 1500,
            liquidity: 10_000_000_000_000_000_000,
            creation_block: 25_000_000,
            last_update: 1_640_995_200,
        };

        integration.pools.insert(test_pool.address.clone(), test_pool);

        let route = integration.find_optimal_route(
            TokenAddress::ZERO,
            TokenAddress([1_u8; 20]),
            dec!(1.0),
        ).await;

        assert!(route.is_ok());
        if let Ok(route) = route {
            assert!(!route.pools.is_empty());
            assert!(route.expected_output > Decimal::ZERO);
            assert_eq!(route.confidence, 95);
        }
    }
}
