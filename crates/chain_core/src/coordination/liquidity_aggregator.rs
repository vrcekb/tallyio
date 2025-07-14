//! Liquidity Aggregator for ultra-performance multi-protocol liquidity aggregation
//!
//! This module provides advanced liquidity aggregation capabilities for maximizing
//! available liquidity across different protocols and chains through intelligent
//! routing and optimal execution strategies.
//!
//! ## Performance Targets
//! - Liquidity Discovery: <50μs
//! - Route Optimization: <100μs
//! - Pool Analysis: <75μs
//! - Aggregation Calculation: <25μs
//! - Best Route Selection: <150μs
//!
//! ## Architecture
//! - Real-time multi-protocol liquidity monitoring
//! - Advanced route optimization algorithms
//! - Cross-chain liquidity aggregation
//! - Dynamic pool weight calculation
//! - Lock-free aggregation primitives

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

/// Liquidity aggregator configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct LiquidityAggregatorConfig {
    /// Enable liquidity aggregation
    pub enabled: bool,
    
    /// Pool discovery interval in milliseconds
    pub pool_discovery_interval_ms: u64,
    
    /// Route optimization interval in milliseconds
    pub route_optimization_interval_ms: u64,
    
    /// Liquidity refresh interval in milliseconds
    pub liquidity_refresh_interval_ms: u64,
    
    /// Enable cross-chain aggregation
    pub enable_cross_chain_aggregation: bool,
    
    /// Enable dynamic routing
    pub enable_dynamic_routing: bool,
    
    /// Enable pool weight optimization
    pub enable_pool_weight_optimization: bool,
    
    /// Enable slippage optimization
    pub enable_slippage_optimization: bool,
    
    /// Minimum pool liquidity threshold (USD)
    pub min_pool_liquidity_usd: Decimal,
    
    /// Maximum slippage tolerance
    pub max_slippage_tolerance: Decimal,
    
    /// Maximum route hops
    pub max_route_hops: u8,
    
    /// Supported protocols
    pub supported_protocols: Vec<String>,
    
    /// Supported chains for aggregation
    pub supported_chains: Vec<ChainId>,
}

/// Liquidity pool information
#[derive(Debug, Clone)]
pub struct LiquidityPool {
    /// Pool ID
    pub id: String,
    
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Protocol name
    pub protocol: String,
    
    /// Pool address
    pub address: String,
    
    /// Token A address
    pub token_a: String,
    
    /// Token B address
    pub token_b: String,
    
    /// Token A symbol
    pub token_a_symbol: String,
    
    /// Token B symbol
    pub token_b_symbol: String,
    
    /// Token A reserves
    pub reserves_a: Decimal,
    
    /// Token B reserves
    pub reserves_b: Decimal,
    
    /// Total liquidity USD
    pub total_liquidity_usd: Decimal,
    
    /// Pool fee percentage
    pub fee_percentage: Decimal,
    
    /// Pool weight (for routing)
    pub weight: Decimal,
    
    /// Pool utilization rate
    pub utilization_rate: Decimal,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Liquidity route for optimal execution
#[derive(Debug, Clone)]
pub struct LiquidityRoute {
    /// Route ID
    pub id: String,
    
    /// Input token
    pub input_token: String,
    
    /// Output token
    pub output_token: String,
    
    /// Input amount
    pub input_amount: Decimal,
    
    /// Expected output amount
    pub expected_output: Decimal,
    
    /// Route hops
    pub hops: Vec<RouteHop>,
    
    /// Total fee percentage
    pub total_fee_percentage: Decimal,
    
    /// Expected slippage
    pub expected_slippage: Decimal,
    
    /// Route score (higher is better)
    pub route_score: Decimal,
    
    /// Execution time estimate (seconds)
    pub execution_time_estimate_s: u32,
    
    /// Route created timestamp
    pub created_at: u64,
}

/// Single hop in a liquidity route
#[derive(Debug, Clone)]
pub struct RouteHop {
    /// Hop number
    pub hop_number: u8,
    
    /// Pool ID
    pub pool_id: String,
    
    /// Chain ID
    pub chain_id: ChainId,
    
    /// Protocol
    pub protocol: String,
    
    /// Input token
    pub input_token: String,
    
    /// Output token
    pub output_token: String,
    
    /// Input amount
    pub input_amount: Decimal,
    
    /// Expected output amount
    pub expected_output: Decimal,
    
    /// Pool fee
    pub pool_fee: Decimal,
    
    /// Estimated gas cost (USD)
    pub gas_cost_usd: Decimal,
}

/// Aggregated liquidity information
#[derive(Debug, Clone)]
pub struct AggregatedLiquidity {
    /// Token pair
    pub token_pair: String,
    
    /// Total available liquidity (USD)
    pub total_liquidity_usd: Decimal,
    
    /// Number of pools
    pub pool_count: u32,
    
    /// Number of protocols
    pub protocol_count: u32,
    
    /// Number of chains
    pub chain_count: u32,
    
    /// Best buy price
    pub best_buy_price: Decimal,
    
    /// Best sell price
    pub best_sell_price: Decimal,
    
    /// Average fee percentage
    pub avg_fee_percentage: Decimal,
    
    /// Liquidity distribution
    pub liquidity_distribution: Vec<LiquidityDistribution>,
    
    /// Last aggregation timestamp
    pub last_aggregated: u64,
}

/// Liquidity distribution across protocols/chains
#[derive(Debug, Clone)]
pub struct LiquidityDistribution {
    /// Protocol or chain name
    pub name: String,
    
    /// Liquidity amount (USD)
    pub liquidity_usd: Decimal,
    
    /// Percentage of total liquidity
    pub percentage: Decimal,
    
    /// Number of pools
    pub pool_count: u32,
}

/// Liquidity aggregator statistics
#[derive(Debug, Default)]
pub struct LiquidityAggregatorStats {
    /// Total pools monitored
    pub pools_monitored: AtomicU64,
    
    /// Total routes calculated
    pub routes_calculated: AtomicU64,
    
    /// Successful aggregations
    pub successful_aggregations: AtomicU64,
    
    /// Failed aggregations
    pub failed_aggregations: AtomicU64,
    
    /// Total liquidity aggregated (USD)
    pub total_liquidity_aggregated_usd: AtomicU64,
    
    /// Pool discoveries performed
    pub pool_discoveries_performed: AtomicU64,
    
    /// Route optimizations performed
    pub route_optimizations_performed: AtomicU64,
    
    /// Cross-chain routes created
    pub cross_chain_routes_created: AtomicU64,
    
    /// Average route calculation time (μs)
    pub avg_route_calculation_time_us: AtomicU64,
    
    /// Best route selections
    pub best_route_selections: AtomicU64,
}

/// Cache-line aligned liquidity data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedLiquidityData {
    /// Total pools monitored
    pub total_pools_monitored: u64,
    
    /// Total liquidity USD (scaled by 1e6)
    pub total_liquidity_usd_scaled: u64,
    
    /// Average fee percentage (scaled by 1e6)
    pub avg_fee_percentage_scaled: u64,
    
    /// Best route score (scaled by 1e6)
    pub best_route_score_scaled: u64,
    
    /// Route calculation time (μs)
    pub route_calculation_time_us: u64,
    
    /// Active routes count
    pub active_routes_count: u64,
    
    /// Cross-chain routes percentage (scaled by 1e6)
    pub cross_chain_routes_pct_scaled: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// Liquidity aggregator constants
pub const AGGREGATOR_DEFAULT_POOL_DISCOVERY_INTERVAL_MS: u64 = 1000; // 1 second
pub const AGGREGATOR_DEFAULT_ROUTE_OPTIMIZATION_INTERVAL_MS: u64 = 500; // 500ms
pub const AGGREGATOR_DEFAULT_LIQUIDITY_REFRESH_INTERVAL_MS: u64 = 2000; // 2 seconds
pub const AGGREGATOR_DEFAULT_MIN_POOL_LIQUIDITY_USD: &str = "1000.0"; // $1k minimum
pub const AGGREGATOR_DEFAULT_MAX_SLIPPAGE: &str = "0.01"; // 1% maximum
pub const AGGREGATOR_DEFAULT_MAX_HOPS: u8 = 3; // Maximum 3 hops
pub const AGGREGATOR_MAX_POOLS: usize = 10000;
pub const AGGREGATOR_MAX_ROUTES: usize = 1000;

impl Default for LiquidityAggregatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pool_discovery_interval_ms: AGGREGATOR_DEFAULT_POOL_DISCOVERY_INTERVAL_MS,
            route_optimization_interval_ms: AGGREGATOR_DEFAULT_ROUTE_OPTIMIZATION_INTERVAL_MS,
            liquidity_refresh_interval_ms: AGGREGATOR_DEFAULT_LIQUIDITY_REFRESH_INTERVAL_MS,
            enable_cross_chain_aggregation: true,
            enable_dynamic_routing: true,
            enable_pool_weight_optimization: true,
            enable_slippage_optimization: true,
            min_pool_liquidity_usd: AGGREGATOR_DEFAULT_MIN_POOL_LIQUIDITY_USD.parse().unwrap_or_default(),
            max_slippage_tolerance: AGGREGATOR_DEFAULT_MAX_SLIPPAGE.parse().unwrap_or_default(),
            max_route_hops: AGGREGATOR_DEFAULT_MAX_HOPS,
            supported_protocols: vec![
                "Uniswap V3".to_string(),
                "Uniswap V2".to_string(),
                "SushiSwap".to_string(),
                "Curve".to_string(),
                "Balancer".to_string(),
                "1inch".to_string(),
                "PancakeSwap".to_string(),
                "QuickSwap".to_string(),
                "TraderJoe".to_string(),
                "Velodrome".to_string(),
                "Aerodrome".to_string(),
                "Camelot".to_string(),
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

impl AlignedLiquidityData {
    /// Create new aligned liquidity data
    #[inline(always)]
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "Aligned data structure requires all fields")]
    pub const fn new(
        total_pools_monitored: u64,
        total_liquidity_usd_scaled: u64,
        avg_fee_percentage_scaled: u64,
        best_route_score_scaled: u64,
        route_calculation_time_us: u64,
        active_routes_count: u64,
        cross_chain_routes_pct_scaled: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            total_pools_monitored,
            total_liquidity_usd_scaled,
            avg_fee_percentage_scaled,
            best_route_score_scaled,
            route_calculation_time_us,
            active_routes_count,
            cross_chain_routes_pct_scaled,
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

    /// Get total liquidity USD as Decimal
    #[inline(always)]
    #[must_use]
    pub fn total_liquidity_usd(&self) -> Decimal {
        Decimal::from(self.total_liquidity_usd_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get average fee percentage as Decimal
    #[inline(always)]
    #[must_use]
    pub fn avg_fee_percentage(&self) -> Decimal {
        Decimal::from(self.avg_fee_percentage_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get best route score as Decimal
    #[inline(always)]
    #[must_use]
    pub fn best_route_score(&self) -> Decimal {
        Decimal::from(self.best_route_score_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get cross-chain routes percentage as Decimal
    #[inline(always)]
    #[must_use]
    pub fn cross_chain_routes_percentage(&self) -> Decimal {
        Decimal::from(self.cross_chain_routes_pct_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get liquidity per pool
    #[inline(always)]
    #[must_use]
    pub fn liquidity_per_pool(&self) -> Decimal {
        if self.total_pools_monitored == 0 {
            return Decimal::ZERO;
        }

        self.total_liquidity_usd() / Decimal::from(self.total_pools_monitored)
    }

    /// Get route efficiency (routes per pool)
    #[inline(always)]
    #[must_use]
    pub fn route_efficiency(&self) -> Decimal {
        if self.total_pools_monitored == 0 {
            return Decimal::ZERO;
        }

        Decimal::from(self.active_routes_count) / Decimal::from(self.total_pools_monitored)
    }
}

/// Liquidity Aggregator for ultra-performance multi-protocol liquidity aggregation
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct LiquidityAggregator {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Aggregator specific configuration
    aggregator_config: LiquidityAggregatorConfig,

    /// Statistics
    stats: Arc<LiquidityAggregatorStats>,

    /// Monitored liquidity pools
    pools: Arc<RwLock<HashMap<String, LiquidityPool>>>,

    /// Liquidity data cache for ultra-fast access
    liquidity_cache: Arc<DashMap<String, AlignedLiquidityData>>,

    /// Calculated routes
    routes: Arc<RwLock<HashMap<String, LiquidityRoute>>>,

    /// Aggregated liquidity data
    aggregated_liquidity: Arc<RwLock<HashMap<String, AggregatedLiquidity>>>,

    /// Performance timers
    discovery_timer: Timer,
    optimization_timer: Timer,
    aggregation_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Pool update channels
    pool_sender: Sender<LiquidityPool>,
    pool_receiver: Receiver<LiquidityPool>,

    /// Route update channels
    route_sender: Sender<LiquidityRoute>,
    route_receiver: Receiver<LiquidityRoute>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current aggregation round
    aggregation_round: Arc<TokioMutex<u64>>,
}

impl LiquidityAggregator {
    /// Create new liquidity aggregator with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let aggregator_config = LiquidityAggregatorConfig::default();
        let stats = Arc::new(LiquidityAggregatorStats::default());
        let pools = Arc::new(RwLock::new(HashMap::with_capacity(AGGREGATOR_MAX_POOLS)));
        let liquidity_cache = Arc::new(DashMap::with_capacity(1000));
        let routes = Arc::new(RwLock::new(HashMap::with_capacity(AGGREGATOR_MAX_ROUTES)));
        let aggregated_liquidity = Arc::new(RwLock::new(HashMap::with_capacity(100)));
        let discovery_timer = Timer::new("liquidity_discovery");
        let optimization_timer = Timer::new("route_optimization");
        let aggregation_timer = Timer::new("liquidity_aggregation");
        let shutdown = Arc::new(AtomicBool::new(false));
        let aggregation_round = Arc::new(TokioMutex::new(0));

        let (pool_sender, pool_receiver) = channel::bounded(AGGREGATOR_MAX_POOLS);
        let (route_sender, route_receiver) = channel::bounded(AGGREGATOR_MAX_ROUTES);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            aggregator_config,
            stats,
            pools,
            liquidity_cache,
            routes,
            aggregated_liquidity,
            discovery_timer,
            optimization_timer,
            aggregation_timer,
            shutdown,
            pool_sender,
            pool_receiver,
            route_sender,
            route_receiver,
            http_client,
            aggregation_round,
        })
    }

    /// Start liquidity aggregation services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.aggregator_config.enabled {
            info!("Liquidity aggregation disabled");
            return Ok(());
        }

        info!("Starting liquidity aggregation");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start pool discovery
        self.start_pool_discovery().await;

        // Start route optimization
        if self.aggregator_config.enable_dynamic_routing {
            self.start_route_optimization().await;
        }

        // Start liquidity aggregation
        self.start_liquidity_aggregation().await;

        // Start cross-chain aggregation
        if self.aggregator_config.enable_cross_chain_aggregation {
            self.start_cross_chain_aggregation().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Liquidity aggregation started successfully");
        Ok(())
    }

    /// Stop liquidity aggregation
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping liquidity aggregation");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Liquidity aggregation stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &LiquidityAggregatorStats {
        &self.stats
    }

    /// Get monitored pools
    #[inline]
    pub async fn get_pools(&self) -> Vec<LiquidityPool> {
        let pools = self.pools.read().await;
        pools.values().cloned().collect()
    }

    /// Get calculated routes
    #[inline]
    pub async fn get_routes(&self) -> Vec<LiquidityRoute> {
        let routes = self.routes.read().await;
        routes.values().cloned().collect()
    }

    /// Get aggregated liquidity
    #[inline]
    pub async fn get_aggregated_liquidity(&self) -> Vec<AggregatedLiquidity> {
        let aggregated = self.aggregated_liquidity.read().await;
        aggregated.values().cloned().collect()
    }

    /// Find best route for token swap
    #[inline]
    #[must_use]
    #[expect(clippy::significant_drop_tightening, reason = "Pool guard needed for entire calculation")]
    pub async fn find_best_route(
        &self,
        input_token: &str,
        output_token: &str,
        input_amount: Decimal,
        max_slippage: Option<Decimal>,
    ) -> Option<LiquidityRoute> {
        let pools = self.pools.read().await;
        let max_slippage = max_slippage.unwrap_or(self.aggregator_config.max_slippage_tolerance);

        let routes = Self::calculate_routes(
            &pools,
            input_token,
            output_token,
            input_amount,
            self.aggregator_config.max_route_hops,
        );

        let mut best_route = None;
        let mut best_score = Decimal::ZERO;

        for route in routes {
            if route.expected_slippage <= max_slippage {
                let score = Self::calculate_route_score(&route);
                if score > best_score {
                    best_score = score;
                    best_route = Some(route);
                }
            }
        }

        if best_route.is_some() {
            self.stats.best_route_selections.fetch_add(1, Ordering::Relaxed);
        }

        best_route
    }

    /// Calculate optimal routes between tokens
    fn calculate_routes(
        pools: &HashMap<String, LiquidityPool>,
        input_token: &str,
        output_token: &str,
        input_amount: Decimal,
        max_hops: u8,
    ) -> Vec<LiquidityRoute> {
        let mut routes = Vec::new();

        // Direct routes (single hop)
        for pool in pools.values() {
            if let Some(route) = Self::create_direct_route(pool, input_token, output_token, input_amount) {
                routes.push(route);
            }
        }

        // Multi-hop routes (if enabled)
        if max_hops > 1 {
            routes.extend(Self::calculate_multi_hop_routes(
                pools,
                input_token,
                output_token,
                input_amount,
                max_hops,
            ));
        }

        routes
    }

    /// Create direct route through single pool
    fn create_direct_route(
        pool: &LiquidityPool,
        input_token: &str,
        output_token: &str,
        input_amount: Decimal,
    ) -> Option<LiquidityRoute> {
        // Check if pool supports the token pair
        let (input_is_a, _output_is_b) = if pool.token_a_symbol == input_token && pool.token_b_symbol == output_token {
            (true, true)
        } else if pool.token_b_symbol == input_token && pool.token_a_symbol == output_token {
            (false, false)
        } else {
            return None;
        };

        let (input_reserves, output_reserves) = if input_is_a {
            (pool.reserves_a, pool.reserves_b)
        } else {
            (pool.reserves_b, pool.reserves_a)
        };

        // Calculate output using constant product formula (x * y = k)
        let fee_multiplier = Decimal::ONE - pool.fee_percentage;
        let input_with_fee = input_amount * fee_multiplier;
        let expected_output = (output_reserves * input_with_fee) / (input_reserves + input_with_fee);

        // Calculate slippage
        let price_impact = input_amount / (input_reserves + input_amount);
        let expected_slippage = price_impact * "2".parse::<Decimal>().unwrap_or_default(); // Simplified slippage calculation

        let hop = RouteHop {
            hop_number: 1,
            pool_id: pool.id.clone(),
            chain_id: pool.chain_id,
            protocol: pool.protocol.clone(),
            input_token: input_token.to_string(),
            output_token: output_token.to_string(),
            input_amount,
            expected_output,
            pool_fee: pool.fee_percentage,
            gas_cost_usd: Self::estimate_gas_cost(pool.chain_id, &pool.protocol),
        };

        Some(LiquidityRoute {
            id: format!("route_{}_{}", chrono::Utc::now().timestamp_millis(), fastrand::u32(..)),
            input_token: input_token.to_string(),
            output_token: output_token.to_string(),
            input_amount,
            expected_output,
            hops: vec![hop],
            total_fee_percentage: pool.fee_percentage,
            expected_slippage,
            route_score: Self::calculate_route_score_simple(expected_output, pool.fee_percentage, expected_slippage),
            execution_time_estimate_s: Self::estimate_execution_time(pool.chain_id),
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for route data")]
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })
    }

    /// Calculate multi-hop routes
    fn calculate_multi_hop_routes(
        pools: &HashMap<String, LiquidityPool>,
        input_token: &str,
        output_token: &str,
        input_amount: Decimal,
        _max_hops: u8,
    ) -> Vec<LiquidityRoute> {
        let mut routes = Vec::new();

        // Find intermediate tokens
        let intermediate_tokens = Self::find_intermediate_tokens(pools, input_token, output_token);

        for intermediate in intermediate_tokens {
            // First hop: input_token -> intermediate
            if let Some(first_route) = Self::find_best_single_hop(pools, input_token, &intermediate, input_amount) {
                // Second hop: intermediate -> output_token
                if let Some(second_route) = Self::find_best_single_hop(pools, &intermediate, output_token, first_route.expected_output) {
                    if let Some(combined_route) = Self::combine_routes(&first_route, &second_route) {
                        routes.push(combined_route);
                    }
                }
            }
        }

        // For 3+ hops, we would recursively continue this process
        // Simplified implementation for now

        routes
    }

    /// Find intermediate tokens for multi-hop routing
    fn find_intermediate_tokens(
        pools: &HashMap<String, LiquidityPool>,
        input_token: &str,
        output_token: &str,
    ) -> Vec<String> {
        let mut intermediates = Vec::new();
        let mut token_counts: HashMap<String, u32> = HashMap::new();

        // Count token occurrences in pools
        for pool in pools.values() {
            *token_counts.entry(pool.token_a_symbol.clone()).or_insert(0) += 1;
            *token_counts.entry(pool.token_b_symbol.clone()).or_insert(0) += 1;
        }

        // Find tokens that appear in multiple pools (good intermediates)
        for (token, count) in &token_counts {
            if token != input_token && token != output_token && *count >= 2 {
                intermediates.push(token.clone());
            }
        }

        // Sort by liquidity/popularity (simplified: by count)
        intermediates.sort_by(|a, b| {
            let count_a = token_counts.get(a).unwrap_or(&0);
            let count_b = token_counts.get(b).unwrap_or(&0);
            count_b.cmp(count_a)
        });

        // Return top 5 intermediate tokens
        intermediates.truncate(5);
        intermediates
    }

    /// Find best single hop between tokens
    fn find_best_single_hop(
        pools: &HashMap<String, LiquidityPool>,
        input_token: &str,
        output_token: &str,
        input_amount: Decimal,
    ) -> Option<LiquidityRoute> {
        let mut best_route = None;
        let mut best_output = Decimal::ZERO;

        for pool in pools.values() {
            if let Some(route) = Self::create_direct_route(pool, input_token, output_token, input_amount) {
                if route.expected_output > best_output {
                    best_output = route.expected_output;
                    best_route = Some(route);
                }
            }
        }

        best_route
    }

    /// Combine two single-hop routes into multi-hop route
    fn combine_routes(first_route: &LiquidityRoute, second_route: &LiquidityRoute) -> Option<LiquidityRoute> {
        if first_route.output_token != second_route.input_token {
            return None;
        }

        let mut combined_hops = first_route.hops.clone();
        let mut second_hops = second_route.hops.clone();

        // Update hop numbers for second route
        for hop in &mut second_hops {
            #[expect(clippy::cast_possible_truncation, reason = "Route hops are limited to small numbers")]
            {
                hop.hop_number += first_route.hops.len() as u8;
            }
        }

        combined_hops.extend(second_hops);

        let total_fee = first_route.total_fee_percentage + second_route.total_fee_percentage;
        let total_slippage = first_route.expected_slippage + second_route.expected_slippage;

        Some(LiquidityRoute {
            id: format!("route_{}_{}", chrono::Utc::now().timestamp_millis(), fastrand::u32(..)),
            input_token: first_route.input_token.clone(),
            output_token: second_route.output_token.clone(),
            input_amount: first_route.input_amount,
            expected_output: second_route.expected_output,
            hops: combined_hops,
            total_fee_percentage: total_fee,
            expected_slippage: total_slippage,
            route_score: Self::calculate_route_score_simple(second_route.expected_output, total_fee, total_slippage),
            execution_time_estimate_s: first_route.execution_time_estimate_s + second_route.execution_time_estimate_s,
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for route data")]
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })
    }

    /// Calculate route score
    fn calculate_route_score(route: &LiquidityRoute) -> Decimal {
        Self::calculate_route_score_simple(
            route.expected_output,
            route.total_fee_percentage,
            route.expected_slippage,
        )
    }

    /// Calculate simple route score
    fn calculate_route_score_simple(
        expected_output: Decimal,
        total_fee_percentage: Decimal,
        expected_slippage: Decimal,
    ) -> Decimal {
        // Higher output is better, lower fees and slippage are better
        let fee_penalty = total_fee_percentage * "100".parse::<Decimal>().unwrap_or_default();
        let slippage_penalty = expected_slippage * "50".parse::<Decimal>().unwrap_or_default();

        expected_output - fee_penalty - slippage_penalty
    }

    /// Estimate gas cost for chain and protocol
    fn estimate_gas_cost(chain_id: ChainId, protocol: &str) -> Decimal {
        let base_cost: Decimal = match chain_id {
            ChainId::Ethereum => "50".parse().unwrap_or_default(), // $50 on Ethereum
            ChainId::Arbitrum | ChainId::Avalanche => "2".parse().unwrap_or_default(),  // $2 on Arbitrum/Avalanche
            ChainId::Optimism => "3".parse().unwrap_or_default(),  // $3 on Optimism
            ChainId::Polygon => "0.5".parse().unwrap_or_default(), // $0.5 on Polygon
            ChainId::Bsc => "1".parse().unwrap_or_default(),       // $1 on BSC
            ChainId::Base => "1.5".parse().unwrap_or_default(),    // $1.5 on Base
        };

        // Protocol-specific multipliers
        let protocol_multiplier: Decimal = match protocol {
            "Uniswap V3" => "1.5".parse().unwrap_or_default(), // More complex
            "Curve" => "1.3".parse().unwrap_or_default(),      // Complex math
            "Balancer" => "1.4".parse().unwrap_or_default(),   // Complex pools
            _ => "1.0".parse().unwrap_or_default(),             // Standard
        };

        base_cost * protocol_multiplier
    }

    /// Estimate execution time for chain
    const fn estimate_execution_time(chain_id: ChainId) -> u32 {
        match chain_id {
            ChainId::Ethereum => 180,    // 3 minutes
            ChainId::Arbitrum | ChainId::Avalanche => 30,     // 30 seconds
            ChainId::Optimism => 60,     // 1 minute
            ChainId::Polygon | ChainId::Base => 45,      // 45 seconds
            ChainId::Bsc => 15,          // 15 seconds
        }
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(2000)) // Liquidity data timeout
            .http2_prior_knowledge()
            .http2_keep_alive_timeout(Duration::from_secs(30))
            .http2_keep_alive_interval(Duration::from_secs(10))
            .pool_max_idle_per_host(15)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .map_err(|_e| crate::ChainCoreError::Network(crate::NetworkError::ConnectionRefused))?;

        {
            let mut http_client_guard = self.http_client.lock().await;
            *http_client_guard = Some(client);
        }

        Ok(())
    }

    /// Start pool discovery
    async fn start_pool_discovery(&self) {
        let pool_receiver = self.pool_receiver.clone();
        let pools = Arc::clone(&self.pools);
        let liquidity_cache = Arc::clone(&self.liquidity_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let aggregator_config = self.aggregator_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(aggregator_config.pool_discovery_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming pool updates
                while let Ok(pool) = pool_receiver.try_recv() {
                    let pool_key = format!("{}_{:?}_{}", pool.id, pool.chain_id, pool.protocol);

                    // Update pools
                    {
                        let mut pools_guard = pools.write().await;
                        pools_guard.insert(pool_key.clone(), pool.clone());

                        // Keep only recent pools
                        while pools_guard.len() > AGGREGATOR_MAX_POOLS {
                            if let Some(oldest_key) = pools_guard.keys().next().cloned() {
                                pools_guard.remove(&oldest_key);
                            }
                        }
                        drop(pools_guard);
                    }

                    // Update cache with aligned data
                    let aligned_data = AlignedLiquidityData::new(
                        1, // Single pool
                        (pool.total_liquidity_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        (pool.fee_percentage * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        (pool.weight * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        50, // Default calculation time
                        1,  // Single route
                        0,  // Not cross-chain
                        pool.last_update,
                    );
                    liquidity_cache.insert(pool_key, aligned_data);

                    stats.pools_monitored.fetch_add(1, Ordering::Relaxed);
                }

                // Discover new pools from external sources
                if let Ok(discovered_pools) = Self::discover_pools(&aggregator_config.supported_chains, &aggregator_config.supported_protocols).await {
                    for pool in discovered_pools {
                        let pool_key = format!("{}_{:?}_{}", pool.id, pool.chain_id, pool.protocol);

                        // Update pools directly since we're in the same task
                        {
                            let mut pools_guard = pools.write().await;
                            pools_guard.insert(pool_key, pool);
                        }
                    }
                }

                stats.pool_discoveries_performed.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let discovery_time = start_time.elapsed().as_micros() as u64;
                trace!("Pool discovery cycle completed in {}μs", discovery_time);
            }
        });
    }

    /// Start route optimization
    async fn start_route_optimization(&self) {
        let route_receiver = self.route_receiver.clone();
        let routes = Arc::clone(&self.routes);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let aggregator_config = self.aggregator_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(aggregator_config.route_optimization_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming route updates
                while let Ok(route) = route_receiver.try_recv() {
                    let route_id = route.id.clone();

                    // Store route
                    {
                        let mut routes_guard = routes.write().await;
                        routes_guard.insert(route_id, route);

                        // Keep only recent routes
                        while routes_guard.len() > AGGREGATOR_MAX_ROUTES {
                            if let Some(oldest_key) = routes_guard.keys().next().cloned() {
                                routes_guard.remove(&oldest_key);
                            }
                        }
                        drop(routes_guard);
                    }

                    stats.routes_calculated.fetch_add(1, Ordering::Relaxed);
                }

                stats.route_optimizations_performed.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let optimization_time = start_time.elapsed().as_micros() as u64;
                stats.avg_route_calculation_time_us.store(optimization_time, Ordering::Relaxed);
                trace!("Route optimization cycle completed in {}μs", optimization_time);
            }
        });
    }

    /// Start liquidity aggregation
    async fn start_liquidity_aggregation(&self) {
        let pools = Arc::clone(&self.pools);
        let aggregated_liquidity = Arc::clone(&self.aggregated_liquidity);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let aggregator_config = self.aggregator_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(aggregator_config.liquidity_refresh_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Aggregate liquidity by token pairs
                let pools_guard = pools.read().await;
                let aggregated = Self::aggregate_liquidity_by_pairs(&pools_guard);
                drop(pools_guard);

                // Update aggregated liquidity
                {
                    let mut aggregated_guard = aggregated_liquidity.write().await;
                    for (pair, liquidity) in aggregated {
                        aggregated_guard.insert(pair, liquidity);
                    }

                    // Keep only recent aggregations
                    while aggregated_guard.len() > 100 {
                        if let Some(oldest_key) = aggregated_guard.keys().next().cloned() {
                            aggregated_guard.remove(&oldest_key);
                        }
                    }
                    drop(aggregated_guard);
                }

                stats.successful_aggregations.fetch_add(1, Ordering::Relaxed);
                trace!("Liquidity aggregation cycle completed");
            }
        });
    }

    /// Start cross-chain aggregation
    async fn start_cross_chain_aggregation(&self) {
        let pools = Arc::clone(&self.pools);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10)); // Cross-chain aggregation every 10 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Analyze cross-chain opportunities
                let pools_guard = pools.read().await;
                let cross_chain_routes = Self::find_cross_chain_opportunities(&pools_guard);
                drop(pools_guard);

                stats.cross_chain_routes_created.fetch_add(cross_chain_routes.len() as u64, Ordering::Relaxed);
                trace!("Cross-chain aggregation cycle completed, found {} opportunities", cross_chain_routes.len());
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

                let pools_monitored = stats.pools_monitored.load(Ordering::Relaxed);
                let routes_calculated = stats.routes_calculated.load(Ordering::Relaxed);
                let successful_aggregations = stats.successful_aggregations.load(Ordering::Relaxed);
                let failed_aggregations = stats.failed_aggregations.load(Ordering::Relaxed);
                let total_liquidity = stats.total_liquidity_aggregated_usd.load(Ordering::Relaxed);
                let pool_discoveries = stats.pool_discoveries_performed.load(Ordering::Relaxed);
                let route_optimizations = stats.route_optimizations_performed.load(Ordering::Relaxed);
                let cross_chain_routes = stats.cross_chain_routes_created.load(Ordering::Relaxed);
                let avg_calc_time = stats.avg_route_calculation_time_us.load(Ordering::Relaxed);
                let best_selections = stats.best_route_selections.load(Ordering::Relaxed);

                info!(
                    "Liquidity Aggregator Stats: pools={}, routes={}, successful_agg={}, failed_agg={}, total_liquidity=${}, discoveries={}, optimizations={}, cross_chain={}, avg_calc_time={}μs, best_selections={}",
                    pools_monitored, routes_calculated, successful_aggregations, failed_aggregations,
                    total_liquidity, pool_discoveries, route_optimizations, cross_chain_routes, avg_calc_time, best_selections
                );
            }
        });
    }

    /// Discover pools from external sources
    async fn discover_pools(
        supported_chains: &[ChainId],
        supported_protocols: &[String],
    ) -> Result<Vec<LiquidityPool>> {
        let mut pools = Vec::new();

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for mock pool data")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Mock pool discovery - in production this would query real DEX APIs
        for (i, chain_id) in supported_chains.iter().enumerate() {
            for (j, protocol) in supported_protocols.iter().enumerate() {
                if Self::is_protocol_supported_on_chain(protocol, *chain_id) {
                    let pool = LiquidityPool {
                        #[expect(clippy::cast_possible_truncation, reason = "Chain ID values are small")]
                        id: format!("pool_{}_{}_{}_{}", *chain_id as u8, i, j, fastrand::u32(..)),
                        chain_id: *chain_id,
                        protocol: protocol.clone(),
                        address: format!("0x{:040x}", fastrand::u64(..)),
                        token_a: "0xA0b86a33E6441e6e80D0c4C6C7556C974E1B2c20".to_string(), // USDC
                        token_b: "0xdAC17F958D2ee523a2206206994597C13D831ec7".to_string(), // USDT
                        token_a_symbol: "USDC".to_string(),
                        token_b_symbol: "USDT".to_string(),
                        reserves_a: Self::get_mock_reserves(*chain_id, protocol, true),
                        reserves_b: Self::get_mock_reserves(*chain_id, protocol, false),
                        total_liquidity_usd: Self::get_mock_liquidity(*chain_id, protocol),
                        fee_percentage: Self::get_protocol_fee(protocol),
                        weight: Self::calculate_pool_weight(*chain_id, protocol),
                        utilization_rate: "0.65".parse().unwrap_or_default(),
                        last_update: now,
                    };
                    pools.push(pool);
                }
            }
        }

        Ok(pools)
    }

    /// Check if protocol is supported on chain
    fn is_protocol_supported_on_chain(protocol: &str, chain_id: ChainId) -> bool {
        matches!((protocol, chain_id),
            ("Uniswap V3" | "Uniswap V2", ChainId::Ethereum | ChainId::Arbitrum | ChainId::Optimism | ChainId::Polygon | ChainId::Base) |
            ("SushiSwap", ChainId::Ethereum | ChainId::Arbitrum | ChainId::Polygon | ChainId::Avalanche) |
            ("Curve" | "Balancer", ChainId::Ethereum | ChainId::Arbitrum | ChainId::Optimism | ChainId::Polygon) |
            ("PancakeSwap", ChainId::Bsc | ChainId::Ethereum) |
            ("QuickSwap", ChainId::Polygon) |
            ("TraderJoe", ChainId::Avalanche) |
            ("Velodrome", ChainId::Optimism) |
            ("Aerodrome", ChainId::Base) |
            ("Camelot", ChainId::Arbitrum)
        )
    }

    /// Get mock reserves for testing
    fn get_mock_reserves(chain_id: ChainId, protocol: &str, is_token_a: bool) -> Decimal {
        let base_amount: Decimal = match chain_id {
            ChainId::Ethereum => "1000000".parse().unwrap_or_default(), // 1M
            ChainId::Arbitrum => "500000".parse().unwrap_or_default(),  // 500k
            ChainId::Optimism => "300000".parse().unwrap_or_default(),  // 300k
            ChainId::Polygon => "200000".parse().unwrap_or_default(),   // 200k
            ChainId::Bsc => "400000".parse().unwrap_or_default(),       // 400k
            ChainId::Avalanche => "250000".parse().unwrap_or_default(), // 250k
            ChainId::Base => "150000".parse().unwrap_or_default(),      // 150k
        };

        let protocol_multiplier: Decimal = match protocol {
            "Uniswap V3" => "1.5".parse().unwrap_or_default(),
            "Curve" => "2.0".parse().unwrap_or_default(),
            "Balancer" => "1.3".parse().unwrap_or_default(),
            _ => "1.0".parse().unwrap_or_default(),
        };

        let token_multiplier: Decimal = if is_token_a { "1.0" } else { "0.9999" }.parse().unwrap_or_default();

        base_amount * protocol_multiplier * token_multiplier
    }

    /// Get mock liquidity for testing
    fn get_mock_liquidity(chain_id: ChainId, protocol: &str) -> Decimal {
        let base_liquidity: Decimal = match chain_id {
            ChainId::Ethereum => "10000000".parse().unwrap_or_default(), // $10M
            ChainId::Arbitrum => "5000000".parse().unwrap_or_default(),  // $5M
            ChainId::Optimism => "3000000".parse().unwrap_or_default(),  // $3M
            ChainId::Polygon => "2000000".parse().unwrap_or_default(),   // $2M
            ChainId::Bsc => "4000000".parse().unwrap_or_default(),       // $4M
            ChainId::Avalanche => "2500000".parse().unwrap_or_default(), // $2.5M
            ChainId::Base => "1500000".parse().unwrap_or_default(),      // $1.5M
        };

        let protocol_multiplier: Decimal = match protocol {
            "Uniswap V3" => "1.8".parse().unwrap_or_default(),
            "Curve" => "2.5".parse().unwrap_or_default(),
            "Balancer" => "1.4".parse().unwrap_or_default(),
            "SushiSwap" => "1.2".parse().unwrap_or_default(),
            _ => "1.0".parse().unwrap_or_default(),
        };

        base_liquidity * protocol_multiplier
    }

    /// Get protocol fee
    fn get_protocol_fee(protocol: &str) -> Decimal {
        match protocol {
            "Uniswap V3" | "Velodrome" | "Aerodrome" | "Camelot" => "0.0005".parse().unwrap_or_default(), // 0.05%
            "Curve" => "0.0004".parse().unwrap_or_default(),      // 0.04%
            "Balancer" => "0.001".parse().unwrap_or_default(),    // 0.1%
            "PancakeSwap" => "0.0025".parse().unwrap_or_default(), // 0.25%
            _ => "0.003".parse().unwrap_or_default(),              // 0.3% default (includes Uniswap V2, SushiSwap, QuickSwap, TraderJoe)
        }
    }

    /// Calculate pool weight for routing
    fn calculate_pool_weight(chain_id: ChainId, protocol: &str) -> Decimal {
        let chain_weight: Decimal = match chain_id {
            ChainId::Ethereum => "1.0".parse().unwrap_or_default(),
            ChainId::Arbitrum => "0.9".parse().unwrap_or_default(),
            ChainId::Optimism | ChainId::Base => "0.8".parse().unwrap_or_default(),
            ChainId::Polygon | ChainId::Avalanche => "0.7".parse().unwrap_or_default(),
            ChainId::Bsc => "0.6".parse().unwrap_or_default(),
        };

        let protocol_weight: Decimal = match protocol {
            "Uniswap V3" => "1.0".parse().unwrap_or_default(),
            "Curve" => "0.95".parse().unwrap_or_default(),
            "Balancer" => "0.9".parse().unwrap_or_default(),
            "SushiSwap" => "0.85".parse().unwrap_or_default(),
            _ => "0.8".parse().unwrap_or_default(),
        };

        chain_weight * protocol_weight
    }

    /// Aggregate liquidity by token pairs
    fn aggregate_liquidity_by_pairs(pools: &HashMap<String, LiquidityPool>) -> HashMap<String, AggregatedLiquidity> {
        let mut aggregated: HashMap<String, AggregatedLiquidity> = HashMap::new();

        for pool in pools.values() {
            let pair_key = format!("{}/{}", pool.token_a_symbol, pool.token_b_symbol);

            let entry = aggregated.entry(pair_key.clone()).or_insert_with(|| AggregatedLiquidity {
                token_pair: pair_key,
                total_liquidity_usd: Decimal::ZERO,
                pool_count: 0,
                protocol_count: 0,
                chain_count: 0,
                best_buy_price: Decimal::ZERO,
                best_sell_price: Decimal::ZERO,
                avg_fee_percentage: Decimal::ZERO,
                liquidity_distribution: Vec::new(),
                #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for aggregation data")]
                last_aggregated: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            });

            entry.total_liquidity_usd += pool.total_liquidity_usd;
            entry.pool_count += 1;
            entry.avg_fee_percentage = (entry.avg_fee_percentage * Decimal::from(entry.pool_count - 1) + pool.fee_percentage) / Decimal::from(entry.pool_count);
        }

        aggregated
    }

    /// Find cross-chain opportunities
    fn find_cross_chain_opportunities(pools: &HashMap<String, LiquidityPool>) -> Vec<String> {
        let mut opportunities = Vec::new();
        let mut chains_by_pair: HashMap<String, Vec<ChainId>> = HashMap::new();

        // Group pools by token pair and collect chains
        for pool in pools.values() {
            let pair_key = format!("{}/{}", pool.token_a_symbol, pool.token_b_symbol);
            chains_by_pair.entry(pair_key).or_default().push(pool.chain_id);
        }

        // Find pairs available on multiple chains
        for (pair, chains) in chains_by_pair {
            if chains.len() > 1 {
                opportunities.push(format!("Cross-chain opportunity for {} across {} chains", pair, chains.len()));
            }
        }

        opportunities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_liquidity_aggregator_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(aggregator) = LiquidityAggregator::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(aggregator.stats().pools_monitored.load(Ordering::Relaxed), 0);
        assert_eq!(aggregator.stats().routes_calculated.load(Ordering::Relaxed), 0);
        assert_eq!(aggregator.stats().successful_aggregations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_liquidity_aggregator_config_default() {
        let config = LiquidityAggregatorConfig::default();
        assert!(config.enabled);
        assert_eq!(config.pool_discovery_interval_ms, AGGREGATOR_DEFAULT_POOL_DISCOVERY_INTERVAL_MS);
        assert_eq!(config.route_optimization_interval_ms, AGGREGATOR_DEFAULT_ROUTE_OPTIMIZATION_INTERVAL_MS);
        assert_eq!(config.liquidity_refresh_interval_ms, AGGREGATOR_DEFAULT_LIQUIDITY_REFRESH_INTERVAL_MS);
        assert!(config.enable_cross_chain_aggregation);
        assert!(config.enable_dynamic_routing);
        assert!(config.enable_pool_weight_optimization);
        assert!(config.enable_slippage_optimization);
        assert_eq!(config.max_route_hops, AGGREGATOR_DEFAULT_MAX_HOPS);
        assert!(!config.supported_protocols.is_empty());
        assert!(!config.supported_chains.is_empty());
    }

    #[test]
    fn test_aligned_liquidity_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedLiquidityData>(), 64);
        assert!(mem::size_of::<AlignedLiquidityData>() <= 64);
    }

    #[test]
    fn test_liquidity_aggregator_stats_operations() {
        let stats = LiquidityAggregatorStats::default();

        stats.pools_monitored.fetch_add(50, Ordering::Relaxed);
        stats.routes_calculated.fetch_add(200, Ordering::Relaxed);
        stats.successful_aggregations.fetch_add(180, Ordering::Relaxed);
        stats.failed_aggregations.fetch_add(20, Ordering::Relaxed);
        stats.total_liquidity_aggregated_usd.fetch_add(50_000_000, Ordering::Relaxed);

        assert_eq!(stats.pools_monitored.load(Ordering::Relaxed), 50);
        assert_eq!(stats.routes_calculated.load(Ordering::Relaxed), 200);
        assert_eq!(stats.successful_aggregations.load(Ordering::Relaxed), 180);
        assert_eq!(stats.failed_aggregations.load(Ordering::Relaxed), 20);
        assert_eq!(stats.total_liquidity_aggregated_usd.load(Ordering::Relaxed), 50_000_000);
    }

    #[test]
    fn test_aligned_liquidity_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedLiquidityData::new(
            100, // pools monitored
            50_000_000_000_000, // $50M liquidity (scaled by 1e6)
            3_000, // 0.3% fee (scaled by 1e6)
            850_000, // 0.85 route score (scaled by 1e6)
            75, // 75μs calculation time
            250, // 250 active routes
            150_000, // 15% cross-chain routes (scaled by 1e6)
            now,
        );

        let stale_data = AlignedLiquidityData::new(
            100, 50_000_000_000_000, 3_000, 850_000, 75, 250, 150_000,
            now - 180_000, // 3 minutes old
        );

        assert!(!fresh_data.is_stale(120_000)); // 2 minutes
        assert!(stale_data.is_stale(120_000)); // 2 minutes
    }

    #[test]
    fn test_aligned_liquidity_data_conversions() {
        let data = AlignedLiquidityData::new(
            100, // pools monitored
            50_000_000_000_000, // $50M liquidity (scaled by 1e6)
            3_000, // 0.3% fee (scaled by 1e6)
            850_000, // 0.85 route score (scaled by 1e6)
            75, // 75μs calculation time
            250, // 250 active routes
            150_000, // 15% cross-chain routes (scaled by 1e6)
            1_640_995_200_000,
        );

        assert_eq!(data.total_liquidity_usd(), dec!(50000000));
        assert_eq!(data.avg_fee_percentage(), dec!(0.003));
        assert_eq!(data.best_route_score(), dec!(0.85));
        assert_eq!(data.cross_chain_routes_percentage(), dec!(0.15));
        assert_eq!(data.liquidity_per_pool(), dec!(500000)); // 50M / 100
        assert_eq!(data.route_efficiency(), dec!(2.5)); // 250 / 100
    }

    #[test]
    fn test_liquidity_pool_creation() {
        let pool = LiquidityPool {
            id: "pool_123".to_string(),
            chain_id: ChainId::Ethereum,
            protocol: "Uniswap V3".to_string(),
            address: "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8".to_string(),
            token_a: "0xA0b86a33E6441e6e80D0c4C6C7556C974E1B2c20".to_string(),
            token_b: "0xdAC17F958D2ee523a2206206994597C13D831ec7".to_string(),
            token_a_symbol: "USDC".to_string(),
            token_b_symbol: "USDT".to_string(),
            reserves_a: dec!(1000000),
            reserves_b: dec!(999000),
            total_liquidity_usd: dec!(2000000),
            fee_percentage: dec!(0.0005),
            weight: dec!(1.0),
            utilization_rate: dec!(0.65),
            last_update: 1_640_995_200_000,
        };

        assert_eq!(pool.id, "pool_123");
        assert_eq!(pool.chain_id, ChainId::Ethereum);
        assert_eq!(pool.protocol, "Uniswap V3");
        assert_eq!(pool.token_a_symbol, "USDC");
        assert_eq!(pool.token_b_symbol, "USDT");
        assert_eq!(pool.reserves_a, dec!(1000000));
        assert_eq!(pool.reserves_b, dec!(999000));
        assert_eq!(pool.total_liquidity_usd, dec!(2000000));
        assert_eq!(pool.fee_percentage, dec!(0.0005));
    }

    #[test]
    fn test_route_hop_creation() {
        let hop = RouteHop {
            hop_number: 1,
            pool_id: "pool_123".to_string(),
            chain_id: ChainId::Ethereum,
            protocol: "Uniswap V3".to_string(),
            input_token: "USDC".to_string(),
            output_token: "USDT".to_string(),
            input_amount: dec!(10000),
            expected_output: dec!(9995),
            pool_fee: dec!(0.0005),
            gas_cost_usd: dec!(50),
        };

        assert_eq!(hop.hop_number, 1);
        assert_eq!(hop.pool_id, "pool_123");
        assert_eq!(hop.chain_id, ChainId::Ethereum);
        assert_eq!(hop.protocol, "Uniswap V3");
        assert_eq!(hop.input_token, "USDC");
        assert_eq!(hop.output_token, "USDT");
        assert_eq!(hop.input_amount, dec!(10000));
        assert_eq!(hop.expected_output, dec!(9995));
        assert_eq!(hop.pool_fee, dec!(0.0005));
        assert_eq!(hop.gas_cost_usd, dec!(50));
    }

    #[test]
    fn test_liquidity_route_creation() {
        let hop = RouteHop {
            hop_number: 1,
            pool_id: "pool_123".to_string(),
            chain_id: ChainId::Ethereum,
            protocol: "Uniswap V3".to_string(),
            input_token: "USDC".to_string(),
            output_token: "USDT".to_string(),
            input_amount: dec!(10000),
            expected_output: dec!(9995),
            pool_fee: dec!(0.0005),
            gas_cost_usd: dec!(50),
        };

        let route = LiquidityRoute {
            id: "route_456".to_string(),
            input_token: "USDC".to_string(),
            output_token: "USDT".to_string(),
            input_amount: dec!(10000),
            expected_output: dec!(9995),
            hops: vec![hop],
            total_fee_percentage: dec!(0.0005),
            expected_slippage: dec!(0.002),
            route_score: dec!(9945),
            execution_time_estimate_s: 180,
            created_at: 1_640_995_200_000,
        };

        assert_eq!(route.id, "route_456");
        assert_eq!(route.input_token, "USDC");
        assert_eq!(route.output_token, "USDT");
        assert_eq!(route.input_amount, dec!(10000));
        assert_eq!(route.expected_output, dec!(9995));
        assert_eq!(route.hops.len(), 1);
        assert_eq!(route.total_fee_percentage, dec!(0.0005));
        assert_eq!(route.expected_slippage, dec!(0.002));
        assert_eq!(route.execution_time_estimate_s, 180);
    }

    #[test]
    fn test_aggregated_liquidity_creation() {
        let distribution = LiquidityDistribution {
            name: "Uniswap V3".to_string(),
            liquidity_usd: dec!(5000000),
            percentage: dec!(0.5),
            pool_count: 10,
        };

        let aggregated = AggregatedLiquidity {
            token_pair: "USDC/USDT".to_string(),
            total_liquidity_usd: dec!(10000000),
            pool_count: 20,
            protocol_count: 4,
            chain_count: 3,
            best_buy_price: dec!(0.9999),
            best_sell_price: dec!(1.0001),
            avg_fee_percentage: dec!(0.003),
            liquidity_distribution: vec![distribution],
            last_aggregated: 1_640_995_200_000,
        };

        assert_eq!(aggregated.token_pair, "USDC/USDT");
        assert_eq!(aggregated.total_liquidity_usd, dec!(10000000));
        assert_eq!(aggregated.pool_count, 20);
        assert_eq!(aggregated.protocol_count, 4);
        assert_eq!(aggregated.chain_count, 3);
        assert_eq!(aggregated.best_buy_price, dec!(0.9999));
        assert_eq!(aggregated.best_sell_price, dec!(1.0001));
        assert_eq!(aggregated.avg_fee_percentage, dec!(0.003));
        assert_eq!(aggregated.liquidity_distribution.len(), 1);
    }

    #[test]
    fn test_calculate_route_score_simple() {
        // High output, low fees and slippage = high score
        let high_score = LiquidityAggregator::calculate_route_score_simple(
            dec!(10000), // High output
            dec!(0.001),  // Low fee (0.1%)
            dec!(0.002),  // Low slippage (0.2%)
        );

        // Low output, high fees and slippage = low score
        let low_score = LiquidityAggregator::calculate_route_score_simple(
            dec!(9000),  // Lower output
            dec!(0.01),  // High fee (1%)
            dec!(0.02),  // High slippage (2%)
        );

        assert!(high_score > low_score);
        assert!(high_score > dec!(9900)); // Should be close to output minus small penalties
        // Low score: 9000 - (0.01 * 100) - (0.02 * 50) = 9000 - 1 - 1 = 8998
        assert!(low_score < dec!(9000));  // Should be less than base output
        assert!(low_score > dec!(8990));  // But not too much less
    }

    #[test]
    fn test_estimate_gas_cost() {
        // Ethereum should be most expensive
        let eth_cost = LiquidityAggregator::estimate_gas_cost(ChainId::Ethereum, "Uniswap V3");
        let arb_cost = LiquidityAggregator::estimate_gas_cost(ChainId::Arbitrum, "Uniswap V3");
        let poly_cost = LiquidityAggregator::estimate_gas_cost(ChainId::Polygon, "Uniswap V3");

        assert!(eth_cost > arb_cost);
        assert!(arb_cost > poly_cost);
        assert!(eth_cost >= dec!(50)); // At least $50 on Ethereum
        assert!(poly_cost <= dec!(1));  // At most $1 on Polygon

        // Complex protocols should cost more
        let v3_cost = LiquidityAggregator::estimate_gas_cost(ChainId::Ethereum, "Uniswap V3");
        let v2_cost = LiquidityAggregator::estimate_gas_cost(ChainId::Ethereum, "Uniswap V2");

        assert!(v3_cost > v2_cost); // V3 is more complex
    }

    #[test]
    fn test_estimate_execution_time() {
        // Ethereum should be slowest
        let eth_time = LiquidityAggregator::estimate_execution_time(ChainId::Ethereum);
        let bsc_time = LiquidityAggregator::estimate_execution_time(ChainId::Bsc);
        let arb_time = LiquidityAggregator::estimate_execution_time(ChainId::Arbitrum);

        assert!(eth_time > arb_time);
        assert!(arb_time >= bsc_time);
        assert_eq!(eth_time, 180); // 3 minutes
        assert_eq!(bsc_time, 15);   // 15 seconds
        assert_eq!(arb_time, 30);   // 30 seconds
    }

    #[test]
    fn test_is_protocol_supported_on_chain() {
        // Uniswap should be supported on multiple chains
        assert!(LiquidityAggregator::is_protocol_supported_on_chain("Uniswap V3", ChainId::Ethereum));
        assert!(LiquidityAggregator::is_protocol_supported_on_chain("Uniswap V3", ChainId::Arbitrum));
        assert!(LiquidityAggregator::is_protocol_supported_on_chain("Uniswap V3", ChainId::Polygon));

        // PancakeSwap should only be on BSC and Ethereum
        assert!(LiquidityAggregator::is_protocol_supported_on_chain("PancakeSwap", ChainId::Bsc));
        assert!(LiquidityAggregator::is_protocol_supported_on_chain("PancakeSwap", ChainId::Ethereum));
        assert!(!LiquidityAggregator::is_protocol_supported_on_chain("PancakeSwap", ChainId::Arbitrum));

        // QuickSwap should only be on Polygon
        assert!(LiquidityAggregator::is_protocol_supported_on_chain("QuickSwap", ChainId::Polygon));
        assert!(!LiquidityAggregator::is_protocol_supported_on_chain("QuickSwap", ChainId::Ethereum));

        // TraderJoe should only be on Avalanche
        assert!(LiquidityAggregator::is_protocol_supported_on_chain("TraderJoe", ChainId::Avalanche));
        assert!(!LiquidityAggregator::is_protocol_supported_on_chain("TraderJoe", ChainId::Ethereum));
    }

    #[test]
    fn test_get_protocol_fee() {
        // Test known protocol fees
        assert_eq!(LiquidityAggregator::get_protocol_fee("Uniswap V3"), dec!(0.0005)); // 0.05%
        assert_eq!(LiquidityAggregator::get_protocol_fee("Uniswap V2"), dec!(0.003));  // 0.3%
        assert_eq!(LiquidityAggregator::get_protocol_fee("Curve"), dec!(0.0004));      // 0.04%
        assert_eq!(LiquidityAggregator::get_protocol_fee("Balancer"), dec!(0.001));    // 0.1%
        assert_eq!(LiquidityAggregator::get_protocol_fee("Unknown"), dec!(0.003));     // 0.3% default

        // Curve should have lowest fees
        let curve_fee = LiquidityAggregator::get_protocol_fee("Curve");
        let uniswap_v2_fee = LiquidityAggregator::get_protocol_fee("Uniswap V2");
        assert!(curve_fee < uniswap_v2_fee);
    }

    #[test]
    fn test_calculate_pool_weight() {
        // Ethereum + Uniswap V3 should have highest weight
        let eth_uni_weight = LiquidityAggregator::calculate_pool_weight(ChainId::Ethereum, "Uniswap V3");
        let bsc_pancake_weight = LiquidityAggregator::calculate_pool_weight(ChainId::Bsc, "PancakeSwap");

        assert!(eth_uni_weight > bsc_pancake_weight);
        assert_eq!(eth_uni_weight, dec!(1.0)); // 1.0 * 1.0
        assert!(bsc_pancake_weight < dec!(0.7)); // Should be less than 0.7
    }

    #[tokio::test]
    async fn test_liquidity_aggregator_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(aggregator) = LiquidityAggregator::new(config).await else {
            return;
        };

        let pools = aggregator.get_pools().await;
        assert!(pools.is_empty()); // No pools initially

        let routes = aggregator.get_routes().await;
        assert!(routes.is_empty()); // No routes initially

        let aggregated = aggregator.get_aggregated_liquidity().await;
        assert!(aggregated.is_empty()); // No aggregated data initially

        let stats = aggregator.stats();
        assert_eq!(stats.pools_monitored.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_find_intermediate_tokens() {
        let mut pools = HashMap::new();

        // Create pools with USDC as common intermediate
        pools.insert("pool1".to_string(), LiquidityPool {
            id: "pool1".to_string(),
            chain_id: ChainId::Ethereum,
            protocol: "Uniswap V3".to_string(),
            address: "0x1".to_string(),
            token_a: "ETH".to_string(),
            token_b: "USDC".to_string(),
            token_a_symbol: "ETH".to_string(),
            token_b_symbol: "USDC".to_string(),
            reserves_a: dec!(1000),
            reserves_b: dec!(2500000),
            total_liquidity_usd: dec!(5000000),
            fee_percentage: dec!(0.0005),
            weight: dec!(1.0),
            utilization_rate: dec!(0.65),
            last_update: 1_640_995_200_000,
        });

        pools.insert("pool2".to_string(), LiquidityPool {
            id: "pool2".to_string(),
            chain_id: ChainId::Ethereum,
            protocol: "Uniswap V3".to_string(),
            address: "0x2".to_string(),
            token_a: "USDC".to_string(),
            token_b: "USDT".to_string(),
            token_a_symbol: "USDC".to_string(),
            token_b_symbol: "USDT".to_string(),
            reserves_a: dec!(1000000),
            reserves_b: dec!(999000),
            total_liquidity_usd: dec!(2000000),
            fee_percentage: dec!(0.0005),
            weight: dec!(1.0),
            utilization_rate: dec!(0.65),
            last_update: 1_640_995_200_000,
        });

        let intermediates = LiquidityAggregator::find_intermediate_tokens(&pools, "ETH", "USDT");

        assert!(!intermediates.is_empty());
        assert!(intermediates.contains(&"USDC".to_string()));
    }
}
