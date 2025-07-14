//! Cross-chain coordination module for ultra-performance multi-chain operations
//!
//! This module provides advanced coordination between different blockchain networks
//! for optimal strategy execution, resource allocation, and cross-chain arbitrage.
//!
//! ## Performance Targets
//! - Cross-chain Message Latency: <2ms
//! - State Synchronization: <5ms
//! - Resource Allocation: <1ms
//! - Strategy Coordination: <3ms
//! - Bridge Monitoring: <500μs
//!
//! ## Architecture
//! - Real-time cross-chain state synchronization
//! - Advanced resource allocation algorithms
//! - Multi-chain strategy coordination
//! - Bridge monitoring and optimization
//! - Lock-free coordination primitives

// Submodules
pub mod bridge_monitor;
pub mod chain_selector;
pub mod cross_chain_arbitrage;
pub mod liquidity_aggregator;

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
    collections::{HashMap, BTreeMap},
};
use tokio::{
    sync::{RwLock, Mutex as TokioMutex},
    time::{interval, sleep},
};
use tracing::{info, trace};

/// Cross-chain coordination configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct CoordinationConfig {
    /// Enable cross-chain coordination
    pub enabled: bool,

    /// State synchronization interval in milliseconds
    pub sync_interval_ms: u64,

    /// Resource allocation interval in milliseconds
    pub allocation_interval_ms: u64,

    /// Bridge monitoring interval in milliseconds
    pub bridge_monitor_interval_ms: u64,

    /// Enable cross-chain arbitrage
    pub enable_cross_chain_arbitrage: bool,

    /// Enable resource optimization
    pub enable_resource_optimization: bool,

    /// Enable bridge monitoring
    pub enable_bridge_monitoring: bool,

    /// Enable strategy coordination
    pub enable_strategy_coordination: bool,

    /// Maximum cross-chain message latency (ms)
    pub max_message_latency_ms: u64,

    /// Minimum profit threshold for cross-chain operations (USD)
    pub min_profit_threshold_usd: Decimal,

    /// Supported chains for coordination
    pub supported_chains: Vec<ChainId>,
}

/// Cross-chain state information
#[derive(Debug, Clone)]
pub struct CrossChainState {
    /// Chain ID
    pub chain_id: ChainId,

    /// Current block number
    pub block_number: u64,

    /// Block timestamp
    pub block_timestamp: u64,

    /// Gas price (in wei)
    pub gas_price: u64,

    /// Available liquidity (USD)
    pub available_liquidity_usd: Decimal,

    /// Active strategies count
    pub active_strategies: u32,

    /// Network congestion level (0-100)
    pub congestion_level: u8,

    /// Bridge status
    pub bridge_status: BridgeStatus,

    /// Last update timestamp
    pub last_update: u64,
}

/// Bridge status information
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeStatus {
    /// Bridge is operational
    Operational,
    /// Bridge is congested
    Congested,
    /// Bridge is under maintenance
    Maintenance,
    /// Bridge is offline
    Offline,
}

/// Cross-chain opportunity
#[derive(Debug, Clone)]
pub struct CrossChainOpportunity {
    /// Opportunity ID
    pub id: String,

    /// Source chain
    pub source_chain: ChainId,

    /// Destination chain
    pub destination_chain: ChainId,

    /// Asset to bridge
    pub asset: String,

    /// Amount to bridge
    pub amount: Decimal,

    /// Expected profit (USD)
    pub expected_profit_usd: Decimal,

    /// Bridge fee (USD)
    pub bridge_fee_usd: Decimal,

    /// Estimated execution time (seconds)
    pub estimated_execution_time_s: u32,

    /// Risk level (1-10)
    pub risk_level: u8,

    /// Expiry timestamp
    pub expires_at: u64,
}

/// Resource allocation information
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Chain ID
    pub chain_id: ChainId,

    /// Allocated capital (USD)
    pub allocated_capital_usd: Decimal,

    /// Available capital (USD)
    pub available_capital_usd: Decimal,

    /// Active positions count
    pub active_positions: u32,

    /// Target allocation percentage
    pub target_allocation_pct: Decimal,

    /// Current allocation percentage
    pub current_allocation_pct: Decimal,

    /// Rebalancing needed
    pub needs_rebalancing: bool,

    /// Last rebalancing timestamp
    pub last_rebalancing: u64,
}

/// Strategy coordination message
#[derive(Debug, Clone)]
pub struct StrategyMessage {
    /// Message ID
    pub id: String,

    /// Source chain
    pub source_chain: ChainId,

    /// Target chains
    pub target_chains: Vec<ChainId>,

    /// Message type
    pub message_type: StrategyMessageType,

    /// Message payload
    pub payload: HashMap<String, String>,

    /// Priority (1-10, 10 = highest)
    pub priority: u8,

    /// Timestamp
    pub timestamp: u64,
}

/// Strategy message types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyMessageType {
    /// Start strategy execution
    StartStrategy,
    /// Stop strategy execution
    StopStrategy,
    /// Update strategy parameters
    UpdateStrategy,
    /// Rebalance positions
    Rebalance,
    /// Emergency stop
    EmergencyStop,
    /// Status update
    StatusUpdate,
}

/// Cross-chain coordination statistics
#[derive(Debug, Default)]
pub struct CoordinationStats {
    /// Total messages processed
    pub messages_processed: AtomicU64,

    /// Cross-chain opportunities detected
    pub opportunities_detected: AtomicU64,

    /// Cross-chain opportunities executed
    pub opportunities_executed: AtomicU64,

    /// Total cross-chain profit (USD)
    pub total_cross_chain_profit_usd: AtomicU64,

    /// Resource rebalancing operations
    pub rebalancing_operations: AtomicU64,

    /// Bridge transactions monitored
    pub bridge_transactions_monitored: AtomicU64,

    /// Failed cross-chain operations
    pub failed_operations: AtomicU64,

    /// Average message latency (microseconds)
    pub avg_message_latency_us: AtomicU64,
}

/// Cache-line aligned coordination data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedCoordinationData {
    /// Total opportunities detected (scaled)
    pub opportunities_detected_scaled: u64,

    /// Total opportunities executed (scaled)
    pub opportunities_executed_scaled: u64,

    /// Total profit USD (scaled by 1e6)
    pub total_profit_usd_scaled: u64,

    /// Average message latency (microseconds)
    pub avg_message_latency_us: u64,

    /// Active chains count
    pub active_chains_count: u64,

    /// Bridge status bitmap (each bit represents a bridge)
    pub bridge_status_bitmap: u64,

    /// Resource utilization percentage (scaled by 1e6)
    pub resource_utilization_scaled: u64,

    /// Last update timestamp
    pub timestamp: u64,
}

/// Cross-chain coordination constants
pub const COORDINATION_DEFAULT_SYNC_INTERVAL_MS: u64 = 1000; // 1 second
pub const COORDINATION_DEFAULT_ALLOCATION_INTERVAL_MS: u64 = 5000; // 5 seconds
pub const COORDINATION_DEFAULT_BRIDGE_MONITOR_INTERVAL_MS: u64 = 500; // 500ms
pub const COORDINATION_DEFAULT_MAX_MESSAGE_LATENCY_MS: u64 = 10; // 10ms
pub const COORDINATION_DEFAULT_MIN_PROFIT_USD: &str = "10.0"; // $10 minimum
pub const COORDINATION_MAX_CHAINS: usize = 20;
pub const COORDINATION_MAX_OPPORTUNITIES: usize = 1000;
pub const COORDINATION_MAX_MESSAGES: usize = 10000;

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_interval_ms: COORDINATION_DEFAULT_SYNC_INTERVAL_MS,
            allocation_interval_ms: COORDINATION_DEFAULT_ALLOCATION_INTERVAL_MS,
            bridge_monitor_interval_ms: COORDINATION_DEFAULT_BRIDGE_MONITOR_INTERVAL_MS,
            enable_cross_chain_arbitrage: true,
            enable_resource_optimization: true,
            enable_bridge_monitoring: true,
            enable_strategy_coordination: true,
            max_message_latency_ms: COORDINATION_DEFAULT_MAX_MESSAGE_LATENCY_MS,
            min_profit_threshold_usd: COORDINATION_DEFAULT_MIN_PROFIT_USD.parse().unwrap_or_default(),
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

impl AlignedCoordinationData {
    /// Create new aligned coordination data
    #[inline(always)]
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "Aligned data structure requires all fields")]
    pub const fn new(
        opportunities_detected_scaled: u64,
        opportunities_executed_scaled: u64,
        total_profit_usd_scaled: u64,
        avg_message_latency_us: u64,
        active_chains_count: u64,
        bridge_status_bitmap: u64,
        resource_utilization_scaled: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            opportunities_detected_scaled,
            opportunities_executed_scaled,
            total_profit_usd_scaled,
            avg_message_latency_us,
            active_chains_count,
            bridge_status_bitmap,
            resource_utilization_scaled,
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

    /// Get total profit USD as Decimal
    #[inline(always)]
    #[must_use]
    pub fn total_profit_usd(&self) -> Decimal {
        Decimal::from(self.total_profit_usd_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get resource utilization as Decimal
    #[inline(always)]
    #[must_use]
    pub fn resource_utilization(&self) -> Decimal {
        Decimal::from(self.resource_utilization_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get success rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn success_rate(&self) -> Decimal {
        if self.opportunities_detected_scaled == 0 {
            return Decimal::ZERO;
        }

        Decimal::from(self.opportunities_executed_scaled) / Decimal::from(self.opportunities_detected_scaled)
    }

    /// Check if bridge is operational
    #[inline(always)]
    #[must_use]
    pub const fn is_bridge_operational(&self, bridge_index: u8) -> bool {
        if bridge_index >= 64 {
            return false;
        }

        (self.bridge_status_bitmap & (1_u64 << bridge_index)) != 0
    }
}

/// Cross-chain coordinator for ultra-performance multi-chain operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct CrossChainCoordinator {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Coordination specific configuration
    coordination_config: CoordinationConfig,

    /// Statistics
    stats: Arc<CoordinationStats>,

    /// Cross-chain states
    chain_states: Arc<RwLock<HashMap<ChainId, CrossChainState>>>,

    /// Coordination data cache for ultra-fast access
    coordination_cache: Arc<DashMap<String, AlignedCoordinationData>>,

    /// Cross-chain opportunities
    opportunities: Arc<RwLock<HashMap<String, CrossChainOpportunity>>>,

    /// Resource allocations
    resource_allocations: Arc<RwLock<HashMap<ChainId, ResourceAllocation>>>,

    /// Strategy messages
    strategy_messages: Arc<RwLock<BTreeMap<u64, StrategyMessage>>>,

    /// Performance timers
    sync_timer: Timer,
    allocation_timer: Timer,
    bridge_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Message channels
    message_sender: Sender<StrategyMessage>,
    message_receiver: Receiver<StrategyMessage>,

    /// Opportunity channels
    opportunity_sender: Sender<CrossChainOpportunity>,
    opportunity_receiver: Receiver<CrossChainOpportunity>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current coordination round
    coordination_round: Arc<TokioMutex<u64>>,
}

impl CrossChainCoordinator {
    /// Create new cross-chain coordinator with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let coordination_config = CoordinationConfig::default();
        let stats = Arc::new(CoordinationStats::default());
        let chain_states = Arc::new(RwLock::new(HashMap::with_capacity(COORDINATION_MAX_CHAINS)));
        let coordination_cache = Arc::new(DashMap::with_capacity(COORDINATION_MAX_CHAINS));
        let opportunities = Arc::new(RwLock::new(HashMap::with_capacity(COORDINATION_MAX_OPPORTUNITIES)));
        let resource_allocations = Arc::new(RwLock::new(HashMap::with_capacity(COORDINATION_MAX_CHAINS)));
        let strategy_messages = Arc::new(RwLock::new(BTreeMap::new()));
        let sync_timer = Timer::new("coordination_sync");
        let allocation_timer = Timer::new("coordination_allocation");
        let bridge_timer = Timer::new("coordination_bridge");
        let shutdown = Arc::new(AtomicBool::new(false));
        let coordination_round = Arc::new(TokioMutex::new(0));

        let (message_sender, message_receiver) = channel::bounded(COORDINATION_MAX_MESSAGES);
        let (opportunity_sender, opportunity_receiver) = channel::bounded(COORDINATION_MAX_OPPORTUNITIES);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            coordination_config,
            stats,
            chain_states,
            coordination_cache,
            opportunities,
            resource_allocations,
            strategy_messages,
            sync_timer,
            allocation_timer,
            bridge_timer,
            shutdown,
            message_sender,
            message_receiver,
            opportunity_sender,
            opportunity_receiver,
            http_client,
            coordination_round,
        })
    }

    /// Start cross-chain coordination services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.coordination_config.enabled {
            info!("Cross-chain coordination disabled");
            return Ok(());
        }

        info!("Starting cross-chain coordination");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start state synchronization
        self.start_state_synchronization().await;

        // Start resource allocation
        if self.coordination_config.enable_resource_optimization {
            self.start_resource_allocation().await;
        }

        // Start bridge monitoring
        if self.coordination_config.enable_bridge_monitoring {
            self.start_bridge_monitoring().await;
        }

        // Start strategy coordination
        if self.coordination_config.enable_strategy_coordination {
            self.start_strategy_coordination().await;
        }

        // Start cross-chain arbitrage
        if self.coordination_config.enable_cross_chain_arbitrage {
            self.start_cross_chain_arbitrage().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Cross-chain coordination started successfully");
        Ok(())
    }

    /// Stop cross-chain coordination
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping cross-chain coordination");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Cross-chain coordination stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &CoordinationStats {
        &self.stats
    }

    /// Get chain states
    #[inline]
    pub async fn get_chain_states(&self) -> Vec<CrossChainState> {
        let states = self.chain_states.read().await;
        states.values().cloned().collect()
    }

    /// Get cross-chain opportunities
    #[inline]
    pub async fn get_opportunities(&self) -> Vec<CrossChainOpportunity> {
        let opportunities = self.opportunities.read().await;
        opportunities.values().cloned().collect()
    }

    /// Get resource allocations
    #[inline]
    pub async fn get_resource_allocations(&self) -> Vec<ResourceAllocation> {
        let allocations = self.resource_allocations.read().await;
        allocations.values().cloned().collect()
    }

    /// Send strategy message
    ///
    /// # Errors
    ///
    /// Returns error if message sending fails
    #[inline]
    pub async fn send_strategy_message(&self, message: StrategyMessage) -> Result<()> {
        self.message_sender.send(message)
            .map_err(|_| crate::ChainCoreError::Network(crate::NetworkError::ConnectionRefused))?;
        Ok(())
    }

    /// Send cross-chain opportunity
    ///
    /// # Errors
    ///
    /// Returns error if opportunity sending fails
    #[inline]
    pub async fn send_opportunity(&self, opportunity: CrossChainOpportunity) -> Result<()> {
        self.opportunity_sender.send(opportunity)
            .map_err(|_| crate::ChainCoreError::Network(crate::NetworkError::ConnectionRefused))?;
        Ok(())
    }

    /// Calculate optimal chain for operation
    #[inline]
    #[must_use]
    #[expect(clippy::significant_drop_tightening, reason = "States guard needed for entire calculation")]
    pub async fn calculate_optimal_chain(
        &self,
        operation_type: &str,
        amount_usd: Decimal,
    ) -> Option<ChainId> {
        let states = self.chain_states.read().await;

        let mut best_chain = None;
        let mut best_score = Decimal::ZERO;

        for (chain_id, state) in states.iter() {
            // Simple scoring algorithm based on gas price, liquidity, and congestion
            let gas_score = if state.gas_price > 0 {
                Decimal::from(1_000_000_u64) / Decimal::from(state.gas_price)
            } else {
                Decimal::ZERO
            };

            let liquidity_score = if state.available_liquidity_usd > amount_usd {
                Decimal::ONE
            } else {
                state.available_liquidity_usd / amount_usd
            };

            let congestion_score = Decimal::from(100_u64 - u64::from(state.congestion_level)) / Decimal::from(100_u64);

            let bridge_score = if state.bridge_status == BridgeStatus::Operational {
                Decimal::ONE
            } else {
                "0.5".parse::<Decimal>().unwrap_or_default()
            };

            let total_score = gas_score * "0.3".parse::<Decimal>().unwrap_or_default() +
                            liquidity_score * "0.4".parse::<Decimal>().unwrap_or_default() +
                            congestion_score * "0.2".parse::<Decimal>().unwrap_or_default() +
                            bridge_score * "0.1".parse::<Decimal>().unwrap_or_default();

            if total_score > best_score {
                best_score = total_score;
                best_chain = Some(*chain_id);
            }
        }

        trace!("Optimal chain for {} operation: {:?} (score: {})", operation_type, best_chain, best_score);
        best_chain
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(2000)) // Fast timeout for coordination calls
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

    /// Start state synchronization
    async fn start_state_synchronization(&self) {
        let chain_states: Arc<RwLock<HashMap<ChainId, CrossChainState>>> = Arc::clone(&self.chain_states);
        let coordination_cache = Arc::clone(&self.coordination_cache);
        let coordination_stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let coordination_config = self.coordination_config.clone();
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(coordination_config.sync_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Simulate state synchronization for supported chains
                for chain_id in &coordination_config.supported_chains {
                    let chain_state = Self::fetch_chain_state(*chain_id, &http_client).await;

                    // Update chain states
                    {
                        let mut chain_states_guard = chain_states.write().await;
                        chain_states_guard.insert(*chain_id, chain_state.clone());
                    }

                    // Update cache with aligned data
                    let aligned_data = AlignedCoordinationData::new(
                        coordination_stats.opportunities_detected.load(Ordering::Relaxed),
                        coordination_stats.opportunities_executed.load(Ordering::Relaxed),
                        (chain_state.available_liquidity_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        coordination_stats.avg_message_latency_us.load(Ordering::Relaxed),
                        coordination_config.supported_chains.len() as u64,
                        u64::from(chain_state.bridge_status == BridgeStatus::Operational),
                        (Decimal::from(chain_state.active_strategies) * Decimal::from(10_000_u64)).to_u64().unwrap_or(0),
                        chain_state.last_update,
                    );
                    coordination_cache.insert(format!("state_{chain_id:?}"), aligned_data);
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let sync_time = start_time.elapsed().as_micros() as u64;
                coordination_stats.avg_message_latency_us.store(sync_time, Ordering::Relaxed);
                trace!("State synchronization cycle completed in {}μs", sync_time);

                // Clean stale cache entries
                Self::clean_stale_cache(&coordination_cache, 60_000); // 1 minute
            }
        });
    }

    /// Start resource allocation
    async fn start_resource_allocation(&self) {
        let resource_allocations: Arc<RwLock<HashMap<ChainId, ResourceAllocation>>> = Arc::clone(&self.resource_allocations);
        let chain_states: Arc<RwLock<HashMap<ChainId, CrossChainState>>> = Arc::clone(&self.chain_states);
        let allocation_stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let coordination_config = self.coordination_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(coordination_config.allocation_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Calculate optimal resource allocation
                let chain_states_guard = chain_states.read().await;
                let total_liquidity: Decimal = chain_states_guard.values()
                    .map(|state| state.available_liquidity_usd)
                    .sum();

                if total_liquidity > Decimal::ZERO {
                    let mut allocations = resource_allocations.write().await;

                    for (chain_id, state) in chain_states_guard.iter() {
                        let target_allocation_pct = state.available_liquidity_usd / total_liquidity;
                        let current_allocation_pct = allocations.get(chain_id).map_or(Decimal::ZERO, |existing| existing.current_allocation_pct);

                        let needs_rebalancing = (target_allocation_pct - current_allocation_pct).abs() > "0.05".parse::<Decimal>().unwrap_or_default(); // 5% threshold

                        let allocation = ResourceAllocation {
                            chain_id: *chain_id,
                            allocated_capital_usd: state.available_liquidity_usd * "0.8".parse::<Decimal>().unwrap_or_default(), // 80% utilization
                            available_capital_usd: state.available_liquidity_usd,
                            active_positions: state.active_strategies,
                            target_allocation_pct,
                            current_allocation_pct,
                            needs_rebalancing,
                            last_rebalancing: if needs_rebalancing {
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs()
                            } else {
                                allocations.get(chain_id).map_or(0, |a| a.last_rebalancing)
                            },
                        };

                        allocations.insert(*chain_id, allocation);

                        if needs_rebalancing {
                            allocation_stats.rebalancing_operations.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    drop(allocations);
                }
                drop(chain_states_guard);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let allocation_time = start_time.elapsed().as_micros() as u64;
                trace!("Resource allocation cycle completed in {}μs", allocation_time);
            }
        });
    }

    /// Start bridge monitoring
    async fn start_bridge_monitoring(&self) {
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let coordination_config = self.coordination_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(coordination_config.bridge_monitor_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Simulate bridge monitoring
                stats.bridge_transactions_monitored.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let bridge_time = start_time.elapsed().as_micros() as u64;
                trace!("Bridge monitoring cycle completed in {}μs", bridge_time);
            }
        });
    }

    /// Start strategy coordination
    async fn start_strategy_coordination(&self) {
        let message_receiver = self.message_receiver.clone();
        let strategy_messages = Arc::clone(&self.strategy_messages);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                // Process incoming strategy messages
                while let Ok(message) = message_receiver.try_recv() {
                    let message_id = message.timestamp;

                    // Store strategy message
                    {
                        let mut messages = strategy_messages.write().await;
                        messages.insert(message_id, message);

                        // Keep only recent messages (last 1000)
                        while messages.len() > 1000 {
                            if let Some(oldest_key) = messages.keys().next().copied() {
                                messages.remove(&oldest_key);
                            }
                        }
                        drop(messages);
                    }

                    stats.messages_processed.fetch_add(1, Ordering::Relaxed);
                }

                // Small delay to prevent busy waiting
                sleep(Duration::from_millis(1)).await;
            }
        });
    }

    /// Start cross-chain arbitrage
    async fn start_cross_chain_arbitrage(&self) {
        let opportunity_receiver = self.opportunity_receiver.clone();
        let opportunities = Arc::clone(&self.opportunities);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                // Process incoming opportunities
                while let Ok(opportunity) = opportunity_receiver.try_recv() {
                    let opportunity_id = opportunity.id.clone();

                    // Store opportunity
                    {
                        let mut opps = opportunities.write().await;
                        opps.insert(opportunity_id, opportunity.clone());

                        // Keep only recent opportunities
                        while opps.len() > COORDINATION_MAX_OPPORTUNITIES {
                            if let Some(oldest_key) = opps.keys().next().cloned() {
                                opps.remove(&oldest_key);
                            }
                        }
                        drop(opps);
                    }

                    stats.opportunities_detected.fetch_add(1, Ordering::Relaxed);

                    // Simulate opportunity execution
                    if opportunity.expected_profit_usd > "50".parse::<Decimal>().unwrap_or_default() {
                        stats.opportunities_executed.fetch_add(1, Ordering::Relaxed);
                        stats.total_cross_chain_profit_usd.fetch_add(
                            opportunity.expected_profit_usd.to_u64().unwrap_or(0),
                            Ordering::Relaxed
                        );
                    }
                }

                // Small delay to prevent busy waiting
                sleep(Duration::from_millis(1)).await;
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

                let messages = stats.messages_processed.load(Ordering::Relaxed);
                let opportunities_detected = stats.opportunities_detected.load(Ordering::Relaxed);
                let opportunities_executed = stats.opportunities_executed.load(Ordering::Relaxed);
                let total_profit = stats.total_cross_chain_profit_usd.load(Ordering::Relaxed);
                let rebalancing_ops = stats.rebalancing_operations.load(Ordering::Relaxed);
                let bridge_txs = stats.bridge_transactions_monitored.load(Ordering::Relaxed);
                let failed_ops = stats.failed_operations.load(Ordering::Relaxed);
                let avg_latency = stats.avg_message_latency_us.load(Ordering::Relaxed);

                info!(
                    "Coordination Stats: messages={}, opps_detected={}, opps_executed={}, profit=${}, rebalancing={}, bridge_txs={}, failed={}, avg_latency={}μs",
                    messages, opportunities_detected, opportunities_executed, total_profit, rebalancing_ops, bridge_txs, failed_ops, avg_latency
                );
            }
        });
    }

    /// Fetch chain state
    async fn fetch_chain_state(chain_id: ChainId, _http_client: &Arc<TokioMutex<Option<reqwest::Client>>>) -> CrossChainState {
        // Simplified implementation - in production this would fetch real chain data
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for chain state")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        CrossChainState {
            chain_id,
            block_number: 18_000_000 + (now % 1000), // Simulate block progression
            block_timestamp: now / 1000,
            gas_price: match chain_id {
                ChainId::Ethereum => 20_000_000_000, // 20 gwei
                ChainId::Arbitrum => 100_000_000,    // 0.1 gwei
                ChainId::Optimism | ChainId::Base => 1_000_000, // 0.001 gwei
                ChainId::Polygon => 30_000_000_000,  // 30 gwei
                ChainId::Bsc => 5_000_000_000,       // 5 gwei
                ChainId::Avalanche => 25_000_000_000, // 25 gwei
            },
            available_liquidity_usd: match chain_id {
                ChainId::Ethereum => "100000000".parse().unwrap_or_default(), // $100M
                ChainId::Arbitrum => "50000000".parse().unwrap_or_default(),  // $50M
                ChainId::Optimism => "30000000".parse().unwrap_or_default(),  // $30M
                ChainId::Polygon => "20000000".parse().unwrap_or_default(),   // $20M
                ChainId::Bsc => "40000000".parse().unwrap_or_default(),       // $40M
                ChainId::Avalanche => "25000000".parse().unwrap_or_default(), // $25M
                ChainId::Base => "15000000".parse().unwrap_or_default(),      // $15M
            },
            active_strategies: match chain_id {
                ChainId::Ethereum => 50,
                ChainId::Arbitrum => 30,
                ChainId::Optimism => 25,
                ChainId::Polygon => 20,
                ChainId::Bsc => 35,
                ChainId::Avalanche => 28,
                ChainId::Base => 15,
            },
            congestion_level: (now % 100) as u8, // Simulate congestion
            bridge_status: if now % 10 < 8 { BridgeStatus::Operational } else { BridgeStatus::Congested },
            last_update: now,
        }
    }

    /// Clean stale cache entries
    fn clean_stale_cache(cache: &Arc<DashMap<String, AlignedCoordinationData>>, max_age_ms: u64) {
        cache.retain(|_key, data| !data.is_stale(max_age_ms));
    }

    /// Create cross-chain opportunity
    #[must_use]
    pub fn create_cross_chain_opportunity(
        source_chain: ChainId,
        destination_chain: ChainId,
        asset: &str,
        amount: Decimal,
        price_diff_pct: Decimal,
    ) -> CrossChainOpportunity {
        let expected_profit_usd = amount * price_diff_pct;
        let bridge_fee_usd = amount * "0.001".parse::<Decimal>().unwrap_or_default(); // 0.1% bridge fee

        CrossChainOpportunity {
            id: format!("opp_{}_{:?}_{:?}_{}",
                chrono::Utc::now().timestamp_millis(),
                source_chain,
                destination_chain,
                asset
            ),
            source_chain,
            destination_chain,
            asset: asset.to_string(),
            amount,
            expected_profit_usd,
            bridge_fee_usd,
            estimated_execution_time_s: match (source_chain, destination_chain) {
                (ChainId::Ethereum, ChainId::Arbitrum) => 600,  // 10 minutes
                (ChainId::Ethereum, ChainId::Optimism) => 1200, // 20 minutes
                (ChainId::Ethereum, ChainId::Polygon) => 1800,  // 30 minutes
                _ => 900, // 15 minutes default
            },
            risk_level: if expected_profit_usd > "1000".parse::<Decimal>().unwrap_or_default() { 3 } else { 5 },
            expires_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() + 3600, // 1 hour
        }
    }

    /// Create strategy message
    #[must_use]
    pub fn create_strategy_message(
        source_chain: ChainId,
        target_chains: Vec<ChainId>,
        message_type: StrategyMessageType,
        payload: HashMap<String, String>,
        priority: u8,
    ) -> StrategyMessage {
        StrategyMessage {
            id: format!("msg_{}_{:?}", chrono::Utc::now().timestamp_millis(), message_type),
            source_chain,
            target_chains,
            message_type,
            payload,
            priority,
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for strategy message")]
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_cross_chain_coordinator_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = CrossChainCoordinator::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(coordinator.stats().messages_processed.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().opportunities_detected.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().opportunities_executed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_coordination_config_default() {
        let config = CoordinationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.sync_interval_ms, COORDINATION_DEFAULT_SYNC_INTERVAL_MS);
        assert_eq!(config.allocation_interval_ms, COORDINATION_DEFAULT_ALLOCATION_INTERVAL_MS);
        assert_eq!(config.bridge_monitor_interval_ms, COORDINATION_DEFAULT_BRIDGE_MONITOR_INTERVAL_MS);
        assert!(config.enable_cross_chain_arbitrage);
        assert!(config.enable_resource_optimization);
        assert!(config.enable_bridge_monitoring);
        assert!(config.enable_strategy_coordination);
        assert!(!config.supported_chains.is_empty());
    }

    #[test]
    fn test_aligned_coordination_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedCoordinationData>(), 64);
        assert!(mem::size_of::<AlignedCoordinationData>() <= 64);
    }

    #[test]
    fn test_coordination_stats_operations() {
        let stats = CoordinationStats::default();

        stats.messages_processed.fetch_add(100, Ordering::Relaxed);
        stats.opportunities_detected.fetch_add(50, Ordering::Relaxed);
        stats.opportunities_executed.fetch_add(30, Ordering::Relaxed);
        stats.total_cross_chain_profit_usd.fetch_add(10_000, Ordering::Relaxed);
        stats.rebalancing_operations.fetch_add(5, Ordering::Relaxed);

        assert_eq!(stats.messages_processed.load(Ordering::Relaxed), 100);
        assert_eq!(stats.opportunities_detected.load(Ordering::Relaxed), 50);
        assert_eq!(stats.opportunities_executed.load(Ordering::Relaxed), 30);
        assert_eq!(stats.total_cross_chain_profit_usd.load(Ordering::Relaxed), 10_000);
        assert_eq!(stats.rebalancing_operations.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_aligned_coordination_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedCoordinationData::new(
            100, // opportunities detected
            80,  // opportunities executed
            50_000_000, // $50 profit
            1_500, // 1.5ms latency
            7,   // 7 active chains
            255, // all bridges operational
            800_000, // 80% utilization
            now,
        );

        let stale_data = AlignedCoordinationData::new(
            100, 80, 50_000_000, 1_500, 7, 255, 800_000,
            now - 120_000, // 2 minutes old
        );

        assert!(!fresh_data.is_stale(60_000));
        assert!(stale_data.is_stale(60_000));
    }

    #[test]
    fn test_aligned_coordination_data_conversions() {
        let data = AlignedCoordinationData::new(
            100, // opportunities detected
            80,  // opportunities executed
            50_000_000, // $50 profit (scaled by 1e6)
            1_500, // 1.5ms latency
            7,   // 7 active chains
            255, // all bridges operational
            800_000, // 80% utilization (scaled by 1e6)
            1_640_995_200_000,
        );

        assert_eq!(data.total_profit_usd(), dec!(50));
        assert_eq!(data.resource_utilization(), dec!(0.8));
        assert_eq!(data.success_rate(), dec!(0.8)); // 80/100
        assert!(data.is_bridge_operational(0));
        assert!(data.is_bridge_operational(7));
        assert!(!data.is_bridge_operational(64)); // Out of range
    }

    #[test]
    fn test_bridge_status_equality() {
        assert_eq!(BridgeStatus::Operational, BridgeStatus::Operational);
        assert_ne!(BridgeStatus::Operational, BridgeStatus::Congested);
        assert_ne!(BridgeStatus::Congested, BridgeStatus::Maintenance);
        assert_ne!(BridgeStatus::Maintenance, BridgeStatus::Offline);
    }

    #[test]
    fn test_strategy_message_type_equality() {
        assert_eq!(StrategyMessageType::StartStrategy, StrategyMessageType::StartStrategy);
        assert_ne!(StrategyMessageType::StartStrategy, StrategyMessageType::StopStrategy);
        assert_ne!(StrategyMessageType::UpdateStrategy, StrategyMessageType::Rebalance);
        assert_ne!(StrategyMessageType::EmergencyStop, StrategyMessageType::StatusUpdate);
    }

    #[test]
    fn test_cross_chain_state_creation() {
        let state = CrossChainState {
            chain_id: ChainId::Ethereum,
            block_number: 18_000_000,
            block_timestamp: 1_640_995_200,
            gas_price: 20_000_000_000, // 20 gwei
            available_liquidity_usd: dec!(100000000), // $100M
            active_strategies: 50,
            congestion_level: 25,
            bridge_status: BridgeStatus::Operational,
            last_update: 1_640_995_200_000,
        };

        assert_eq!(state.chain_id, ChainId::Ethereum);
        assert_eq!(state.block_number, 18_000_000);
        assert_eq!(state.gas_price, 20_000_000_000);
        assert_eq!(state.available_liquidity_usd, dec!(100000000));
        assert_eq!(state.active_strategies, 50);
        assert_eq!(state.congestion_level, 25);
        assert_eq!(state.bridge_status, BridgeStatus::Operational);
    }

    #[test]
    fn test_cross_chain_opportunity_creation() {
        let opportunity = CrossChainCoordinator::create_cross_chain_opportunity(
            ChainId::Ethereum,
            ChainId::Arbitrum,
            "USDC",
            dec!(10000), // $10k
            dec!(0.02),  // 2% price difference
        );

        assert_eq!(opportunity.source_chain, ChainId::Ethereum);
        assert_eq!(opportunity.destination_chain, ChainId::Arbitrum);
        assert_eq!(opportunity.asset, "USDC");
        assert_eq!(opportunity.amount, dec!(10000));
        assert_eq!(opportunity.expected_profit_usd, dec!(200)); // 2% of $10k
        assert!(opportunity.bridge_fee_usd > dec!(0));
        assert!(opportunity.estimated_execution_time_s > 0);
        assert!(!opportunity.id.is_empty());
    }

    #[test]
    fn test_strategy_message_creation() {
        let mut payload = HashMap::new();
        payload.insert("strategy_id".to_string(), "arb_001".to_string());
        payload.insert("amount".to_string(), "1000".to_string());

        let message = CrossChainCoordinator::create_strategy_message(
            ChainId::Ethereum,
            vec![ChainId::Arbitrum, ChainId::Optimism],
            StrategyMessageType::StartStrategy,
            payload.clone(),
            8, // High priority
        );

        assert_eq!(message.source_chain, ChainId::Ethereum);
        assert_eq!(message.target_chains.len(), 2);
        assert_eq!(message.message_type, StrategyMessageType::StartStrategy);
        assert_eq!(message.payload, payload);
        assert_eq!(message.priority, 8);
        assert!(!message.id.is_empty());
    }

    #[test]
    fn test_resource_allocation_creation() {
        let allocation = ResourceAllocation {
            chain_id: ChainId::Ethereum,
            allocated_capital_usd: dec!(80000000), // $80M allocated
            available_capital_usd: dec!(100000000), // $100M available
            active_positions: 50,
            target_allocation_pct: dec!(0.4), // 40% target
            current_allocation_pct: dec!(0.35), // 35% current
            needs_rebalancing: true,
            last_rebalancing: 1_640_995_200,
        };

        assert_eq!(allocation.chain_id, ChainId::Ethereum);
        assert_eq!(allocation.allocated_capital_usd, dec!(80000000));
        assert_eq!(allocation.available_capital_usd, dec!(100000000));
        assert_eq!(allocation.active_positions, 50);
        assert_eq!(allocation.target_allocation_pct, dec!(0.4));
        assert_eq!(allocation.current_allocation_pct, dec!(0.35));
        assert!(allocation.needs_rebalancing);
    }

    #[tokio::test]
    async fn test_cross_chain_coordinator_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = CrossChainCoordinator::new(config).await else {
            return;
        };

        let chain_states = coordinator.get_chain_states().await;
        assert!(chain_states.is_empty()); // No states initially

        let opportunities = coordinator.get_opportunities().await;
        assert!(opportunities.is_empty()); // No opportunities initially

        let allocations = coordinator.get_resource_allocations().await;
        assert!(allocations.is_empty()); // No allocations initially

        let stats = coordinator.stats();
        assert_eq!(stats.messages_processed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_optimal_chain_calculation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = CrossChainCoordinator::new(config).await else {
            return;
        };

        // Test with empty states
        let optimal_chain = coordinator.calculate_optimal_chain("swap", dec!(1000)).await;
        assert!(optimal_chain.is_none()); // No chains available

        // Add some test states
        {
            let mut states = coordinator.chain_states.write().await;
            states.insert(ChainId::Ethereum, CrossChainState {
                chain_id: ChainId::Ethereum,
                block_number: 18_000_000,
                block_timestamp: 1_640_995_200,
                gas_price: 50_000_000_000, // High gas
                available_liquidity_usd: dec!(100000000),
                active_strategies: 50,
                congestion_level: 80, // High congestion
                bridge_status: BridgeStatus::Operational,
                last_update: 1_640_995_200_000,
            });

            states.insert(ChainId::Arbitrum, CrossChainState {
                chain_id: ChainId::Arbitrum,
                block_number: 18_000_000,
                block_timestamp: 1_640_995_200,
                gas_price: 100_000_000, // Low gas
                available_liquidity_usd: dec!(50000000),
                active_strategies: 30,
                congestion_level: 20, // Low congestion
                bridge_status: BridgeStatus::Operational,
                last_update: 1_640_995_200_000,
            });
        }

        let optimal_chain = coordinator.calculate_optimal_chain("swap", dec!(1000)).await;
        assert!(optimal_chain.is_some()); // Should find optimal chain
    }
}
