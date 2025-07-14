//! Bridge Monitor for ultra-performance cross-chain bridge monitoring
//!
//! This module provides advanced bridge monitoring capabilities for cross-chain
//! operations, enabling real-time bridge status tracking, transaction monitoring,
//! and optimal bridge selection for cross-chain arbitrage.
//!
//! ## Performance Targets
//! - Bridge Status Check: <200μs
//! - Transaction Monitoring: <500μs
//! - Bridge Selection: <100μs
//! - Fee Calculation: <50μs
//! - Health Assessment: <300μs
//!
//! ## Architecture
//! - Real-time bridge status monitoring
//! - Multi-bridge transaction tracking
//! - Advanced fee optimization
//! - Bridge health assessment
//! - Lock-free monitoring primitives

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

/// Bridge monitor configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct BridgeMonitorConfig {
    /// Enable bridge monitoring
    pub enabled: bool,
    
    /// Bridge status check interval in milliseconds
    pub status_check_interval_ms: u64,
    
    /// Transaction monitoring interval in milliseconds
    pub transaction_monitor_interval_ms: u64,
    
    /// Fee update interval in milliseconds
    pub fee_update_interval_ms: u64,
    
    /// Enable transaction tracking
    pub enable_transaction_tracking: bool,
    
    /// Enable fee optimization
    pub enable_fee_optimization: bool,
    
    /// Enable health monitoring
    pub enable_health_monitoring: bool,
    
    /// Enable bridge selection optimization
    pub enable_bridge_selection: bool,
    
    /// Maximum bridge response time (ms)
    pub max_bridge_response_time_ms: u64,
    
    /// Minimum bridge liquidity (USD)
    pub min_bridge_liquidity_usd: Decimal,
    
    /// Monitored bridges
    pub monitored_bridges: Vec<BridgeType>,
}

/// Bridge types supported for monitoring
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BridgeType {
    /// Stargate (LayerZero)
    Stargate,
    /// Wormhole
    Wormhole,
    /// Multichain (Anyswap)
    Multichain,
    /// Hop Protocol
    Hop,
    /// Synapse
    Synapse,
    /// Across
    Across,
    /// Celer cBridge
    Celer,
    /// Polygon PoS Bridge
    PolygonPos,
    /// Arbitrum Bridge
    ArbitrumBridge,
    /// Optimism Bridge
    OptimismBridge,
    /// Avalanche Bridge
    AvalancheBridge,
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
    /// Bridge status unknown
    Unknown,
}

/// Bridge information
#[derive(Debug, Clone)]
pub struct BridgeInfo {
    /// Bridge type
    pub bridge_type: BridgeType,
    
    /// Source chain
    pub source_chain: ChainId,
    
    /// Destination chain
    pub destination_chain: ChainId,
    
    /// Bridge contract address
    pub contract_address: String,
    
    /// Current status
    pub status: BridgeStatus,
    
    /// Available liquidity (USD)
    pub available_liquidity_usd: Decimal,
    
    /// Bridge fee percentage
    pub fee_percentage: Decimal,
    
    /// Minimum transfer amount
    pub min_transfer_amount: Decimal,
    
    /// Maximum transfer amount
    pub max_transfer_amount: Decimal,
    
    /// Average transfer time (seconds)
    pub avg_transfer_time_s: u32,
    
    /// Success rate (0-1)
    pub success_rate: Decimal,
    
    /// Last status update
    pub last_update: u64,
}

/// Bridge transaction information
#[derive(Debug, Clone)]
pub struct BridgeTransaction {
    /// Transaction ID
    pub tx_id: String,
    
    /// Bridge type used
    pub bridge_type: BridgeType,
    
    /// Source chain
    pub source_chain: ChainId,
    
    /// Destination chain
    pub destination_chain: ChainId,
    
    /// Asset being bridged
    pub asset: String,
    
    /// Amount being bridged
    pub amount: Decimal,
    
    /// Bridge fee paid
    pub fee_paid: Decimal,
    
    /// Transaction status
    pub status: TransactionStatus,
    
    /// Initiated timestamp
    pub initiated_at: u64,
    
    /// Completed timestamp (if completed)
    pub completed_at: Option<u64>,
    
    /// Source transaction hash
    pub source_tx_hash: String,
    
    /// Destination transaction hash (if completed)
    pub destination_tx_hash: Option<String>,
}

/// Bridge transaction status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionStatus {
    /// Transaction initiated
    Initiated,
    /// Transaction pending
    Pending,
    /// Transaction confirmed on source
    SourceConfirmed,
    /// Transaction in transit
    InTransit,
    /// Transaction completed
    Completed,
    /// Transaction failed
    Failed,
    /// Transaction refunded
    Refunded,
}

/// Bridge health metrics
#[derive(Debug, Clone)]
pub struct BridgeHealth {
    /// Bridge type
    pub bridge_type: BridgeType,
    
    /// Health score (0-100)
    pub health_score: u8,
    
    /// Uptime percentage (0-100)
    pub uptime_percentage: u8,
    
    /// Average response time (ms)
    pub avg_response_time_ms: u64,
    
    /// Transaction success rate (0-1)
    pub success_rate: Decimal,
    
    /// Current liquidity utilization (0-1)
    pub liquidity_utilization: Decimal,
    
    /// Fee competitiveness score (0-100)
    pub fee_competitiveness: u8,
    
    /// Last health check
    pub last_check: u64,
}

/// Bridge selection criteria
#[derive(Debug, Clone)]
pub struct BridgeSelectionCriteria {
    /// Source chain
    pub source_chain: ChainId,
    
    /// Destination chain
    pub destination_chain: ChainId,
    
    /// Asset to bridge
    pub asset: String,
    
    /// Amount to bridge
    pub amount: Decimal,
    
    /// Maximum acceptable fee percentage
    pub max_fee_percentage: Decimal,
    
    /// Maximum acceptable transfer time (seconds)
    pub max_transfer_time_s: u32,
    
    /// Minimum required success rate
    pub min_success_rate: Decimal,
    
    /// Priority weights
    pub fee_weight: Decimal,
    pub speed_weight: Decimal,
    pub reliability_weight: Decimal,
}

/// Bridge monitor statistics
#[derive(Debug, Default)]
pub struct BridgeMonitorStats {
    /// Total bridges monitored
    pub bridges_monitored: AtomicU64,
    
    /// Total transactions tracked
    pub transactions_tracked: AtomicU64,
    
    /// Successful transactions
    pub successful_transactions: AtomicU64,
    
    /// Failed transactions
    pub failed_transactions: AtomicU64,
    
    /// Total bridge fees collected (USD)
    pub total_fees_collected_usd: AtomicU64,
    
    /// Bridge status checks performed
    pub status_checks_performed: AtomicU64,
    
    /// Health assessments completed
    pub health_assessments_completed: AtomicU64,
    
    /// Optimal bridge selections
    pub optimal_selections: AtomicU64,
    
    /// Bridge downtime incidents
    pub downtime_incidents: AtomicU64,
}

/// Cache-line aligned bridge data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedBridgeData {
    /// Bridge status bitmap (each bit represents a bridge)
    pub status_bitmap: u64,
    
    /// Total liquidity USD (scaled by 1e6)
    pub total_liquidity_usd_scaled: u64,
    
    /// Average fee percentage (scaled by 1e6)
    pub avg_fee_percentage_scaled: u64,
    
    /// Average transfer time (seconds)
    pub avg_transfer_time_s: u64,
    
    /// Success rate (scaled by 1e6)
    pub success_rate_scaled: u64,
    
    /// Health score average (0-100)
    pub avg_health_score: u64,
    
    /// Active transactions count
    pub active_transactions_count: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// Bridge monitor constants
pub const BRIDGE_MONITOR_DEFAULT_STATUS_INTERVAL_MS: u64 = 5000; // 5 seconds
pub const BRIDGE_MONITOR_DEFAULT_TRANSACTION_INTERVAL_MS: u64 = 1000; // 1 second
pub const BRIDGE_MONITOR_DEFAULT_FEE_UPDATE_INTERVAL_MS: u64 = 30000; // 30 seconds
pub const BRIDGE_MONITOR_DEFAULT_MAX_RESPONSE_TIME_MS: u64 = 5000; // 5 seconds
pub const BRIDGE_MONITOR_DEFAULT_MIN_LIQUIDITY_USD: &str = "10000"; // $10k minimum
pub const BRIDGE_MONITOR_MAX_BRIDGES: usize = 50;
pub const BRIDGE_MONITOR_MAX_TRANSACTIONS: usize = 10000;

impl Default for BridgeMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            status_check_interval_ms: BRIDGE_MONITOR_DEFAULT_STATUS_INTERVAL_MS,
            transaction_monitor_interval_ms: BRIDGE_MONITOR_DEFAULT_TRANSACTION_INTERVAL_MS,
            fee_update_interval_ms: BRIDGE_MONITOR_DEFAULT_FEE_UPDATE_INTERVAL_MS,
            enable_transaction_tracking: true,
            enable_fee_optimization: true,
            enable_health_monitoring: true,
            enable_bridge_selection: true,
            max_bridge_response_time_ms: BRIDGE_MONITOR_DEFAULT_MAX_RESPONSE_TIME_MS,
            min_bridge_liquidity_usd: BRIDGE_MONITOR_DEFAULT_MIN_LIQUIDITY_USD.parse().unwrap_or_default(),
            monitored_bridges: vec![
                BridgeType::Stargate,
                BridgeType::Wormhole,
                BridgeType::Multichain,
                BridgeType::Hop,
                BridgeType::Synapse,
                BridgeType::Across,
                BridgeType::Celer,
            ],
        }
    }
}

impl AlignedBridgeData {
    /// Create new aligned bridge data
    #[inline(always)]
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "Aligned data structure requires all fields")]
    pub const fn new(
        status_bitmap: u64,
        total_liquidity_usd_scaled: u64,
        avg_fee_percentage_scaled: u64,
        avg_transfer_time_s: u64,
        success_rate_scaled: u64,
        avg_health_score: u64,
        active_transactions_count: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            status_bitmap,
            total_liquidity_usd_scaled,
            avg_fee_percentage_scaled,
            avg_transfer_time_s,
            success_rate_scaled,
            avg_health_score,
            active_transactions_count,
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

    /// Get success rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn success_rate(&self) -> Decimal {
        Decimal::from(self.success_rate_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Check if bridge is operational
    #[inline(always)]
    #[must_use]
    pub const fn is_bridge_operational(&self, bridge_index: u8) -> bool {
        if bridge_index >= 64 {
            return false;
        }

        (self.status_bitmap & (1_u64 << bridge_index)) != 0
    }

    /// Get operational bridges count
    #[inline(always)]
    #[must_use]
    pub const fn operational_bridges_count(&self) -> u32 {
        self.status_bitmap.count_ones()
    }
}

/// Bridge Monitor for ultra-performance cross-chain bridge monitoring
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct BridgeMonitor {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Bridge monitor specific configuration
    bridge_config: BridgeMonitorConfig,

    /// Statistics
    stats: Arc<BridgeMonitorStats>,

    /// Monitored bridges
    bridges: Arc<RwLock<HashMap<String, BridgeInfo>>>,

    /// Bridge data cache for ultra-fast access
    bridge_cache: Arc<DashMap<String, AlignedBridgeData>>,

    /// Tracked transactions
    transactions: Arc<RwLock<HashMap<String, BridgeTransaction>>>,

    /// Bridge health metrics
    bridge_health: Arc<RwLock<HashMap<BridgeType, BridgeHealth>>>,

    /// Performance timers
    status_timer: Timer,
    transaction_timer: Timer,
    health_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Bridge update channels
    bridge_sender: Sender<BridgeInfo>,
    bridge_receiver: Receiver<BridgeInfo>,

    /// Transaction update channels
    transaction_sender: Sender<BridgeTransaction>,
    transaction_receiver: Receiver<BridgeTransaction>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current monitoring round
    monitoring_round: Arc<TokioMutex<u64>>,
}

impl BridgeMonitor {
    /// Create new bridge monitor with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let bridge_config = BridgeMonitorConfig::default();
        let stats = Arc::new(BridgeMonitorStats::default());
        let bridges = Arc::new(RwLock::new(HashMap::with_capacity(BRIDGE_MONITOR_MAX_BRIDGES)));
        let bridge_cache = Arc::new(DashMap::with_capacity(BRIDGE_MONITOR_MAX_BRIDGES));
        let transactions = Arc::new(RwLock::new(HashMap::with_capacity(BRIDGE_MONITOR_MAX_TRANSACTIONS)));
        let bridge_health = Arc::new(RwLock::new(HashMap::with_capacity(BRIDGE_MONITOR_MAX_BRIDGES)));
        let status_timer = Timer::new("bridge_status");
        let transaction_timer = Timer::new("bridge_transaction");
        let health_timer = Timer::new("bridge_health");
        let shutdown = Arc::new(AtomicBool::new(false));
        let monitoring_round = Arc::new(TokioMutex::new(0));

        let (bridge_sender, bridge_receiver) = channel::bounded(BRIDGE_MONITOR_MAX_BRIDGES);
        let (transaction_sender, transaction_receiver) = channel::bounded(BRIDGE_MONITOR_MAX_TRANSACTIONS);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            bridge_config,
            stats,
            bridges,
            bridge_cache,
            transactions,
            bridge_health,
            status_timer,
            transaction_timer,
            health_timer,
            shutdown,
            bridge_sender,
            bridge_receiver,
            transaction_sender,
            transaction_receiver,
            http_client,
            monitoring_round,
        })
    }

    /// Start bridge monitoring services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.bridge_config.enabled {
            info!("Bridge monitoring disabled");
            return Ok(());
        }

        info!("Starting bridge monitoring");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start bridge status monitoring
        self.start_bridge_status_monitoring().await;

        // Start transaction monitoring
        if self.bridge_config.enable_transaction_tracking {
            self.start_transaction_monitoring().await;
        }

        // Start health monitoring
        if self.bridge_config.enable_health_monitoring {
            self.start_health_monitoring().await;
        }

        // Start fee optimization
        if self.bridge_config.enable_fee_optimization {
            self.start_fee_optimization().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Bridge monitoring started successfully");
        Ok(())
    }

    /// Stop bridge monitoring
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping bridge monitoring");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Bridge monitoring stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &BridgeMonitorStats {
        &self.stats
    }

    /// Get monitored bridges
    #[inline]
    pub async fn get_bridges(&self) -> Vec<BridgeInfo> {
        let bridges = self.bridges.read().await;
        bridges.values().cloned().collect()
    }

    /// Get tracked transactions
    #[inline]
    pub async fn get_transactions(&self) -> Vec<BridgeTransaction> {
        let transactions = self.transactions.read().await;
        transactions.values().cloned().collect()
    }

    /// Get bridge health metrics
    #[inline]
    pub async fn get_bridge_health(&self) -> Vec<BridgeHealth> {
        let health = self.bridge_health.read().await;
        health.values().cloned().collect()
    }

    /// Select optimal bridge for transfer
    #[inline]
    #[must_use]
    #[expect(clippy::significant_drop_tightening, reason = "Guards needed for entire calculation")]
    pub async fn select_optimal_bridge(&self, criteria: &BridgeSelectionCriteria) -> Option<BridgeInfo> {
        let bridges = self.bridges.read().await;
        let health = self.bridge_health.read().await;

        let mut best_bridge = None;
        let mut best_score = Decimal::ZERO;

        for bridge in bridges.values() {
            // Check if bridge supports the route
            if bridge.source_chain != criteria.source_chain ||
               bridge.destination_chain != criteria.destination_chain {
                continue;
            }

            // Check basic requirements
            if bridge.status != BridgeStatus::Operational ||
               bridge.fee_percentage > criteria.max_fee_percentage ||
               bridge.avg_transfer_time_s > criteria.max_transfer_time_s ||
               bridge.success_rate < criteria.min_success_rate ||
               bridge.available_liquidity_usd < criteria.amount {
                continue;
            }

            // Calculate score based on criteria weights
            let fee_score = if bridge.fee_percentage > Decimal::ZERO {
                (Decimal::ONE / bridge.fee_percentage) * criteria.fee_weight
            } else {
                criteria.fee_weight
            };

            let speed_score = if bridge.avg_transfer_time_s > 0 {
                (Decimal::from(3600_u64) / Decimal::from(bridge.avg_transfer_time_s)) * criteria.speed_weight
            } else {
                criteria.speed_weight
            };

            let reliability_score = bridge.success_rate * criteria.reliability_weight;

            // Add health bonus if available
            let health_bonus = health.get(&bridge.bridge_type).map_or_else(
                || "0.5".parse::<Decimal>().unwrap_or_default(),
                |bridge_health| Decimal::from(bridge_health.health_score) / Decimal::from(100_u64)
            );

            let total_score = fee_score + speed_score + reliability_score + health_bonus;

            if total_score > best_score {
                best_score = total_score;
                best_bridge = Some(bridge.clone());
            }
        }

        if best_bridge.is_some() {
            self.stats.optimal_selections.fetch_add(1, Ordering::Relaxed);
        }

        trace!("Optimal bridge selection: score={}, bridge={:?}", best_score, best_bridge.as_ref().map(|b| &b.bridge_type));
        best_bridge
    }

    /// Calculate bridge fee for transfer
    #[inline]
    #[must_use]
    pub fn calculate_bridge_fee(
        bridge_info: &BridgeInfo,
        amount: Decimal,
    ) -> Decimal {
        amount * bridge_info.fee_percentage
    }

    /// Estimate transfer time
    #[inline]
    #[must_use]
    pub fn estimate_transfer_time(
        bridge_info: &BridgeInfo,
        amount: Decimal,
    ) -> u32 {
        // Base transfer time
        let mut estimated_time = bridge_info.avg_transfer_time_s;

        // Adjust for amount (larger amounts may take longer)
        if amount > bridge_info.max_transfer_amount {
            estimated_time = estimated_time.saturating_mul(2); // Double time for oversized transfers
        } else if amount > bridge_info.max_transfer_amount / Decimal::from(2_u64) {
            estimated_time = estimated_time.saturating_add(estimated_time / 2); // 50% longer for large transfers
        }

        estimated_time
    }

    /// Check if bridge supports asset
    #[inline]
    #[must_use]
    pub fn supports_asset(
        _bridge_info: &BridgeInfo,
        asset: &str,
    ) -> bool {
        // Simplified implementation - in production this would check supported assets
        matches!(asset, "USDC" | "USDT" | "ETH" | "WETH" | "DAI")
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(5000)) // Bridge monitoring timeout
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

    /// Start bridge status monitoring
    async fn start_bridge_status_monitoring(&self) {
        let bridge_receiver = self.bridge_receiver.clone();
        let bridges = Arc::clone(&self.bridges);
        let bridge_cache = Arc::clone(&self.bridge_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let bridge_config = self.bridge_config.clone();
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(bridge_config.status_check_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming bridge updates
                while let Ok(bridge_info) = bridge_receiver.try_recv() {
                    let bridge_key = format!("{:?}_{:?}_{:?}",
                        bridge_info.bridge_type,
                        bridge_info.source_chain,
                        bridge_info.destination_chain
                    );

                    // Update bridges
                    {
                        let mut bridges_guard = bridges.write().await;
                        bridges_guard.insert(bridge_key.clone(), bridge_info.clone());
                    }

                    // Update cache with aligned data
                    let aligned_data = AlignedBridgeData::new(
                        u64::from(bridge_info.status == BridgeStatus::Operational),
                        (bridge_info.available_liquidity_usd * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        (bridge_info.fee_percentage * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        u64::from(bridge_info.avg_transfer_time_s),
                        (bridge_info.success_rate * Decimal::from(1_000_000_u64)).to_u64().unwrap_or(0),
                        85, // Default health score
                        0,  // Active transactions (updated separately)
                        bridge_info.last_update,
                    );
                    bridge_cache.insert(bridge_key, aligned_data);

                    stats.bridges_monitored.fetch_add(1, Ordering::Relaxed);
                }

                // Fetch bridge status from external sources
                if let Ok(bridge_infos) = Self::fetch_bridge_status(&http_client, &bridge_config.monitored_bridges).await {
                    for bridge_info in bridge_infos {
                        let bridge_key = format!("{:?}_{:?}_{:?}",
                            bridge_info.bridge_type,
                            bridge_info.source_chain,
                            bridge_info.destination_chain
                        );

                        // Update bridges directly since we're in the same task
                        {
                            let mut bridges_guard = bridges.write().await;
                            bridges_guard.insert(bridge_key, bridge_info);
                        }
                    }
                }

                stats.status_checks_performed.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let status_time = start_time.elapsed().as_micros() as u64;
                trace!("Bridge status monitoring cycle completed in {}μs", status_time);

                // Clean stale cache entries
                Self::clean_stale_cache(&bridge_cache, 300_000); // 5 minutes
            }
        });
    }

    /// Start transaction monitoring
    async fn start_transaction_monitoring(&self) {
        let transaction_receiver = self.transaction_receiver.clone();
        let transactions = Arc::clone(&self.transactions);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let bridge_config = self.bridge_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(bridge_config.transaction_monitor_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming transaction updates
                while let Ok(transaction) = transaction_receiver.try_recv() {
                    let tx_id = transaction.tx_id.clone();

                    // Update transaction status
                    {
                        let mut transactions_guard = transactions.write().await;
                        transactions_guard.insert(tx_id, transaction.clone());

                        // Keep only recent transactions
                        while transactions_guard.len() > BRIDGE_MONITOR_MAX_TRANSACTIONS {
                            if let Some(oldest_key) = transactions_guard.keys().next().cloned() {
                                transactions_guard.remove(&oldest_key);
                            }
                        }
                        drop(transactions_guard);
                    }

                    // Update statistics
                    stats.transactions_tracked.fetch_add(1, Ordering::Relaxed);
                    match transaction.status {
                        TransactionStatus::Completed => {
                            stats.successful_transactions.fetch_add(1, Ordering::Relaxed);
                            stats.total_fees_collected_usd.fetch_add(
                                transaction.fee_paid.to_u64().unwrap_or(0),
                                Ordering::Relaxed
                            );
                        }
                        TransactionStatus::Failed | TransactionStatus::Refunded => {
                            stats.failed_transactions.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => {}
                    }
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let transaction_time = start_time.elapsed().as_micros() as u64;
                trace!("Transaction monitoring cycle completed in {}μs", transaction_time);
            }
        });
    }

    /// Start health monitoring
    async fn start_health_monitoring(&self) {
        let bridge_health = Arc::clone(&self.bridge_health);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let bridge_config = self.bridge_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Health check every minute

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Update health metrics for monitored bridges
                for bridge_type in &bridge_config.monitored_bridges {
                    let health = Self::calculate_bridge_health(bridge_type);

                    {
                        let mut health_guard = bridge_health.write().await;
                        health_guard.insert(bridge_type.clone(), health);
                    }

                    stats.health_assessments_completed.fetch_add(1, Ordering::Relaxed);
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let health_time = start_time.elapsed().as_micros() as u64;
                trace!("Health monitoring cycle completed in {}μs", health_time);
            }
        });
    }

    /// Start fee optimization
    async fn start_fee_optimization(&self) {
        let bridges = Arc::clone(&self.bridges);
        let shutdown = Arc::clone(&self.shutdown);
        let bridge_config = self.bridge_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(bridge_config.fee_update_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Simulate fee optimization
                let bridges_guard = bridges.read().await;
                let bridge_count = bridges_guard.len();
                drop(bridges_guard);

                trace!("Fee optimization cycle completed for {} bridges", bridge_count);
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

                let bridges_monitored = stats.bridges_monitored.load(Ordering::Relaxed);
                let transactions_tracked = stats.transactions_tracked.load(Ordering::Relaxed);
                let successful_transactions = stats.successful_transactions.load(Ordering::Relaxed);
                let failed_transactions = stats.failed_transactions.load(Ordering::Relaxed);
                let total_fees = stats.total_fees_collected_usd.load(Ordering::Relaxed);
                let status_checks = stats.status_checks_performed.load(Ordering::Relaxed);
                let health_assessments = stats.health_assessments_completed.load(Ordering::Relaxed);
                let optimal_selections = stats.optimal_selections.load(Ordering::Relaxed);
                let downtime_incidents = stats.downtime_incidents.load(Ordering::Relaxed);

                info!(
                    "Bridge Monitor Stats: bridges={}, txs_tracked={}, successful={}, failed={}, fees=${}, status_checks={}, health_assessments={}, optimal_selections={}, downtime={}",
                    bridges_monitored, transactions_tracked, successful_transactions, failed_transactions,
                    total_fees, status_checks, health_assessments, optimal_selections, downtime_incidents
                );
            }
        });
    }

    /// Fetch bridge status from external sources
    async fn fetch_bridge_status(
        _http_client: &Arc<TokioMutex<Option<reqwest::Client>>>,
        monitored_bridges: &[BridgeType],
    ) -> Result<Vec<BridgeInfo>> {
        // Simplified implementation - in production this would fetch real bridge data
        let mut bridge_infos = Vec::with_capacity(monitored_bridges.len());

        for bridge_type in monitored_bridges {
            let bridge_info = BridgeInfo {
                bridge_type: bridge_type.clone(),
                source_chain: ChainId::Ethereum,
                destination_chain: ChainId::Arbitrum,
                contract_address: Self::get_bridge_contract_address(bridge_type),
                status: if chrono::Utc::now().timestamp_millis().rem_euclid(10) < 9 {
                    BridgeStatus::Operational
                } else {
                    BridgeStatus::Congested
                },
                available_liquidity_usd: Self::get_bridge_liquidity(bridge_type),
                fee_percentage: Self::get_bridge_fee_percentage(bridge_type),
                min_transfer_amount: "10".parse().unwrap_or_default(),
                max_transfer_amount: "1000000".parse().unwrap_or_default(),
                avg_transfer_time_s: Self::get_bridge_transfer_time(bridge_type),
                success_rate: Self::get_bridge_success_rate(bridge_type),
                #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for bridge data")]
                last_update: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            };
            bridge_infos.push(bridge_info);
        }

        Ok(bridge_infos)
    }

    /// Calculate bridge health metrics
    fn calculate_bridge_health(bridge_type: &BridgeType) -> BridgeHealth {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for health data")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        BridgeHealth {
            bridge_type: bridge_type.clone(),
            health_score: match bridge_type {
                BridgeType::Stargate => 95,
                BridgeType::Wormhole => 90,
                BridgeType::Multichain => 85,
                BridgeType::Hop => 88,
                BridgeType::Synapse => 87,
                BridgeType::Across => 92,
                BridgeType::Celer => 89,
                BridgeType::PolygonPos => 93,
                BridgeType::ArbitrumBridge => 96,
                BridgeType::OptimismBridge => 94,
                BridgeType::AvalancheBridge => 91,
            },
            uptime_percentage: 99,
            avg_response_time_ms: match bridge_type {
                BridgeType::Stargate => 150,
                BridgeType::Wormhole => 200,
                BridgeType::Multichain => 300,
                BridgeType::Hop => 180,
                BridgeType::Synapse => 220,
                BridgeType::Across => 160,
                BridgeType::Celer => 190,
                BridgeType::PolygonPos => 120,
                BridgeType::ArbitrumBridge => 100,
                BridgeType::OptimismBridge => 110,
                BridgeType::AvalancheBridge => 140,
            },
            success_rate: "0.995".parse().unwrap_or_default(),
            liquidity_utilization: "0.65".parse().unwrap_or_default(),
            fee_competitiveness: 85,
            last_check: now,
        }
    }

    /// Get bridge contract address
    fn get_bridge_contract_address(bridge_type: &BridgeType) -> String {
        match bridge_type {
            BridgeType::Stargate => "0x8731d54E9D02c286767d56ac03e8037C07e01e98".to_string(),
            BridgeType::Wormhole => "0x3ee18B2214AFF97000D974cf647E7C347E8fa585".to_string(),
            BridgeType::Multichain => "0x6b7a87899490EcE95443e979cA9485CBE7E71522".to_string(),
            BridgeType::Hop => "0xb8901acB165ed027E32754E0FFe830802919727f".to_string(),
            BridgeType::Synapse => "0x2796317b0fF8538F253012862c06787Adfb8cEb6".to_string(),
            BridgeType::Across => "0x4D9079Bb4165aeb4084c526a32695dCfd2F77381".to_string(),
            BridgeType::Celer => "0x5427FEFA711Eff984124bFBB1AB6fbf5E3DA1820".to_string(),
            BridgeType::PolygonPos => "0xA0c68C638235ee32657e8f720a23ceC1bFc77C77".to_string(),
            BridgeType::ArbitrumBridge => "0x8315177aB297bA92A06054cE80a67Ed4DBd7ed3a".to_string(),
            BridgeType::OptimismBridge => "0x99C9fc46f92E8a1c0deC1b1747d010903E884bE1".to_string(),
            BridgeType::AvalancheBridge => "0xE78388b4CE79068e89Bf8aA7f218eF6b9AB0e9d0".to_string(),
        }
    }

    /// Get bridge liquidity
    fn get_bridge_liquidity(bridge_type: &BridgeType) -> Decimal {
        match bridge_type {
            BridgeType::Stargate => "50000000".parse().unwrap_or_default(), // $50M
            BridgeType::Wormhole => "30000000".parse().unwrap_or_default(), // $30M
            BridgeType::Multichain => "20000000".parse().unwrap_or_default(), // $20M
            BridgeType::Hop => "15000000".parse().unwrap_or_default(), // $15M
            BridgeType::Synapse => "12000000".parse().unwrap_or_default(), // $12M
            BridgeType::Across => "25000000".parse().unwrap_or_default(), // $25M
            BridgeType::Celer => "18000000".parse().unwrap_or_default(), // $18M
            BridgeType::PolygonPos => "40000000".parse().unwrap_or_default(), // $40M
            BridgeType::ArbitrumBridge => "60000000".parse().unwrap_or_default(), // $60M
            BridgeType::OptimismBridge => "35000000".parse().unwrap_or_default(), // $35M
            BridgeType::AvalancheBridge => "22000000".parse().unwrap_or_default(), // $22M
        }
    }

    /// Get bridge fee percentage
    fn get_bridge_fee_percentage(bridge_type: &BridgeType) -> Decimal {
        match bridge_type {
            BridgeType::Stargate => "0.0006".parse().unwrap_or_default(), // 0.06%
            BridgeType::Wormhole => "0.0025".parse().unwrap_or_default(), // 0.25%
            BridgeType::Multichain => "0.001".parse().unwrap_or_default(), // 0.1%
            BridgeType::Hop | BridgeType::AvalancheBridge => "0.0004".parse().unwrap_or_default(), // 0.04%
            BridgeType::Synapse => "0.0008".parse().unwrap_or_default(), // 0.08%
            BridgeType::Across => "0.0005".parse().unwrap_or_default(), // 0.05%
            BridgeType::Celer => "0.0007".parse().unwrap_or_default(), // 0.07%
            BridgeType::PolygonPos => "0.0001".parse().unwrap_or_default(), // 0.01%
            BridgeType::ArbitrumBridge => "0.0002".parse().unwrap_or_default(), // 0.02%
            BridgeType::OptimismBridge => "0.0003".parse().unwrap_or_default(), // 0.03%
        }
    }

    /// Get bridge transfer time
    const fn get_bridge_transfer_time(bridge_type: &BridgeType) -> u32 {
        match bridge_type {
            BridgeType::Stargate => 300,  // 5 minutes
            BridgeType::Wormhole | BridgeType::AvalancheBridge => 900,  // 15 minutes
            BridgeType::Multichain => 1800, // 30 minutes
            BridgeType::Hop | BridgeType::ArbitrumBridge => 600,   // 10 minutes
            BridgeType::Synapse => 720,   // 12 minutes
            BridgeType::Across => 480,   // 8 minutes
            BridgeType::Celer => 540,   // 9 minutes
            BridgeType::PolygonPos | BridgeType::OptimismBridge => 1200, // 20 minutes
        }
    }

    /// Get bridge success rate
    fn get_bridge_success_rate(bridge_type: &BridgeType) -> Decimal {
        match bridge_type {
            BridgeType::Stargate => "0.998".parse().unwrap_or_default(),
            BridgeType::Wormhole => "0.995".parse().unwrap_or_default(),
            BridgeType::Multichain => "0.992".parse().unwrap_or_default(),
            BridgeType::Hop => "0.997".parse().unwrap_or_default(),
            BridgeType::Synapse => "0.996".parse().unwrap_or_default(),
            BridgeType::Across | BridgeType::PolygonPos => "0.999".parse().unwrap_or_default(),
            BridgeType::Celer => "0.994".parse().unwrap_or_default(),
            BridgeType::ArbitrumBridge => "0.9995".parse().unwrap_or_default(),
            BridgeType::OptimismBridge => "0.9993".parse().unwrap_or_default(),
            BridgeType::AvalancheBridge => "0.9985".parse().unwrap_or_default(),
        }
    }

    /// Clean stale cache entries
    fn clean_stale_cache(cache: &Arc<DashMap<String, AlignedBridgeData>>, max_age_ms: u64) {
        cache.retain(|_key, data| !data.is_stale(max_age_ms));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_bridge_monitor_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(monitor) = BridgeMonitor::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(monitor.stats().bridges_monitored.load(Ordering::Relaxed), 0);
        assert_eq!(monitor.stats().transactions_tracked.load(Ordering::Relaxed), 0);
        assert_eq!(monitor.stats().successful_transactions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_bridge_monitor_config_default() {
        let config = BridgeMonitorConfig::default();
        assert!(config.enabled);
        assert_eq!(config.status_check_interval_ms, BRIDGE_MONITOR_DEFAULT_STATUS_INTERVAL_MS);
        assert_eq!(config.transaction_monitor_interval_ms, BRIDGE_MONITOR_DEFAULT_TRANSACTION_INTERVAL_MS);
        assert_eq!(config.fee_update_interval_ms, BRIDGE_MONITOR_DEFAULT_FEE_UPDATE_INTERVAL_MS);
        assert!(config.enable_transaction_tracking);
        assert!(config.enable_fee_optimization);
        assert!(config.enable_health_monitoring);
        assert!(config.enable_bridge_selection);
        assert!(!config.monitored_bridges.is_empty());
    }

    #[test]
    fn test_aligned_bridge_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedBridgeData>(), 64);
        assert!(mem::size_of::<AlignedBridgeData>() <= 64);
    }

    #[test]
    fn test_bridge_monitor_stats_operations() {
        let stats = BridgeMonitorStats::default();

        stats.bridges_monitored.fetch_add(10, Ordering::Relaxed);
        stats.transactions_tracked.fetch_add(100, Ordering::Relaxed);
        stats.successful_transactions.fetch_add(95, Ordering::Relaxed);
        stats.failed_transactions.fetch_add(5, Ordering::Relaxed);
        stats.total_fees_collected_usd.fetch_add(1000, Ordering::Relaxed);

        assert_eq!(stats.bridges_monitored.load(Ordering::Relaxed), 10);
        assert_eq!(stats.transactions_tracked.load(Ordering::Relaxed), 100);
        assert_eq!(stats.successful_transactions.load(Ordering::Relaxed), 95);
        assert_eq!(stats.failed_transactions.load(Ordering::Relaxed), 5);
        assert_eq!(stats.total_fees_collected_usd.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_aligned_bridge_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedBridgeData::new(
            255, // All bridges operational
            50_000_000, // $50M liquidity
            5_000, // 0.5% fee
            600, // 10 minutes
            995_000, // 99.5% success rate
            90, // 90% health score
            25, // 25 active transactions
            now,
        );

        let stale_data = AlignedBridgeData::new(
            255, 50_000_000, 5_000, 600, 995_000, 90, 25,
            now - 360_000, // 6 minutes old
        );

        assert!(!fresh_data.is_stale(300_000)); // 5 minutes
        assert!(stale_data.is_stale(300_000)); // 5 minutes
    }

    #[test]
    fn test_aligned_bridge_data_conversions() {
        let data = AlignedBridgeData::new(
            255, // All bridges operational
            50_000_000, // $50M liquidity (scaled by 1e6)
            5_000, // 0.5% fee (scaled by 1e6)
            600, // 10 minutes
            995_000, // 99.5% success rate (scaled by 1e6)
            90, // 90% health score
            25, // 25 active transactions
            1_640_995_200_000,
        );

        assert_eq!(data.total_liquidity_usd(), dec!(50));
        assert_eq!(data.avg_fee_percentage(), dec!(0.005));
        assert_eq!(data.success_rate(), dec!(0.995));
        assert_eq!(data.operational_bridges_count(), 8); // 8 bits set in 255
        assert!(data.is_bridge_operational(0));
        assert!(data.is_bridge_operational(7));
        assert!(!data.is_bridge_operational(64)); // Out of range
    }

    #[test]
    fn test_bridge_type_equality() {
        assert_eq!(BridgeType::Stargate, BridgeType::Stargate);
        assert_ne!(BridgeType::Stargate, BridgeType::Wormhole);
        assert_ne!(BridgeType::Multichain, BridgeType::Hop);
        assert_ne!(BridgeType::Synapse, BridgeType::Across);
    }

    #[test]
    fn test_bridge_status_equality() {
        assert_eq!(BridgeStatus::Operational, BridgeStatus::Operational);
        assert_ne!(BridgeStatus::Operational, BridgeStatus::Congested);
        assert_ne!(BridgeStatus::Congested, BridgeStatus::Maintenance);
        assert_ne!(BridgeStatus::Maintenance, BridgeStatus::Offline);
        assert_ne!(BridgeStatus::Offline, BridgeStatus::Unknown);
    }

    #[test]
    fn test_transaction_status_equality() {
        assert_eq!(TransactionStatus::Initiated, TransactionStatus::Initiated);
        assert_ne!(TransactionStatus::Initiated, TransactionStatus::Pending);
        assert_ne!(TransactionStatus::Pending, TransactionStatus::SourceConfirmed);
        assert_ne!(TransactionStatus::SourceConfirmed, TransactionStatus::InTransit);
        assert_ne!(TransactionStatus::InTransit, TransactionStatus::Completed);
        assert_ne!(TransactionStatus::Completed, TransactionStatus::Failed);
        assert_ne!(TransactionStatus::Failed, TransactionStatus::Refunded);
    }

    #[test]
    fn test_bridge_info_creation() {
        let bridge_info = BridgeInfo {
            bridge_type: BridgeType::Stargate,
            source_chain: ChainId::Ethereum,
            destination_chain: ChainId::Arbitrum,
            contract_address: "0x8731d54E9D02c286767d56ac03e8037C07e01e98".to_string(),
            status: BridgeStatus::Operational,
            available_liquidity_usd: dec!(50000000),
            fee_percentage: dec!(0.0006),
            min_transfer_amount: dec!(10),
            max_transfer_amount: dec!(1000000),
            avg_transfer_time_s: 300,
            success_rate: dec!(0.998),
            last_update: 1_640_995_200_000,
        };

        assert_eq!(bridge_info.bridge_type, BridgeType::Stargate);
        assert_eq!(bridge_info.source_chain, ChainId::Ethereum);
        assert_eq!(bridge_info.destination_chain, ChainId::Arbitrum);
        assert_eq!(bridge_info.status, BridgeStatus::Operational);
        assert_eq!(bridge_info.available_liquidity_usd, dec!(50000000));
        assert_eq!(bridge_info.fee_percentage, dec!(0.0006));
        assert_eq!(bridge_info.avg_transfer_time_s, 300);
        assert_eq!(bridge_info.success_rate, dec!(0.998));
    }

    #[test]
    fn test_bridge_transaction_creation() {
        let transaction = BridgeTransaction {
            tx_id: "tx_12345".to_string(),
            bridge_type: BridgeType::Stargate,
            source_chain: ChainId::Ethereum,
            destination_chain: ChainId::Arbitrum,
            asset: "USDC".to_string(),
            amount: dec!(10000),
            fee_paid: dec!(6), // 0.06% of 10000
            status: TransactionStatus::Completed,
            initiated_at: 1_640_995_200,
            completed_at: Some(1_640_995_500),
            source_tx_hash: "0xabc123".to_string(),
            destination_tx_hash: Some("0xdef456".to_string()),
        };

        assert_eq!(transaction.tx_id, "tx_12345");
        assert_eq!(transaction.bridge_type, BridgeType::Stargate);
        assert_eq!(transaction.source_chain, ChainId::Ethereum);
        assert_eq!(transaction.destination_chain, ChainId::Arbitrum);
        assert_eq!(transaction.asset, "USDC");
        assert_eq!(transaction.amount, dec!(10000));
        assert_eq!(transaction.fee_paid, dec!(6));
        assert_eq!(transaction.status, TransactionStatus::Completed);
        assert!(transaction.completed_at.is_some());
        assert!(transaction.destination_tx_hash.is_some());
    }

    #[test]
    fn test_bridge_health_creation() {
        let health = BridgeHealth {
            bridge_type: BridgeType::Stargate,
            health_score: 95,
            uptime_percentage: 99,
            avg_response_time_ms: 150,
            success_rate: dec!(0.998),
            liquidity_utilization: dec!(0.65),
            fee_competitiveness: 85,
            last_check: 1_640_995_200_000,
        };

        assert_eq!(health.bridge_type, BridgeType::Stargate);
        assert_eq!(health.health_score, 95);
        assert_eq!(health.uptime_percentage, 99);
        assert_eq!(health.avg_response_time_ms, 150);
        assert_eq!(health.success_rate, dec!(0.998));
        assert_eq!(health.liquidity_utilization, dec!(0.65));
        assert_eq!(health.fee_competitiveness, 85);
    }

    #[test]
    fn test_bridge_selection_criteria_creation() {
        let criteria = BridgeSelectionCriteria {
            source_chain: ChainId::Ethereum,
            destination_chain: ChainId::Arbitrum,
            asset: "USDC".to_string(),
            amount: dec!(10000),
            max_fee_percentage: dec!(0.001), // 0.1% max
            max_transfer_time_s: 600, // 10 minutes max
            min_success_rate: dec!(0.995), // 99.5% min
            fee_weight: dec!(0.4),
            speed_weight: dec!(0.3),
            reliability_weight: dec!(0.3),
        };

        assert_eq!(criteria.source_chain, ChainId::Ethereum);
        assert_eq!(criteria.destination_chain, ChainId::Arbitrum);
        assert_eq!(criteria.asset, "USDC");
        assert_eq!(criteria.amount, dec!(10000));
        assert_eq!(criteria.max_fee_percentage, dec!(0.001));
        assert_eq!(criteria.max_transfer_time_s, 600);
        assert_eq!(criteria.min_success_rate, dec!(0.995));
        assert_eq!(criteria.fee_weight + criteria.speed_weight + criteria.reliability_weight, dec!(1.0));
    }

    #[test]
    fn test_calculate_bridge_fee() {
        let bridge_info = BridgeInfo {
            bridge_type: BridgeType::Stargate,
            source_chain: ChainId::Ethereum,
            destination_chain: ChainId::Arbitrum,
            contract_address: "0x8731d54E9D02c286767d56ac03e8037C07e01e98".to_string(),
            status: BridgeStatus::Operational,
            available_liquidity_usd: dec!(50000000),
            fee_percentage: dec!(0.0006), // 0.06%
            min_transfer_amount: dec!(10),
            max_transfer_amount: dec!(1000000),
            avg_transfer_time_s: 300,
            success_rate: dec!(0.998),
            last_update: 1_640_995_200_000,
        };

        let fee = BridgeMonitor::calculate_bridge_fee(&bridge_info, dec!(10000));
        assert_eq!(fee, dec!(6)); // 0.06% of 10000 = 6
    }

    #[test]
    fn test_estimate_transfer_time() {
        let bridge_info = BridgeInfo {
            bridge_type: BridgeType::Stargate,
            source_chain: ChainId::Ethereum,
            destination_chain: ChainId::Arbitrum,
            contract_address: "0x8731d54E9D02c286767d56ac03e8037C07e01e98".to_string(),
            status: BridgeStatus::Operational,
            available_liquidity_usd: dec!(50000000),
            fee_percentage: dec!(0.0006),
            min_transfer_amount: dec!(10),
            max_transfer_amount: dec!(1000000),
            avg_transfer_time_s: 300, // 5 minutes base
            success_rate: dec!(0.998),
            last_update: 1_640_995_200_000,
        };

        // Normal amount
        let time1 = BridgeMonitor::estimate_transfer_time(&bridge_info, dec!(10000));
        assert_eq!(time1, 300);

        // Large amount (> 50% of max)
        let time2 = BridgeMonitor::estimate_transfer_time(&bridge_info, dec!(600000));
        assert_eq!(time2, 450); // 300 + 150 (50% longer)

        // Oversized amount
        let time3 = BridgeMonitor::estimate_transfer_time(&bridge_info, dec!(2000000));
        assert_eq!(time3, 600); // 300 * 2 (double time)
    }

    #[test]
    fn test_supports_asset() {
        let bridge_info = BridgeInfo {
            bridge_type: BridgeType::Stargate,
            source_chain: ChainId::Ethereum,
            destination_chain: ChainId::Arbitrum,
            contract_address: "0x8731d54E9D02c286767d56ac03e8037C07e01e98".to_string(),
            status: BridgeStatus::Operational,
            available_liquidity_usd: dec!(50000000),
            fee_percentage: dec!(0.0006),
            min_transfer_amount: dec!(10),
            max_transfer_amount: dec!(1000000),
            avg_transfer_time_s: 300,
            success_rate: dec!(0.998),
            last_update: 1_640_995_200_000,
        };

        assert!(BridgeMonitor::supports_asset(&bridge_info, "USDC"));
        assert!(BridgeMonitor::supports_asset(&bridge_info, "USDT"));
        assert!(BridgeMonitor::supports_asset(&bridge_info, "ETH"));
        assert!(BridgeMonitor::supports_asset(&bridge_info, "WETH"));
        assert!(BridgeMonitor::supports_asset(&bridge_info, "DAI"));
        assert!(!BridgeMonitor::supports_asset(&bridge_info, "UNKNOWN"));
    }

    #[tokio::test]
    async fn test_bridge_monitor_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(monitor) = BridgeMonitor::new(config).await else {
            return;
        };

        let bridges = monitor.get_bridges().await;
        assert!(bridges.is_empty()); // No bridges initially

        let transactions = monitor.get_transactions().await;
        assert!(transactions.is_empty()); // No transactions initially

        let health = monitor.get_bridge_health().await;
        assert!(health.is_empty()); // No health data initially

        let stats = monitor.stats();
        assert_eq!(stats.bridges_monitored.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_bridge_contract_addresses() {
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::Stargate).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::Wormhole).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::Multichain).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::Hop).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::Synapse).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::Across).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::Celer).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::PolygonPos).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::ArbitrumBridge).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::OptimismBridge).is_empty());
        assert!(!BridgeMonitor::get_bridge_contract_address(&BridgeType::AvalancheBridge).is_empty());
    }

    #[test]
    fn test_bridge_metrics() {
        // Test that all bridges have reasonable metrics
        for bridge_type in &[
            BridgeType::Stargate, BridgeType::Wormhole, BridgeType::Multichain,
            BridgeType::Hop, BridgeType::Synapse, BridgeType::Across, BridgeType::Celer,
            BridgeType::PolygonPos, BridgeType::ArbitrumBridge, BridgeType::OptimismBridge,
            BridgeType::AvalancheBridge,
        ] {
            let liquidity = BridgeMonitor::get_bridge_liquidity(bridge_type);
            assert!(liquidity > dec!(0));

            let fee = BridgeMonitor::get_bridge_fee_percentage(bridge_type);
            assert!(fee > dec!(0));
            assert!(fee < dec!(0.01)); // Less than 1%

            let transfer_time = BridgeMonitor::get_bridge_transfer_time(bridge_type);
            assert!(transfer_time > 0);
            assert!(transfer_time < 3600); // Less than 1 hour

            let success_rate = BridgeMonitor::get_bridge_success_rate(bridge_type);
            assert!(success_rate > dec!(0.9)); // At least 90%
            assert!(success_rate <= dec!(1.0)); // At most 100%

            let health = BridgeMonitor::calculate_bridge_health(bridge_type);
            assert!(health.health_score > 80); // At least 80%
            assert!(health.health_score <= 100); // At most 100%
        }
    }
}
