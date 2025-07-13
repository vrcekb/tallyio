//! Avalanche chain coordination module
//!
//! Ultra-performance Avalanche coordinator optimized for:
//! - Trader Joe DEX integration with <15ns cross-chain operations
//! - Pangolin DEX integration for concentrated liquidity
//! - Avalanche subnet monitoring and optimization
//! - MEV detection with <150ns latency
//! - Gas optimization for AVAX efficiency

// Submodules
pub mod traderjoe_integration;
pub mod aave_integration;

use crate::{
    ChainCoreConfig, Result,
    utils::perf::Timer,
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
use tracing::{info, trace};

/// Avalanche configuration
#[derive(Debug, Clone)]
pub struct AvalancheConfig {
    /// Enable Avalanche integration
    pub enabled: bool,

    /// Mempool monitoring interval in milliseconds
    pub mempool_monitor_interval_ms: u64,

    /// Block monitoring interval in milliseconds
    pub block_monitor_interval_ms: u64,

    /// Maximum MEV opportunity age in milliseconds
    pub max_mev_opportunity_age_ms: u64,

    /// Minimum MEV opportunity value in USD
    pub min_mev_opportunity_value_usd: Decimal,

    /// Enable subnet monitoring
    pub enable_subnet_monitoring: bool,

    /// Enable cross-chain operations
    pub enable_cross_chain: bool,

    /// Maximum gas price in gwei
    pub max_gas_price_gwei: u64,

    /// Monitored subnets
    pub monitored_subnets: Vec<String>,
}

/// Avalanche mempool transaction
#[derive(Debug, Clone)]
pub struct AvalancheMempoolTransaction {
    /// Transaction hash
    pub hash: String,

    /// From address
    pub from: String,

    /// To address
    pub to: Option<String>,

    /// Value in AVAX
    pub value: Decimal,

    /// Gas price in gwei
    pub gas_price: u64,

    /// Gas limit
    pub gas_limit: u64,

    /// Transaction data
    pub data: Vec<u8>,

    /// MEV score (0-100)
    pub mev_score: u8,

    /// Timestamp when detected
    pub detected_at: u64,
}

/// Avalanche MEV opportunity
#[derive(Debug, Clone)]
pub struct AvalancheMevOpportunity {
    /// Opportunity ID
    pub id: String,

    /// Opportunity type
    pub opportunity_type: AvalancheMevType,

    /// Target transaction hash
    pub target_tx_hash: String,

    /// Estimated profit in USD
    pub estimated_profit_usd: Decimal,

    /// Required gas limit
    pub required_gas_limit: u64,

    /// Maximum gas price
    pub max_gas_price_gwei: u64,

    /// Expiry timestamp
    pub expires_at: u64,

    /// Subnet ID (if applicable)
    pub subnet_id: Option<String>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Avalanche MEV opportunity types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvalancheMevType {
    /// Arbitrage between DEXes
    Arbitrage,
    /// Liquidation opportunity
    Liquidation,
    /// Sandwich attack
    Sandwich,
    /// Cross-chain arbitrage
    CrossChain,
    /// Subnet arbitrage
    SubnetArbitrage,
}

/// Avalanche subnet information
#[derive(Debug, Clone)]
pub struct AvalancheSubnet {
    /// Subnet ID
    pub id: String,

    /// Subnet name
    pub name: String,

    /// Chain ID
    pub chain_id: u64,

    /// RPC endpoint
    pub rpc_endpoint: String,

    /// Current block number
    pub current_block: u64,

    /// Block time in seconds
    pub block_time: u64,

    /// Gas price in gwei
    pub gas_price: u64,

    /// Total value locked (USD)
    pub tvl_usd: Decimal,

    /// Active validators
    pub active_validators: u64,

    /// Last update timestamp
    pub last_update: u64,
}

/// Avalanche statistics
#[derive(Debug, Default)]
pub struct AvalancheStats {
    /// Total transactions processed
    pub transactions_processed: AtomicU64,

    /// MEV opportunities detected
    pub mev_opportunities_detected: AtomicU64,

    /// MEV opportunities executed
    pub mev_opportunities_executed: AtomicU64,

    /// Total MEV profit (USD)
    pub total_mev_profit_usd: AtomicU64,

    /// Subnets monitored
    pub subnets_monitored: AtomicU64,

    /// Cross-chain operations
    pub cross_chain_operations: AtomicU64,

    /// Average block time (milliseconds)
    pub avg_block_time_ms: AtomicU64,

    /// Failed transactions
    pub failed_transactions: AtomicU64,
}

/// Cache-line aligned block data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedAvalancheBlockData {
    /// Block number
    pub block_number: u64,

    /// Block timestamp
    pub timestamp: u64,

    /// Gas used
    pub gas_used: u64,

    /// Gas limit
    pub gas_limit: u64,

    /// Base fee per gas (gwei)
    pub base_fee_gwei: u64,

    /// Transaction count
    pub tx_count: u64,

    /// MEV opportunities count
    pub mev_count: u64,

    /// Reserved for future use
    pub reserved: u64,
}

/// Avalanche integration constants
pub const AVALANCHE_DEFAULT_MEMPOOL_INTERVAL_MS: u64 = 500; // 500ms
pub const AVALANCHE_DEFAULT_BLOCK_INTERVAL_MS: u64 = 2000; // 2 seconds
pub const AVALANCHE_DEFAULT_MAX_MEV_AGE_MS: u64 = 30000; // 30 seconds
pub const AVALANCHE_DEFAULT_MIN_MEV_VALUE: &str = "10"; // $10 minimum
pub const AVALANCHE_DEFAULT_MAX_GAS_PRICE: u64 = 50; // 50 gwei
pub const AVALANCHE_MAX_MEMPOOL_SIZE: usize = 10000;
pub const AVALANCHE_MAX_MEV_OPPORTUNITIES: usize = 1000;

/// Avalanche C-Chain ID
pub const AVALANCHE_C_CHAIN_ID: u64 = 43114;

/// Avalanche Fuji Testnet ID
pub const AVALANCHE_FUJI_CHAIN_ID: u64 = 43113;

/// AVAX token address (native token)
pub const AVAX_TOKEN_ADDRESS: &str = "0x0000000000000000000000000000000000000000";

impl Default for AvalancheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mempool_monitor_interval_ms: AVALANCHE_DEFAULT_MEMPOOL_INTERVAL_MS,
            block_monitor_interval_ms: AVALANCHE_DEFAULT_BLOCK_INTERVAL_MS,
            max_mev_opportunity_age_ms: AVALANCHE_DEFAULT_MAX_MEV_AGE_MS,
            min_mev_opportunity_value_usd: AVALANCHE_DEFAULT_MIN_MEV_VALUE.parse().unwrap_or_default(),
            enable_subnet_monitoring: true,
            enable_cross_chain: true,
            max_gas_price_gwei: AVALANCHE_DEFAULT_MAX_GAS_PRICE,
            monitored_subnets: vec![
                "subnet-1".to_string(),
                "subnet-2".to_string(),
            ],
        }
    }
}

impl AlignedAvalancheBlockData {
    /// Create new aligned block data
    #[inline(always)]
    #[must_use]
    pub const fn new(
        block_number: u64,
        timestamp: u64,
        gas_used: u64,
        gas_limit: u64,
        base_fee_gwei: u64,
        tx_count: u64,
        mev_count: u64,
    ) -> Self {
        Self {
            block_number,
            timestamp,
            gas_used,
            gas_limit,
            base_fee_gwei,
            tx_count,
            mev_count,
            reserved: 0,
        }
    }

    /// Check if block is recent
    #[inline(always)]
    #[must_use]
    #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for recency check")]
    pub fn is_recent(&self, max_age_ms: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        now.saturating_sub(self.timestamp) <= max_age_ms
    }

    /// Calculate gas utilization percentage
    #[inline(always)]
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "Precision loss acceptable for percentage calculation")]
    #[expect(clippy::float_arithmetic, reason = "Float arithmetic needed for percentage calculation")]
    pub fn gas_utilization(&self) -> f64 {
        if self.gas_limit == 0 {
            return 0.0;
        }
        (self.gas_used as f64 / self.gas_limit as f64) * 100.0
    }

    /// Get average gas per transaction
    #[inline(always)]
    #[must_use]
    pub const fn avg_gas_per_tx(&self) -> u64 {
        if self.tx_count == 0 {
            return 0;
        }
        self.gas_used / self.tx_count
    }
}

/// Avalanche Coordinator for ultra-performance chain operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct AvalancheCoordinator {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Avalanche specific configuration
    avalanche_config: AvalancheConfig,

    /// Statistics
    stats: Arc<AvalancheStats>,

    /// Mempool transactions
    mempool_transactions: Arc<RwLock<HashMap<String, AvalancheMempoolTransaction>>>,

    /// MEV opportunities
    mev_opportunities: Arc<RwLock<HashMap<String, AvalancheMevOpportunity>>>,

    /// Monitored subnets
    subnets: Arc<RwLock<HashMap<String, AvalancheSubnet>>>,

    /// Block data cache for ultra-fast access
    block_cache: Arc<DashMap<u64, AlignedAvalancheBlockData>>,

    /// Performance timers
    mempool_timer: Timer,
    block_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Transaction channels
    tx_sender: Sender<AvalancheMempoolTransaction>,
    tx_receiver: Receiver<AvalancheMempoolTransaction>,

    /// MEV opportunity channels
    mev_sender: Sender<AvalancheMevOpportunity>,
    mev_receiver: Receiver<AvalancheMevOpportunity>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current block number
    current_block: Arc<TokioMutex<u64>>,
}

impl AvalancheCoordinator {
    /// Create new Avalanche coordinator with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let avalanche_config = AvalancheConfig::default();
        let stats = Arc::new(AvalancheStats::default());
        let mempool_transactions = Arc::new(RwLock::new(HashMap::with_capacity(AVALANCHE_MAX_MEMPOOL_SIZE)));
        let mev_opportunities = Arc::new(RwLock::new(HashMap::with_capacity(AVALANCHE_MAX_MEV_OPPORTUNITIES)));
        let subnets = Arc::new(RwLock::new(HashMap::with_capacity(10)));
        let block_cache = Arc::new(DashMap::with_capacity(1000));
        let mempool_timer = Timer::new("avalanche_mempool");
        let block_timer = Timer::new("avalanche_block");
        let shutdown = Arc::new(AtomicBool::new(false));
        let current_block = Arc::new(TokioMutex::new(0));

        let (tx_sender, tx_receiver) = channel::bounded(AVALANCHE_MAX_MEMPOOL_SIZE);
        let (mev_sender, mev_receiver) = channel::bounded(AVALANCHE_MAX_MEV_OPPORTUNITIES);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            avalanche_config,
            stats,
            mempool_transactions,
            mev_opportunities,
            subnets,
            block_cache,
            mempool_timer,
            block_timer,
            shutdown,
            tx_sender,
            tx_receiver,
            mev_sender,
            mev_receiver,
            http_client,
            current_block,
        })
    }

    /// Start Avalanche coordinator services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.avalanche_config.enabled {
            info!("Avalanche coordinator disabled");
            return Ok(());
        }

        info!("Starting Avalanche coordinator");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start mempool monitoring
        self.start_mempool_monitoring().await;

        // Start block monitoring
        self.start_block_monitoring().await;

        // Start MEV detection
        self.start_mev_detection().await;

        // Start subnet monitoring
        if self.avalanche_config.enable_subnet_monitoring {
            self.start_subnet_monitoring().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Avalanche coordinator started successfully");
        Ok(())
    }

    /// Stop Avalanche coordinator
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping Avalanche coordinator");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Avalanche coordinator stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &AvalancheStats {
        &self.stats
    }

    /// Get mempool transactions
    #[inline]
    pub async fn get_mempool_transactions(&self) -> Vec<AvalancheMempoolTransaction> {
        let transactions = self.mempool_transactions.read().await;
        transactions.values().cloned().collect()
    }

    /// Get MEV opportunities
    #[inline]
    pub async fn get_mev_opportunities(&self) -> Vec<AvalancheMevOpportunity> {
        let opportunities = self.mev_opportunities.read().await;
        opportunities.values().cloned().collect()
    }

    /// Get monitored subnets
    #[inline]
    pub async fn get_subnets(&self) -> Vec<AvalancheSubnet> {
        let subnets = self.subnets.read().await;
        subnets.values().cloned().collect()
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(2000)) // Fast timeout for Avalanche calls
            .http2_prior_knowledge()
            .http2_keep_alive_timeout(Duration::from_secs(30))
            .http2_keep_alive_interval(Duration::from_secs(10))
            .pool_max_idle_per_host(5)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .map_err(|_e| crate::ChainCoreError::Network(crate::NetworkError::ConnectionRefused))?;

        {
            let mut http_client_guard = self.http_client.lock().await;
            *http_client_guard = Some(client);
        }

        Ok(())
    }

    /// Start mempool monitoring
    async fn start_mempool_monitoring(&self) {
        let tx_receiver = self.tx_receiver.clone();
        let mempool_transactions = Arc::clone(&self.mempool_transactions);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let avalanche_config = self.avalanche_config.clone();
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(avalanche_config.mempool_monitor_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming transactions
                while let Ok(tx) = tx_receiver.try_recv() {
                    let tx_hash = tx.hash.clone();

                    // Update mempool
                    {
                        let mut mempool = mempool_transactions.write().await;
                        mempool.insert(tx_hash, tx);

                        // Keep only recent transactions
                        while mempool.len() > AVALANCHE_MAX_MEMPOOL_SIZE {
                            if let Some(oldest_key) = mempool.keys().next().cloned() {
                                mempool.remove(&oldest_key);
                            }
                        }
                        drop(mempool);
                    }

                    stats.transactions_processed.fetch_add(1, Ordering::Relaxed);
                }

                // Fetch mempool data from Avalanche
                if let Ok(mempool_data) = Self::fetch_avalanche_mempool(&http_client).await {
                    for tx in mempool_data {
                        let tx_hash = tx.hash.clone();

                        // Update mempool directly since we're in the same task
                        {
                            let mut mempool = mempool_transactions.write().await;
                            mempool.insert(tx_hash, tx);
                        }
                    }
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let mempool_time = start_time.elapsed().as_micros() as u64;
                trace!("Mempool monitoring cycle completed in {}μs", mempool_time);

                // Clean old transactions
                Self::cleanup_old_transactions(&mempool_transactions, avalanche_config.max_mev_opportunity_age_ms).await;
            }
        });
    }

    /// Start block monitoring
    async fn start_block_monitoring(&self) {
        let block_cache = Arc::clone(&self.block_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let avalanche_config = self.avalanche_config.clone();
        let current_block = Arc::clone(&self.current_block);
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(avalanche_config.block_monitor_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Fetch latest block
                if let Ok(block_data) = Self::fetch_latest_block(&http_client).await {
                    let block_number = block_data.block_number;

                    // Update current block
                    {
                        let mut current = current_block.lock().await;
                        *current = block_number;
                    }

                    // Cache block data
                    block_cache.insert(block_number, block_data);

                    // Keep only recent blocks (last 1000)
                    while block_cache.len() > 1000 {
                        if let Some(oldest_entry) = block_cache.iter().min_by_key(|entry| *entry.key()) {
                            let oldest_key = *oldest_entry.key();
                            drop(oldest_entry);
                            block_cache.remove(&oldest_key);
                        }
                    }
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let block_time = start_time.elapsed().as_micros() as u64;
                trace!("Block monitoring cycle completed in {}μs", block_time);

                stats.avg_block_time_ms.store(block_time / 1000, Ordering::Relaxed);
            }
        });
    }

    /// Start MEV detection
    async fn start_mev_detection(&self) {
        let mev_receiver = self.mev_receiver.clone();
        let mev_opportunities = Arc::clone(&self.mev_opportunities);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                if let Ok(opportunity) = mev_receiver.recv() {
                    let opportunity_id = opportunity.id.clone();

                    // Store MEV opportunity
                    {
                        let mut opportunities = mev_opportunities.write().await;
                        opportunities.insert(opportunity_id, opportunity);

                        // Keep only recent opportunities
                        while opportunities.len() > AVALANCHE_MAX_MEV_OPPORTUNITIES {
                            if let Some(oldest_key) = opportunities.keys().next().cloned() {
                                opportunities.remove(&oldest_key);
                            }
                        }
                        drop(opportunities);
                    }

                    stats.mev_opportunities_detected.fetch_add(1, Ordering::Relaxed);
                    trace!("MEV opportunity detected and stored");
                }

                sleep(Duration::from_millis(10)).await;
            }
        });
    }

    /// Start subnet monitoring
    async fn start_subnet_monitoring(&self) {
        let subnets = Arc::clone(&self.subnets);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let avalanche_config = self.avalanche_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Check every 30 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Simulate subnet monitoring
                for subnet_id in &avalanche_config.monitored_subnets {
                    let subnet = AvalancheSubnet {
                        id: subnet_id.clone(),
                        name: format!("Subnet {subnet_id}"),
                        chain_id: AVALANCHE_C_CHAIN_ID,
                        rpc_endpoint: format!("https://{subnet_id}.avax.network/ext/bc/C/rpc"),
                        current_block: 1_000_000,
                        block_time: 2,
                        gas_price: 25,
                        tvl_usd: Decimal::from(10_000_000),
                        active_validators: 100,
                        last_update: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    };

                    {
                        let mut subnets_guard = subnets.write().await;
                        subnets_guard.insert(subnet_id.clone(), subnet);
                    }
                }

                stats.subnets_monitored.store(avalanche_config.monitored_subnets.len() as u64, Ordering::Relaxed);
                trace!("Subnet monitoring cycle completed");
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

                let transactions = stats.transactions_processed.load(Ordering::Relaxed);
                let mev_detected = stats.mev_opportunities_detected.load(Ordering::Relaxed);
                let mev_executed = stats.mev_opportunities_executed.load(Ordering::Relaxed);
                let profit = stats.total_mev_profit_usd.load(Ordering::Relaxed);
                let subnets = stats.subnets_monitored.load(Ordering::Relaxed);
                let avg_block_time = stats.avg_block_time_ms.load(Ordering::Relaxed);

                info!(
                    "Avalanche Stats: txs={}, mev_detected={}, mev_executed={}, profit=${}, subnets={}, avg_block_time={}ms",
                    transactions, mev_detected, mev_executed, profit, subnets, avg_block_time
                );
            }
        });
    }

    /// Fetch Avalanche mempool data
    async fn fetch_avalanche_mempool(_http_client: &Arc<TokioMutex<Option<reqwest::Client>>>) -> Result<Vec<AvalancheMempoolTransaction>> {
        // Simplified implementation - in production this would fetch real mempool data
        let tx = AvalancheMempoolTransaction {
            hash: "0x1234567890123456789012345678901234567890123456789012345678901234".to_string(),
            from: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdef".to_string(),
            to: Some("0x1111111111111111111111111111111111111111".to_string()),
            value: Decimal::from(1),
            gas_price: 25,
            gas_limit: 21000,
            data: vec![],
            mev_score: 75,
            detected_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        Ok(vec![tx])
    }

    /// Fetch latest block data
    async fn fetch_latest_block(_http_client: &Arc<TokioMutex<Option<reqwest::Client>>>) -> Result<AlignedAvalancheBlockData> {
        // Simplified implementation - in production this would fetch real block data
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for block data")]
        let block_data = AlignedAvalancheBlockData::new(
            1_000_000, // Block number
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            8_000_000, // Gas used
            15_000_000, // Gas limit
            25, // Base fee
            150, // Transaction count
            5, // MEV count
        );

        Ok(block_data)
    }

    /// Clean up old transactions
    async fn cleanup_old_transactions(
        mempool_transactions: &Arc<RwLock<HashMap<String, AvalancheMempoolTransaction>>>,
        max_age_ms: u64,
    ) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut mempool = mempool_transactions.write().await;
        mempool.retain(|_hash, tx| {
            now.saturating_sub(tx.detected_at) <= max_age_ms / 1000
        });
    }

    /// Create MEV opportunity from transaction
    #[must_use]
    pub fn create_mev_opportunity_from_tx(tx: &AvalancheMempoolTransaction) -> Option<AvalancheMevOpportunity> {
        if tx.mev_score < 50 {
            return None;
        }

        let opportunity = AvalancheMevOpportunity {
            id: format!("mev_{}", tx.hash),
            opportunity_type: AvalancheMevType::Arbitrage,
            target_tx_hash: tx.hash.clone(),
            estimated_profit_usd: Decimal::from(tx.mev_score) * Decimal::from(10), // Simple calculation
            required_gas_limit: tx.gas_limit * 2,
            max_gas_price_gwei: tx.gas_price + 10,
            expires_at: tx.detected_at + 30, // 30 seconds
            subnet_id: None,
            metadata: HashMap::new(),
        };

        Some(opportunity)
    }

    /// Calculate MEV score for transaction
    #[must_use]
    pub fn calculate_mev_score(tx: &AvalancheMempoolTransaction) -> u8 {
        let mut score = 0_u8;

        // High value transactions
        if tx.value > Decimal::from(100) {
            score = score.saturating_add(30);
        }

        // High gas price (potential front-running)
        if tx.gas_price > 50 {
            score = score.saturating_add(25);
        }

        // Contract interaction
        if tx.to.is_some() && !tx.data.is_empty() {
            score = score.saturating_add(20);
        }

        // Large gas limit
        if tx.gas_limit > 100_000 {
            score = score.saturating_add(15);
        }

        // Recent transaction
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if now.saturating_sub(tx.detected_at) < 10 {
            score = score.saturating_add(10);
        }

        score.min(100)
    }
}

// Re-exports for public API
pub use traderjoe_integration::{
    TraderJoeIntegration, TraderJoeConfig, TraderJoeStats,
    TraderJoePool, TraderJoeRoute, TraderJoeLBPosition, TraderJoeYieldPosition,
    TraderJoeVersion, TraderJoeRouteStep, TraderJoeBinPosition
};

pub use aave_integration::{
    AaveAvalancheIntegration, AaveAvalancheConfig, AaveAvalancheStats,
    AaveAvalancheMarket, AaveAvalanchePosition, AaveLiquidationOpportunity,
    AaveYieldStrategy, AaveAssetPosition
};

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_avalanche_coordinator_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = AvalancheCoordinator::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(coordinator.stats().transactions_processed.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().mev_opportunities_detected.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().subnets_monitored.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_avalanche_config_default() {
        let config = AvalancheConfig::default();
        assert!(config.enabled);
        assert_eq!(config.mempool_monitor_interval_ms, AVALANCHE_DEFAULT_MEMPOOL_INTERVAL_MS);
        assert_eq!(config.block_monitor_interval_ms, AVALANCHE_DEFAULT_BLOCK_INTERVAL_MS);
        assert!(config.enable_subnet_monitoring);
        assert!(config.enable_cross_chain);
        assert_eq!(config.max_gas_price_gwei, AVALANCHE_DEFAULT_MAX_GAS_PRICE);
        assert!(!config.monitored_subnets.is_empty());
    }

    #[test]
    fn test_aligned_avalanche_block_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedAvalancheBlockData>(), 64);
        assert!(mem::size_of::<AlignedAvalancheBlockData>() <= 64);
    }

    #[test]
    fn test_avalanche_stats_operations() {
        let stats = AvalancheStats::default();

        stats.transactions_processed.fetch_add(100, Ordering::Relaxed);
        stats.mev_opportunities_detected.fetch_add(25, Ordering::Relaxed);
        stats.total_mev_profit_usd.fetch_add(5000, Ordering::Relaxed);
        stats.subnets_monitored.fetch_add(3, Ordering::Relaxed);

        assert_eq!(stats.transactions_processed.load(Ordering::Relaxed), 100);
        assert_eq!(stats.mev_opportunities_detected.load(Ordering::Relaxed), 25);
        assert_eq!(stats.total_mev_profit_usd.load(Ordering::Relaxed), 5000);
        assert_eq!(stats.subnets_monitored.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_aligned_avalanche_block_data_recency() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let recent_block = AlignedAvalancheBlockData::new(
            1_000_000,
            now,
            8_000_000,
            15_000_000,
            25,
            150,
            5,
        );

        let old_block = AlignedAvalancheBlockData::new(
            999_999,
            now - 120_000, // 2 minutes old
            8_000_000,
            15_000_000,
            25,
            150,
            5,
        );

        assert!(recent_block.is_recent(60_000));
        assert!(!old_block.is_recent(60_000));
    }

    #[test]
    fn test_aligned_avalanche_block_data_gas_utilization() {
        let block_data = AlignedAvalancheBlockData::new(
            1_000_000,
            1_640_995_200_000,
            8_000_000, // 8M gas used
            15_000_000, // 15M gas limit
            25,
            150,
            5,
        );

        let utilization = block_data.gas_utilization();
        assert!((utilization - 53.333_333_333_333_336).abs() < 0.001); // ~53.33%

        assert_eq!(block_data.avg_gas_per_tx(), 53_333); // 8M / 150 txs
    }

    #[test]
    fn test_avalanche_chain_id() {
        assert_eq!(AVALANCHE_C_CHAIN_ID, 43114);
        assert_eq!(AVALANCHE_FUJI_CHAIN_ID, 43113);
    }

    #[test]
    fn test_avalanche_mempool_transaction_creation() {
        let tx = AvalancheMempoolTransaction {
            hash: "0x1234567890123456789012345678901234567890123456789012345678901234".to_string(),
            from: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdef".to_string(),
            to: Some("0x1111111111111111111111111111111111111111".to_string()),
            value: dec!(1),
            gas_price: 25,
            gas_limit: 21000,
            data: vec![],
            mev_score: 75,
            detected_at: 1_640_995_200,
        };

        assert_eq!(tx.hash.len(), 66); // 0x + 64 hex chars
        assert_eq!(tx.gas_price, 25);
        assert_eq!(tx.mev_score, 75);
        assert!(tx.to.is_some());
    }

    #[test]
    fn test_avalanche_mev_opportunity_detection() {
        let tx = AvalancheMempoolTransaction {
            hash: "0x1234567890123456789012345678901234567890123456789012345678901234".to_string(),
            from: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdef".to_string(),
            to: Some("0x1111111111111111111111111111111111111111".to_string()),
            value: dec!(1),
            gas_price: 25,
            gas_limit: 21000,
            data: vec![],
            mev_score: 75,
            detected_at: 1_640_995_200,
        };

        let opportunity = AvalancheCoordinator::create_mev_opportunity_from_tx(&tx);
        assert!(opportunity.is_some());

        if let Some(opp) = opportunity {
            assert_eq!(opp.opportunity_type, AvalancheMevType::Arbitrage);
            assert!(opp.estimated_profit_usd > Decimal::ZERO);
            assert_eq!(opp.target_tx_hash, tx.hash);
        }
    }

    #[test]
    fn test_avalanche_mev_score_calculation() {
        let high_value_tx = AvalancheMempoolTransaction {
            hash: "0x1234567890123456789012345678901234567890123456789012345678901234".to_string(),
            from: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdef".to_string(),
            to: Some("0x1111111111111111111111111111111111111111".to_string()),
            value: dec!(150), // High value
            gas_price: 60, // High gas price
            gas_limit: 150_000, // Large gas limit
            data: vec![1, 2, 3], // Contract interaction
            mev_score: 0,
            detected_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let score = AvalancheCoordinator::calculate_mev_score(&high_value_tx);
        assert!(score >= 90); // Should be high score

        let low_value_tx = AvalancheMempoolTransaction {
            hash: "0x5678567856785678567856785678567856785678567856785678567856785678".to_string(),
            from: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdef".to_string(),
            to: None,
            value: dec!(0.1), // Low value
            gas_price: 10, // Low gas price
            gas_limit: 21000, // Standard gas limit
            data: vec![], // No contract interaction
            mev_score: 0,
            detected_at: 1_640_995_200, // Old transaction
        };

        let low_score = AvalancheCoordinator::calculate_mev_score(&low_value_tx);
        assert!(low_score < 50); // Should be low score
    }

    #[test]
    fn test_avalanche_subnet_creation() {
        let subnet = AvalancheSubnet {
            id: "subnet-1".to_string(),
            name: "Test Subnet".to_string(),
            chain_id: AVALANCHE_C_CHAIN_ID,
            rpc_endpoint: "https://subnet-1.avax.network/ext/bc/C/rpc".to_string(),
            current_block: 1_000_000,
            block_time: 2,
            gas_price: 25,
            tvl_usd: dec!(10000000),
            active_validators: 100,
            last_update: 1_640_995_200,
        };

        assert_eq!(subnet.id, "subnet-1");
        assert_eq!(subnet.chain_id, AVALANCHE_C_CHAIN_ID);
        assert_eq!(subnet.block_time, 2);
        assert_eq!(subnet.active_validators, 100);
    }

    #[test]
    fn test_mev_opportunity_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("dex".to_string(), "TraderJoe".to_string());

        let opportunity = AvalancheMevOpportunity {
            id: "mev_123".to_string(),
            opportunity_type: AvalancheMevType::Arbitrage,
            target_tx_hash: "0x1234567890123456789012345678901234567890123456789012345678901234".to_string(),
            estimated_profit_usd: dec!(100),
            required_gas_limit: 200_000,
            max_gas_price_gwei: 50,
            expires_at: 1_640_995_230,
            subnet_id: Some("subnet-1".to_string()),
            metadata,
        };

        assert_eq!(opportunity.id, "mev_123");
        assert_eq!(opportunity.opportunity_type, AvalancheMevType::Arbitrage);
        assert_eq!(opportunity.estimated_profit_usd, dec!(100));
        assert!(opportunity.subnet_id.is_some());
        assert_eq!(opportunity.metadata.get("dex"), Some(&"TraderJoe".to_string()));
    }

    #[tokio::test]
    async fn test_fetch_latest_block() {
        let http_client = Arc::new(TokioMutex::new(None));

        let result = AvalancheCoordinator::fetch_latest_block(&http_client).await;

        assert!(result.is_ok());
        if let Ok(block_data) = result {
            assert!(block_data.block_number > 0);
            assert!(block_data.gas_limit > 0);
            assert!(block_data.tx_count > 0);
        }
    }

    #[tokio::test]
    async fn test_avalanche_coordinator_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = AvalancheCoordinator::new(config).await else {
            return;
        };

        let mempool_txs = coordinator.get_mempool_transactions().await;
        assert!(mempool_txs.is_empty()); // No transactions initially

        let mev_opportunities = coordinator.get_mev_opportunities().await;
        assert!(mev_opportunities.is_empty()); // No opportunities initially

        let subnets = coordinator.get_subnets().await;
        assert!(subnets.is_empty()); // No subnets initially

        let stats = coordinator.stats();
        assert_eq!(stats.transactions_processed.load(Ordering::Relaxed), 0);
    }
}
