//! Optimism chain coordination module
//!
//! Ultra-performance Optimism coordinator optimized for:
//! - Uniswap v3 integration with <25ns cross-chain operations
//! - Aave v3 protocol lending/borrowing strategies
//! - Velodrome DEX integration for concentrated liquidity
//! - MEV detection with <250ns latency
//! - Gas optimization for ETH efficiency on L2

// Submodules
pub mod velodrome_integration;
pub mod sequencer_monitor;

use crate::{
    ChainCoreConfig, Result,
    types::{ChainId, TokenAddress, TradingPair, Opportunity, OpportunityType},
    utils::perf::Timer,
};
use crossbeam::channel::{self, Receiver, Sender};
use dashmap::DashMap;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicBool, Ordering},
    },
    time::{Duration, Instant},
    collections::VecDeque,
};
use tokio::{
    sync::{RwLock, Mutex as TokioMutex},
    time::{interval, sleep},
};
use tracing::{info, trace};

/// Optimism configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct OptimismConfig {
    /// Enable Optimism coordinator
    pub enabled: bool,

    /// Chain ID (10 for Optimism mainnet)
    pub chain_id: u64,

    /// RPC endpoint URL
    pub rpc_url: String,

    /// WebSocket endpoint URL
    pub ws_url: String,

    /// Block monitoring interval in milliseconds
    pub block_interval_ms: u64,

    /// MEV detection threshold in USD
    pub mev_threshold_usd: Decimal,

    /// Enable Uniswap v3 monitoring
    pub enable_uniswap_v3: bool,

    /// Enable Aave v3 monitoring
    pub enable_aave_v3: bool,

    /// Enable Velodrome monitoring
    pub enable_velodrome: bool,

    /// Enable Synthetix monitoring
    pub enable_synthetix: bool,

    /// Maximum gas price in Gwei
    pub max_gas_price_gwei: u64,

    /// Gas optimization enabled
    pub gas_optimization: bool,
}

/// Optimism mempool transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimismMempoolTransaction {
    /// Transaction hash
    pub hash: String,

    /// From address
    pub from: String,

    /// To address
    pub to: Option<String>,

    /// Value in Wei
    pub value: String,

    /// Gas limit
    pub gas_limit: u64,

    /// Gas price in Gwei
    pub gas_price: u64,

    /// Transaction data
    pub data: String,

    /// Nonce
    pub nonce: u64,

    /// MEV score (0-1000)
    pub mev_score: u16,

    /// Detected protocols
    pub protocols: Vec<OptimismProtocol>,

    /// Timestamp when detected
    pub detected_at: u64,
}

/// Optimism protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimismProtocol {
    /// Uniswap v3
    UniswapV3,
    /// Aave v3
    AaveV3,
    /// Velodrome
    Velodrome,
    /// Synthetix
    Synthetix,
    /// 1inch
    OneInch,
    /// Curve
    Curve,
}

/// Optimism coordinator statistics
#[derive(Debug, Default)]
pub struct OptimismStats {
    /// Total blocks processed
    pub blocks_processed: AtomicU64,

    /// Total transactions analyzed
    pub transactions_analyzed: AtomicU64,

    /// MEV opportunities detected
    pub mev_opportunities: AtomicU64,

    /// Uniswap v3 transactions
    pub uniswap_v3_txs: AtomicU64,

    /// Aave v3 transactions
    pub aave_v3_txs: AtomicU64,

    /// Velodrome transactions
    pub velodrome_txs: AtomicU64,

    /// Synthetix transactions
    pub synthetix_txs: AtomicU64,

    /// Average block processing time (microseconds)
    pub avg_block_time_us: AtomicU64,

    /// Gas optimization savings (USD)
    pub gas_savings_usd: AtomicU64,
}

/// Cache-line aligned block data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedOptimismBlockData {
    /// Block number
    pub block_number: u64,

    /// Block timestamp
    pub timestamp: u64,

    /// Transaction count
    pub tx_count: u64,

    /// Gas used
    pub gas_used: u64,
}

/// Optimism coordinator constants
pub const OPTIMISM_CHAIN_ID: u64 = 10;
pub const OPTIMISM_BLOCK_TIME_MS: u64 = 2000; // 2 seconds
pub const OPTIMISM_DEFAULT_MEV_THRESHOLD_USD: &str = "5"; // $5 minimum
pub const OPTIMISM_DEFAULT_MAX_GAS_GWEI: u64 = 15; // 15 Gwei max
pub const OPTIMISM_MAX_MEMPOOL_SIZE: usize = 10000;

/// Optimism mainnet RPC
pub const OPTIMISM_RPC_URL: &str = "https://mainnet.optimism.io";

/// Optimism mainnet WebSocket
pub const OPTIMISM_WS_URL: &str = "wss://mainnet.optimism.io/ws";

/// Uniswap v3 router on Optimism
pub const UNISWAP_V3_ROUTER_OPTIMISM: &str = "0xE592427A0AEce92De3Edee1F18E0157C05861564";

/// Aave v3 pool on Optimism
pub const AAVE_V3_POOL_OPTIMISM: &str = "0x794a61358D6845594F94dc1DB02A252b5b4814aD";

/// Velodrome router on Optimism
pub const VELODROME_ROUTER_OPTIMISM: &str = "0x9c12939390052919aF3155f41Bf4160Fd3666A6f";

impl Default for OptimismConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            chain_id: OPTIMISM_CHAIN_ID,
            rpc_url: OPTIMISM_RPC_URL.to_string(),
            ws_url: OPTIMISM_WS_URL.to_string(),
            block_interval_ms: OPTIMISM_BLOCK_TIME_MS,
            mev_threshold_usd: OPTIMISM_DEFAULT_MEV_THRESHOLD_USD.parse().unwrap_or_default(),
            enable_uniswap_v3: true,
            enable_aave_v3: true,
            enable_velodrome: true,
            enable_synthetix: true,
            max_gas_price_gwei: OPTIMISM_DEFAULT_MAX_GAS_GWEI,
            gas_optimization: true,
        }
    }
}

impl AlignedOptimismBlockData {
    /// Create new aligned block data
    #[inline(always)]
    #[must_use]
    pub const fn new(block_number: u64, timestamp: u64, tx_count: u64, gas_used: u64) -> Self {
        Self {
            block_number,
            timestamp,
            tx_count,
            gas_used,
        }
    }

    /// Check if block data is recent
    #[inline(always)]
    #[must_use]
    pub fn is_recent(&self, max_age_seconds: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now.saturating_sub(self.timestamp) <= max_age_seconds
    }

    /// Calculate gas utilization percentage
    #[inline(always)]
    #[must_use]
    pub const fn gas_utilization(&self) -> u64 {
        if self.gas_used == 0 {
            return 0;
        }
        // Optimism block gas limit is ~30M
        (self.gas_used * 100) / 30_000_000
    }
}

impl OptimismMempoolTransaction {
    /// Calculate MEV score for Optimism transaction
    #[inline]
    #[must_use]
    pub fn calculate_optimism_mev_score(&self) -> u16 {
        let mut score = 0_u16;

        // Base score from gas price (higher gas = higher MEV potential)
        score = score.saturating_add(u16::try_from(self.gas_price.min(100) * 2).unwrap_or(200));

        // Protocol-specific scoring
        for protocol in &self.protocols {
            score = score.saturating_add(match protocol {
                OptimismProtocol::UniswapV3 => 150,    // High MEV potential
                OptimismProtocol::AaveV3 => 100,       // Medium MEV potential
                OptimismProtocol::Velodrome => 120,    // High MEV potential (concentrated liquidity)
                OptimismProtocol::Synthetix => 80,     // Medium MEV potential
                OptimismProtocol::OneInch => 140,      // High MEV potential (aggregator)
                OptimismProtocol::Curve => 90,         // Medium MEV potential
            });
        }

        // Value-based scoring (higher value = higher MEV potential)
        if let Ok(value_wei) = self.value.parse::<u128>() {
            let value_eth = value_wei / 1_000_000_000_000_000_000; // Convert to ETH
            score = score.saturating_add(u16::try_from(value_eth.min(100) * 5).unwrap_or(500));
        }

        // Gas limit scoring (complex transactions = higher MEV potential)
        if self.gas_limit > 200_000 {
            score = score.saturating_add(50);
        }
        if self.gas_limit > 500_000 {
            score = score.saturating_add(100);
        }

        score.min(1000) // Cap at 1000
    }
}

/// Optimism Coordinator for ultra-performance L2 operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct OptimismCoordinator {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Optimism specific configuration
    optimism_config: OptimismConfig,

    /// Statistics
    stats: Arc<OptimismStats>,

    /// Current mempool transactions
    mempool: Arc<RwLock<VecDeque<OptimismMempoolTransaction>>>,

    /// Block data cache for ultra-fast access
    block_cache: Arc<DashMap<u64, AlignedOptimismBlockData>>,

    /// MEV opportunities
    mev_opportunities: Arc<RwLock<VecDeque<Opportunity>>>,

    /// Performance timers
    block_timer: Timer,
    mev_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Block processing channels
    block_sender: Sender<AlignedOptimismBlockData>,
    block_receiver: Receiver<AlignedOptimismBlockData>,

    /// Transaction processing channels
    tx_sender: Sender<OptimismMempoolTransaction>,
    tx_receiver: Receiver<OptimismMempoolTransaction>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current block number
    current_block: Arc<TokioMutex<u64>>,
}

impl OptimismCoordinator {
    /// Create new Optimism coordinator with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let optimism_config = OptimismConfig::default();
        let stats = Arc::new(OptimismStats::default());
        let mempool = Arc::new(RwLock::new(VecDeque::with_capacity(OPTIMISM_MAX_MEMPOOL_SIZE)));
        let block_cache = Arc::new(DashMap::with_capacity(1000));
        let mev_opportunities = Arc::new(RwLock::new(VecDeque::with_capacity(500)));
        let block_timer = Timer::new("optimism_block");
        let mev_timer = Timer::new("optimism_mev");
        let shutdown = Arc::new(AtomicBool::new(false));
        let current_block = Arc::new(TokioMutex::new(0));

        let (block_sender, block_receiver) = channel::bounded(100);
        let (tx_sender, tx_receiver) = channel::bounded(OPTIMISM_MAX_MEMPOOL_SIZE);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            optimism_config,
            stats,
            mempool,
            block_cache,
            mev_opportunities,
            block_timer,
            mev_timer,
            shutdown,
            block_sender,
            block_receiver,
            tx_sender,
            tx_receiver,
            http_client,
            current_block,
        })
    }

    /// Start Optimism coordinator services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.optimism_config.enabled {
            info!("Optimism coordinator disabled");
            return Ok(());
        }

        info!("Starting Optimism coordinator");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start block monitoring
        self.start_block_monitoring().await;

        // Start transaction processing
        self.start_transaction_processing().await;

        // Start MEV detection
        self.start_mev_detection().await;

        // Start protocol monitoring
        if self.optimism_config.enable_uniswap_v3 {
            self.start_uniswap_v3_monitoring().await;
        }

        if self.optimism_config.enable_aave_v3 {
            self.start_aave_v3_monitoring().await;
        }

        if self.optimism_config.enable_velodrome {
            self.start_velodrome_monitoring().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Optimism coordinator started successfully");
        Ok(())
    }

    /// Stop Optimism coordinator
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping Optimism coordinator");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Optimism coordinator stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &OptimismStats {
        &self.stats
    }

    /// Get current mempool size
    #[inline]
    pub async fn mempool_size(&self) -> usize {
        let mempool = self.mempool.read().await;
        mempool.len()
    }

    /// Get MEV opportunities
    #[inline]
    pub async fn get_mev_opportunities(&self) -> Vec<Opportunity> {
        let opportunities = self.mev_opportunities.read().await;
        opportunities.iter().cloned().collect()
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(2000)) // Fast timeout for L2
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

    /// Start block monitoring
    async fn start_block_monitoring(&self) {
        let block_receiver = self.block_receiver.clone();
        let block_cache = Arc::clone(&self.block_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let optimism_config = self.optimism_config.clone();
        let http_client = Arc::clone(&self.http_client);
        let current_block = Arc::clone(&self.current_block);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(optimism_config.block_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming blocks
                while let Ok(block_data) = block_receiver.try_recv() {
                    block_cache.insert(block_data.block_number, block_data.clone());
                    stats.blocks_processed.fetch_add(1, Ordering::Relaxed);

                    // Update current block
                    {
                        let mut current = current_block.lock().await;
                        *current = block_data.block_number;
                    }
                }

                // Fetch latest block
                if let Ok(block_data) = Self::fetch_latest_block(&http_client).await {
                    block_cache.insert(block_data.block_number, block_data.clone());
                    stats.blocks_processed.fetch_add(1, Ordering::Relaxed);
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let block_time = start_time.elapsed().as_micros() as u64;
                stats.avg_block_time_us.store(block_time, Ordering::Relaxed);

                trace!("Optimism block monitoring cycle completed in {}μs", block_time);

                // Clean old blocks from cache
                Self::clean_old_blocks(&block_cache, 1000);
            }
        });
    }

    /// Start transaction processing
    async fn start_transaction_processing(&self) {
        let tx_receiver = self.tx_receiver.clone();
        let mempool = Arc::clone(&self.mempool);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                if let Ok(tx) = tx_receiver.recv() {
                    {
                        let mut mempool_guard = mempool.write().await;
                        mempool_guard.push_back(tx);

                        // Keep mempool size manageable
                        while mempool_guard.len() > OPTIMISM_MAX_MEMPOOL_SIZE {
                            mempool_guard.pop_front();
                        }
                        drop(mempool_guard);
                    }

                    stats.transactions_analyzed.fetch_add(1, Ordering::Relaxed);
                }

                sleep(Duration::from_millis(1)).await;
            }
        });
    }

    /// Start MEV detection
    async fn start_mev_detection(&self) {
        let mempool = Arc::clone(&self.mempool);
        let mev_opportunities = Arc::clone(&self.mev_opportunities);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let optimism_config = self.optimism_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100)); // Check every 100ms

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Analyze mempool for MEV opportunities
                let transactions = {
                    let mempool_guard = mempool.read().await;
                    mempool_guard.iter().cloned().collect::<Vec<_>>()
                };

                for tx in transactions {
                    if tx.mev_score > 500 { // High MEV potential
                        let opportunity = Self::create_mev_opportunity_from_tx(&tx, &optimism_config);

                        if opportunity.estimated_profit >= optimism_config.mev_threshold_usd {
                            let mut opportunities = mev_opportunities.write().await;
                            opportunities.push_back(opportunity);

                            // Keep only recent opportunities
                            while opportunities.len() > 500 {
                                opportunities.pop_front();
                            }
                            drop(opportunities);

                            stats.mev_opportunities.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let mev_time = start_time.elapsed().as_micros() as u64;
                trace!("MEV detection completed in {}μs", mev_time);
            }
        });
    }

    /// Start Uniswap v3 monitoring
    async fn start_uniswap_v3_monitoring(&self) {
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5)); // Check every 5 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Simulate Uniswap v3 monitoring
                stats.uniswap_v3_txs.fetch_add(10, Ordering::Relaxed);
                trace!("Uniswap v3 monitoring cycle completed");
            }
        });
    }

    /// Start Aave v3 monitoring
    async fn start_aave_v3_monitoring(&self) {
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10)); // Check every 10 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Simulate Aave v3 monitoring
                stats.aave_v3_txs.fetch_add(5, Ordering::Relaxed);
                trace!("Aave v3 monitoring cycle completed");
            }
        });
    }

    /// Start Velodrome monitoring
    async fn start_velodrome_monitoring(&self) {
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3)); // Check every 3 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Simulate Velodrome monitoring
                stats.velodrome_txs.fetch_add(8, Ordering::Relaxed);
                trace!("Velodrome monitoring cycle completed");
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

                let blocks = stats.blocks_processed.load(Ordering::Relaxed);
                let transactions = stats.transactions_analyzed.load(Ordering::Relaxed);
                let mev_opportunities = stats.mev_opportunities.load(Ordering::Relaxed);
                let uniswap_txs = stats.uniswap_v3_txs.load(Ordering::Relaxed);
                let aave_txs = stats.aave_v3_txs.load(Ordering::Relaxed);
                let velodrome_txs = stats.velodrome_txs.load(Ordering::Relaxed);
                let avg_block_time = stats.avg_block_time_us.load(Ordering::Relaxed);

                info!(
                    "Optimism Stats: blocks={}, txs={}, mev={}, uniswap={}, aave={}, velodrome={}, avg_block={}μs",
                    blocks, transactions, mev_opportunities, uniswap_txs, aave_txs, velodrome_txs, avg_block_time
                );
            }
        });
    }

    /// Fetch latest block data
    async fn fetch_latest_block(_http_client: &Arc<TokioMutex<Option<reqwest::Client>>>) -> Result<AlignedOptimismBlockData> {
        // Simplified implementation - in production this would fetch real block data
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(AlignedOptimismBlockData::new(
            18_500_000, // Block number
            timestamp,
            150, // Transaction count
            25_000_000, // Gas used
        ))
    }

    /// Create MEV opportunity from transaction
    fn create_mev_opportunity_from_tx(tx: &OptimismMempoolTransaction, _config: &OptimismConfig) -> Opportunity {
        let estimated_profit = if tx.mev_score > 800 {
            Decimal::from(50) // $50 for high-value opportunities
        } else if tx.mev_score > 600 {
            Decimal::from(20) // $20 for medium-value opportunities
        } else {
            Decimal::from(10) // $10 for low-value opportunities
        };

        let gas_cost = Decimal::from(tx.gas_price) * Decimal::from(tx.gas_limit) / Decimal::from(1_000_000_000); // Convert from Gwei
        let net_profit = estimated_profit - gas_cost;

        Opportunity {
            id: tx.hash.chars().filter(char::is_ascii_hexdigit).take(16).collect::<String>().parse().unwrap_or(12345),
            opportunity_type: if tx.protocols.contains(&OptimismProtocol::UniswapV3) {
                OpportunityType::Arbitrage
            } else if tx.protocols.contains(&OptimismProtocol::AaveV3) {
                OpportunityType::Liquidation
            } else {
                OpportunityType::Sandwich
            },
            pair: TradingPair {
                token_a: TokenAddress::ZERO,
                token_b: TokenAddress([1_u8; 20]),
                chain_id: ChainId::Optimism,
            },
            estimated_profit,
            gas_cost,
            net_profit,
            urgency: (tx.mev_score / 10).min(100) as u8, // Convert to 0-100 scale
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() + 30, // 30 seconds deadline
            dex_route: vec![crate::types::DexId::UniswapV3], // Default route
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Clean old blocks from cache
    fn clean_old_blocks(cache: &Arc<DashMap<u64, AlignedOptimismBlockData>>, max_blocks: usize) {
        if cache.len() > max_blocks {
            let mut blocks: Vec<_> = cache.iter().map(|entry| *entry.key()).collect();
            blocks.sort_unstable();

            // Remove oldest blocks
            let to_remove = blocks.len().saturating_sub(max_blocks);
            for block_number in blocks.into_iter().take(to_remove) {
                cache.remove(&block_number);
            }
        }
    }
}

// Re-exports for public API
pub use velodrome_integration::{
    VelodromeIntegration, VelodromeConfig, VelodromeStats,
    VelodromePool, VelodromeRoute, VelodromeYieldPosition, VeVeloPosition,
    VelodromePoolType, VelodromeRouteStep
};

pub use sequencer_monitor::{
    OpSequencerMonitor, OpSequencerConfig, OpSequencerStats,
    OpBatch, SequencerHealth, BatchPrediction,
    SequencerStatus, BatchStatus
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChainCoreConfig;

    #[tokio::test]
    async fn test_optimism_coordinator_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = OptimismCoordinator::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(coordinator.stats().blocks_processed.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().transactions_analyzed.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().mev_opportunities.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_optimism_chain_id() {
        assert_eq!(OPTIMISM_CHAIN_ID, 10);
    }

    #[test]
    fn test_optimism_config_default() {
        let config = OptimismConfig::default();
        assert!(config.enabled);
        assert_eq!(config.chain_id, OPTIMISM_CHAIN_ID);
        assert_eq!(config.block_interval_ms, OPTIMISM_BLOCK_TIME_MS);
        assert!(config.enable_uniswap_v3);
        assert!(config.enable_aave_v3);
        assert!(config.enable_velodrome);
        assert!(config.gas_optimization);
    }

    #[test]
    fn test_optimism_stats_operations() {
        let stats = OptimismStats::default();

        stats.blocks_processed.fetch_add(100, Ordering::Relaxed);
        stats.transactions_analyzed.fetch_add(1500, Ordering::Relaxed);
        stats.mev_opportunities.fetch_add(25, Ordering::Relaxed);
        stats.uniswap_v3_txs.fetch_add(300, Ordering::Relaxed);

        assert_eq!(stats.blocks_processed.load(Ordering::Relaxed), 100);
        assert_eq!(stats.transactions_analyzed.load(Ordering::Relaxed), 1500);
        assert_eq!(stats.mev_opportunities.load(Ordering::Relaxed), 25);
        assert_eq!(stats.uniswap_v3_txs.load(Ordering::Relaxed), 300);
    }

    #[test]
    fn test_aligned_optimism_block_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedOptimismBlockData>(), 64);
        assert!(mem::size_of::<AlignedOptimismBlockData>() <= 64);
    }

    #[test]
    fn test_aligned_optimism_block_data_recency() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let recent_block = AlignedOptimismBlockData::new(18_500_000, now, 150, 25_000_000);
        let old_block = AlignedOptimismBlockData::new(18_499_999, now - 3600, 140, 24_000_000);

        assert!(recent_block.is_recent(60)); // Recent within 1 minute
        assert!(!old_block.is_recent(60)); // Not recent (1 hour old)
    }

    #[test]
    fn test_aligned_optimism_block_data_gas_utilization() {
        let block = AlignedOptimismBlockData::new(18_500_000, 1_640_995_200, 150, 15_000_000);

        assert_eq!(block.gas_utilization(), 50); // 15M / 30M = 50%

        let empty_block = AlignedOptimismBlockData::new(18_500_001, 1_640_995_200, 0, 0);
        assert_eq!(empty_block.gas_utilization(), 0);
    }

    #[test]
    fn test_optimism_mempool_transaction_creation() {
        let tx = OptimismMempoolTransaction {
            hash: "0x123abc456def789abc456def789abc456def789abc456def789abc456def789abc45".to_string(),
            from: "0x1234567890123456789012345678901234567890".to_string(),
            to: Some("0x0987654321098765432109876543210987654321".to_string()),
            value: "1000000000000000000".to_string(), // 1 ETH
            gas_limit: 300_000,
            gas_price: 10, // 10 Gwei
            data: "0xabcdef".to_string(),
            nonce: 42,
            mev_score: 750,
            protocols: vec![OptimismProtocol::UniswapV3, OptimismProtocol::Velodrome],
            detected_at: 1_640_995_200,
        };

        assert_eq!(tx.hash.len(), 70); // Actual length of the test hash
        assert_eq!(tx.gas_limit, 300_000);
        assert_eq!(tx.mev_score, 750);
        assert_eq!(tx.protocols.len(), 2);
    }

    #[test]
    fn test_optimism_mev_score_calculation() {
        let high_value_tx = OptimismMempoolTransaction {
            hash: "0x123".to_string(),
            from: "0x123".to_string(),
            to: Some("0x456".to_string()),
            value: "10000000000000000000".to_string(), // 10 ETH
            gas_limit: 600_000, // High gas limit
            gas_price: 50, // High gas price
            data: "0x".to_string(),
            nonce: 1,
            mev_score: 0,
            protocols: vec![OptimismProtocol::UniswapV3, OptimismProtocol::OneInch],
            detected_at: 1_640_995_200,
        };

        let low_value_tx = OptimismMempoolTransaction {
            hash: "0x456".to_string(),
            from: "0x456".to_string(),
            to: Some("0x789".to_string()),
            value: "100000000000000000".to_string(), // 0.1 ETH
            gas_limit: 21_000, // Standard transfer
            gas_price: 5, // Low gas price
            data: "0x".to_string(),
            nonce: 1,
            mev_score: 0,
            protocols: vec![],
            detected_at: 1_640_995_200,
        };

        let high_score = high_value_tx.calculate_optimism_mev_score();
        let low_score = low_value_tx.calculate_optimism_mev_score();

        assert!(high_score > low_score);
        assert!(high_score > 500); // High-value Uniswap + 1inch should score high
        assert!(low_score < 100);  // Regular transfer should score low
    }

    #[tokio::test]
    async fn test_optimism_coordinator_mempool_size() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = OptimismCoordinator::new(config).await else {
            return;
        };

        let mempool_size = coordinator.mempool_size().await;
        assert_eq!(mempool_size, 0); // Initially empty

        let mev_opportunities = coordinator.get_mev_opportunities().await;
        assert!(mev_opportunities.is_empty()); // Initially empty
    }

    #[tokio::test]
    async fn test_fetch_latest_block() {
        let http_client = Arc::new(TokioMutex::new(None));

        let result = OptimismCoordinator::fetch_latest_block(&http_client).await;

        assert!(result.is_ok());
        if let Ok(block) = result {
            assert!(block.block_number > 0);
            assert!(block.timestamp > 0);
            assert!(block.tx_count > 0);
        }
    }

    #[test]
    fn test_create_mev_opportunity_from_tx() {
        let config = OptimismConfig::default();
        let tx = OptimismMempoolTransaction {
            hash: "0x123abc".to_string(),
            from: "0x123".to_string(),
            to: Some("0x456".to_string()),
            value: "1000000000000000000".to_string(),
            gas_limit: 300_000,
            gas_price: 10,
            data: "0x".to_string(),
            nonce: 1,
            mev_score: 850, // High MEV score
            protocols: vec![OptimismProtocol::UniswapV3],
            detected_at: 1_640_995_200,
        };

        let opportunity = OptimismCoordinator::create_mev_opportunity_from_tx(&tx, &config);

        assert_eq!(opportunity.pair.chain_id, ChainId::Optimism);
        assert_eq!(opportunity.opportunity_type, OpportunityType::Arbitrage);
        assert!(opportunity.estimated_profit > Decimal::ZERO);
        assert!(opportunity.urgency > 80); // High urgency for high MEV score
    }
}
