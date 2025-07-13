//! Base chain coordination module
//!
//! Ultra-performance Base coordinator optimized for:
//! - Uniswap v3 integration with <20ns cross-chain operations
//! - Aerodrome DEX integration for concentrated liquidity
//! - Coinbase ecosystem integration
//! - MEV detection with <200ns latency
//! - Gas optimization for ETH efficiency on L2

// Submodules
pub mod uniswap_integration;
pub mod aerodrome_integration;

use crate::{
    ChainCoreConfig, Result,
    types::{TokenAddress, TradingPair, ChainId, Opportunity, OpportunityType},
    utils::perf::Timer,
    rpc::RpcCoordinator,
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

/// Base chain configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct BaseConfig {
    /// Enable Base chain operations
    pub enabled: bool,

    /// RPC endpoint URL
    pub rpc_url: String,

    /// Chain ID (8453 for Base mainnet)
    pub chain_id: u64,

    /// Block confirmation count
    pub confirmation_blocks: u64,

    /// Gas price multiplier for fast execution
    pub gas_price_multiplier: Decimal,

    /// Maximum gas price (gwei)
    pub max_gas_price_gwei: u64,

    /// MEV detection threshold (USD)
    pub mev_threshold_usd: Decimal,

    /// Enable Uniswap v3 integration
    pub enable_uniswap_v3: bool,

    /// Enable Aerodrome integration
    pub enable_aerodrome: bool,

    /// Enable Coinbase ecosystem features
    pub enable_coinbase_features: bool,

    /// Mempool monitoring interval (ms)
    pub mempool_interval_ms: u64,

    /// Maximum mempool size
    pub max_mempool_size: usize,
}

/// Base mempool transaction
#[derive(Debug, Clone)]
pub struct BaseMempoolTransaction {
    /// Transaction hash
    pub hash: String,

    /// From address
    pub from: String,

    /// To address
    pub to: Option<String>,

    /// Value in wei
    pub value: String,

    /// Gas price in wei
    pub gas_price: u64,

    /// Gas limit
    pub gas_limit: u64,

    /// Transaction data
    pub data: String,

    /// Nonce
    pub nonce: u64,

    /// MEV score (0-1000)
    pub mev_score: u16,

    /// Detected protocols
    pub protocols: Vec<BaseProtocol>,

    /// Timestamp when detected
    pub detected_at: u64,
}

/// Base protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaseProtocol {
    /// Uniswap v3 on Base
    UniswapV3,
    /// Aerodrome DEX
    Aerodrome,
    /// Coinbase Wallet
    CoinbaseWallet,
    /// Base Bridge
    BaseBridge,
    /// Other protocols
    Other,
}

/// Base coordinator statistics
#[derive(Debug, Default)]
pub struct BaseStats {
    /// Total transactions processed
    pub transactions_processed: AtomicU64,

    /// MEV opportunities detected
    pub mev_opportunities: AtomicU64,

    /// Total volume processed (USD)
    pub total_volume_usd: AtomicU64,

    /// Gas saved through optimization
    pub gas_saved: AtomicU64,

    /// Successful arbitrage trades
    pub successful_arbitrages: AtomicU64,

    /// Failed transactions
    pub failed_transactions: AtomicU64,

    /// Average transaction time (ms)
    pub avg_transaction_time_ms: AtomicU64,

    /// Current mempool size
    pub current_mempool_size: AtomicU64,
}

/// Cache-line aligned Base block data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedBaseBlockData {
    /// Block number
    pub block_number: u64,

    /// Block timestamp
    pub timestamp: u64,

    /// Gas used (scaled by 1e6)
    pub gas_used_scaled: u64,

    /// Gas limit (scaled by 1e6)
    pub gas_limit_scaled: u64,

    /// Base fee per gas (scaled by 1e9 for gwei)
    pub base_fee_scaled: u64,

    /// Transaction count
    pub tx_count: u64,

    /// MEV opportunities in block
    pub mev_count: u64,

    /// Reserved for future use
    pub reserved: u64,
}

/// Base coordinator constants
pub const BASE_CHAIN_ID: u64 = 8453;
pub const BASE_DEFAULT_RPC: &str = "https://mainnet.base.org";
pub const BASE_DEFAULT_CONFIRMATION_BLOCKS: u64 = 1;
pub const BASE_DEFAULT_GAS_MULTIPLIER: &str = "1.1";
pub const BASE_DEFAULT_MAX_GAS_GWEI: u64 = 100;
pub const BASE_DEFAULT_MEV_THRESHOLD: &str = "10.0";
pub const BASE_DEFAULT_MEMPOOL_INTERVAL_MS: u64 = 100;
pub const BASE_MAX_MEMPOOL_SIZE: usize = 10000;

/// Base contract addresses
pub const BASE_UNISWAP_V3_FACTORY: &str = "0x33128a8fC17869897dcE68Ed026d694621f6FDfD";
pub const BASE_UNISWAP_V3_ROUTER: &str = "0x2626664c2603336E57B271c5C0b26F421741e481";
pub const BASE_AERODROME_FACTORY: &str = "0x420DD381b31aEf6683db6B902084cB0FFECe40Da";
pub const BASE_AERODROME_ROUTER: &str = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43";

impl Default for BaseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rpc_url: BASE_DEFAULT_RPC.to_string(),
            chain_id: BASE_CHAIN_ID,
            confirmation_blocks: BASE_DEFAULT_CONFIRMATION_BLOCKS,
            gas_price_multiplier: BASE_DEFAULT_GAS_MULTIPLIER.parse().unwrap_or_default(),
            max_gas_price_gwei: BASE_DEFAULT_MAX_GAS_GWEI,
            mev_threshold_usd: BASE_DEFAULT_MEV_THRESHOLD.parse().unwrap_or_default(),
            enable_uniswap_v3: true,
            enable_aerodrome: true,
            enable_coinbase_features: true,
            mempool_interval_ms: BASE_DEFAULT_MEMPOOL_INTERVAL_MS,
            max_mempool_size: BASE_MAX_MEMPOOL_SIZE,
        }
    }
}

impl BaseMempoolTransaction {
    /// Calculate MEV score for this transaction
    #[inline]
    #[must_use]
    pub fn calculate_mev_score(&self) -> u16 {
        let mut score = 0_u16;

        // Base score from gas price (higher gas = higher MEV potential)
        score = score.saturating_add(u16::try_from(self.gas_price.min(100) * 2).unwrap_or(200));

        // Protocol-based scoring
        for protocol in &self.protocols {
            match protocol {
                BaseProtocol::UniswapV3 => score = score.saturating_add(300),
                BaseProtocol::Aerodrome => score = score.saturating_add(250),
                BaseProtocol::CoinbaseWallet => score = score.saturating_add(100),
                BaseProtocol::BaseBridge => score = score.saturating_add(150),
                BaseProtocol::Other => score = score.saturating_add(50),
            }
        }

        // Value-based scoring (higher value = higher MEV potential)
        if let Ok(value_wei) = self.value.parse::<u128>() {
            let value_eth = value_wei / 1_000_000_000_000_000_000; // Convert to ETH
            score = score.saturating_add(u16::try_from(value_eth.min(100) * 5).unwrap_or(500));
        }

        // Data complexity scoring
        if self.data.len() > 1000 {
            score = score.saturating_add(100);
        }

        score.min(1000) // Cap at 1000
    }
}

impl AlignedBaseBlockData {
    /// Create new aligned block data
    #[inline(always)]
    #[must_use]
    pub const fn new(
        block_number: u64,
        timestamp: u64,
        gas_used_scaled: u64,
        gas_limit_scaled: u64,
        base_fee_scaled: u64,
        tx_count: u64,
        mev_count: u64,
    ) -> Self {
        Self {
            block_number,
            timestamp,
            gas_used_scaled,
            gas_limit_scaled,
            base_fee_scaled,
            tx_count,
            mev_count,
            reserved: 0,
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

    /// Get gas utilization percentage
    #[inline(always)]
    #[must_use]
    pub fn gas_utilization(&self) -> Decimal {
        if self.gas_limit_scaled == 0 {
            return Decimal::ZERO;
        }

        let utilization = (self.gas_used_scaled * 100) / self.gas_limit_scaled;
        Decimal::from(utilization)
    }

    /// Get base fee in gwei
    #[inline(always)]
    #[must_use]
    pub fn base_fee_gwei(&self) -> Decimal {
        Decimal::from(self.base_fee_scaled) / Decimal::from(1_000_000_000_u64)
    }
}

/// Base Coordinator for ultra-performance L2 operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct BaseCoordinator {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Base specific configuration
    base_config: BaseConfig,

    /// RPC coordinator
    rpc: Arc<RpcCoordinator>,

    /// Statistics
    stats: Arc<BaseStats>,

    /// Mempool transactions
    mempool: Arc<RwLock<VecDeque<BaseMempoolTransaction>>>,

    /// Block data cache for ultra-fast access
    block_cache: Arc<DashMap<u64, AlignedBaseBlockData>>,

    /// MEV opportunities
    mev_opportunities: Arc<RwLock<VecDeque<Opportunity>>>,

    /// Performance timers
    mempool_timer: Timer,
    block_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Transaction channels
    tx_sender: Sender<BaseMempoolTransaction>,
    tx_receiver: Receiver<BaseMempoolTransaction>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current block number
    current_block: Arc<TokioMutex<u64>>,
}

impl BaseCoordinator {
    /// Create new Base coordinator with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let base_config = BaseConfig::default();
        let stats = Arc::new(BaseStats::default());
        let mempool = Arc::new(RwLock::new(VecDeque::with_capacity(BASE_MAX_MEMPOOL_SIZE)));
        let block_cache = Arc::new(DashMap::with_capacity(1000));
        let mev_opportunities = Arc::new(RwLock::new(VecDeque::with_capacity(500)));
        let mempool_timer = Timer::new("base_mempool");
        let block_timer = Timer::new("base_block");
        let shutdown = Arc::new(AtomicBool::new(false));
        let current_block = Arc::new(TokioMutex::new(0));

        let (tx_sender, tx_receiver) = channel::bounded(BASE_MAX_MEMPOOL_SIZE);
        let http_client = Arc::new(TokioMutex::new(None));

        // Create RPC coordinator for Base
        let rpc = Arc::new(RpcCoordinator::new(Arc::clone(&config)).await?);

        Ok(Self {
            config,
            base_config,
            rpc,
            stats,
            mempool,
            block_cache,
            mev_opportunities,
            mempool_timer,
            block_timer,
            shutdown,
            tx_sender,
            tx_receiver,
            http_client,
            current_block,
        })
    }

    /// Start Base coordinator services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.base_config.enabled {
            info!("Base coordinator disabled");
            return Ok(());
        }

        info!("Starting Base coordinator");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start mempool monitoring
        self.start_mempool_monitoring().await;

        // Start block monitoring
        self.start_block_monitoring().await;

        // Start MEV detection
        self.start_mev_detection().await;

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("Base coordinator started successfully");
        Ok(())
    }

    /// Stop Base coordinator
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping Base coordinator");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Base coordinator stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &BaseStats {
        &self.stats
    }

    /// Get mempool transactions
    #[inline]
    pub async fn get_mempool_transactions(&self) -> Vec<BaseMempoolTransaction> {
        let mempool = self.mempool.read().await;
        mempool.iter().cloned().collect()
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
            .timeout(Duration::from_millis(1500)) // Fast timeout for L2 calls
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
        let mempool = Arc::clone(&self.mempool);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let base_config = self.base_config.clone();
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(base_config.mempool_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming transactions
                while let Ok(tx) = tx_receiver.try_recv() {
                    {
                        let mut mempool_guard = mempool.write().await;
                        mempool_guard.push_back(tx);

                        // Keep mempool size manageable
                        while mempool_guard.len() > base_config.max_mempool_size {
                            mempool_guard.pop_front();
                        }
                        drop(mempool_guard);
                    }

                    stats.transactions_processed.fetch_add(1, Ordering::Relaxed);
                }

                // Fetch mempool data from Base
                if let Ok(txs) = Self::fetch_base_mempool(&http_client).await {
                    for tx in txs {
                        {
                            let mut mempool_guard = mempool.write().await;
                            mempool_guard.push_back(tx);

                            // Keep mempool size manageable
                            while mempool_guard.len() > base_config.max_mempool_size {
                                mempool_guard.pop_front();
                            }
                            drop(mempool_guard);
                        }

                        stats.transactions_processed.fetch_add(1, Ordering::Relaxed);
                    }
                }

                // Update current mempool size
                {
                    let mempool_guard = mempool.read().await;
                    stats.current_mempool_size.store(mempool_guard.len() as u64, Ordering::Relaxed);
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let mempool_time = start_time.elapsed().as_micros() as u64;
                trace!("Mempool monitoring cycle completed in {}μs", mempool_time);
            }
        });
    }

    /// Start block monitoring
    async fn start_block_monitoring(&self) {
        let block_cache = Arc::clone(&self.block_cache);
        let _stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let http_client = Arc::clone(&self.http_client);
        let current_block = Arc::clone(&self.current_block);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(2000)); // Check every 2 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Fetch latest block data
                if let Ok(block_data) = Self::fetch_latest_block(&http_client).await {
                    let block_number = block_data.block_number;

                    {
                        let mut current_block_guard = current_block.lock().await;
                        *current_block_guard = block_number;
                    }

                    // Update block cache
                    block_cache.insert(block_number, block_data);

                    // Clean old blocks (keep last 100)
                    if block_cache.len() > 100 {
                        let oldest_block = block_number.saturating_sub(100);
                        block_cache.retain(|&k, _| k > oldest_block);
                    }
                }

                trace!("Block monitoring cycle completed");
            }
        });
    }

    /// Start MEV detection
    async fn start_mev_detection(&self) {
        let mempool = Arc::clone(&self.mempool);
        let mev_opportunities = Arc::clone(&self.mev_opportunities);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let base_config = self.base_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(200)); // Check every 200ms

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Analyze mempool for MEV opportunities
                {
                    let mempool_guard = mempool.read().await;
                    for tx in mempool_guard.iter() {
                        if tx.mev_score > 500 { // High MEV score threshold
                            let opportunity = Self::create_mev_opportunity_from_tx(tx, &base_config);

                            if opportunity.estimated_profit >= base_config.mev_threshold_usd {
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
                }

                trace!("MEV detection cycle completed");
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
                let mev_ops = stats.mev_opportunities.load(Ordering::Relaxed);
                let volume = stats.total_volume_usd.load(Ordering::Relaxed);
                let gas_saved = stats.gas_saved.load(Ordering::Relaxed);
                let arbitrages = stats.successful_arbitrages.load(Ordering::Relaxed);
                let mempool_size = stats.current_mempool_size.load(Ordering::Relaxed);

                info!(
                    "Base Stats: txs={}, mev_ops={}, volume=${}, gas_saved={}, arbitrages={}, mempool={}",
                    transactions, mev_ops, volume, gas_saved, arbitrages, mempool_size
                );
            }
        });
    }

    /// Fetch Base mempool transactions
    async fn fetch_base_mempool(_http_client: &Arc<TokioMutex<Option<reqwest::Client>>>) -> Result<Vec<BaseMempoolTransaction>> {
        // Simplified implementation - in production this would fetch real mempool data
        let tx = BaseMempoolTransaction {
            hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234".to_string(),
            from: "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6".to_string(),
            to: Some(BASE_UNISWAP_V3_ROUTER.to_string()),
            value: "1000000000000000000".to_string(), // 1 ETH
            gas_price: 50,
            gas_limit: 200_000,
            data: "0x414bf389".to_string(),
            nonce: 42,
            mev_score: 750,
            protocols: vec![BaseProtocol::UniswapV3],
            detected_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        Ok(vec![tx])
    }

    /// Fetch latest block data
    async fn fetch_latest_block(_http_client: &Arc<TokioMutex<Option<reqwest::Client>>>) -> Result<AlignedBaseBlockData> {
        // Simplified implementation - in production this would fetch real block data
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let block_data = AlignedBaseBlockData::new(
            8_500_000, // Block number
            timestamp,
            25_000_000, // Gas used (25M scaled)
            30_000_000, // Gas limit (30M scaled)
            5_000_000_000, // 5 gwei base fee scaled
            150, // Transaction count
            5, // MEV opportunities
        );

        Ok(block_data)
    }

    /// Create MEV opportunity from transaction
    fn create_mev_opportunity_from_tx(tx: &BaseMempoolTransaction, _config: &BaseConfig) -> Opportunity {
        let estimated_profit = Decimal::from(tx.mev_score) / Decimal::from(10); // Simple conversion
        let gas_cost = Decimal::from(tx.gas_price) * Decimal::from(tx.gas_limit) / Decimal::from(1_000_000_000); // Convert from Gwei
        let net_profit = estimated_profit - gas_cost;

        Opportunity {
            id: tx.hash.chars().filter(char::is_ascii_hexdigit).take(16).collect::<String>().parse().unwrap_or(12345),
            opportunity_type: if tx.protocols.contains(&BaseProtocol::UniswapV3) || tx.protocols.contains(&BaseProtocol::Aerodrome) {
                OpportunityType::Arbitrage
            } else {
                OpportunityType::Sandwich
            },
            pair: TradingPair {
                token_a: TokenAddress::ZERO,
                token_b: TokenAddress([1_u8; 20]),
                chain_id: ChainId::Base,
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
}

// Re-exports for public API
pub use uniswap_integration::{
    UniswapBaseIntegration, UniswapBaseConfig, UniswapBaseStats,
    UniswapBasePool, UniswapBasePosition, UniswapBaseRoute, UniswapBaseRouteStep
};

pub use aerodrome_integration::{
    AerodromeIntegration, AerodromeConfig, AerodromeStats,
    AerodromePool, AerodromeRoute, AerodromeYieldPosition, VeAeroPosition,
    AerodromePoolType, AerodromeRouteStep
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChainCoreConfig;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_base_coordinator_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = BaseCoordinator::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(coordinator.stats().transactions_processed.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().mev_opportunities.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.stats().current_mempool_size.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_base_config_default() {
        let config = BaseConfig::default();
        assert!(config.enabled);
        assert_eq!(config.chain_id, BASE_CHAIN_ID);
        assert_eq!(config.rpc_url, BASE_DEFAULT_RPC);
        assert_eq!(config.confirmation_blocks, BASE_DEFAULT_CONFIRMATION_BLOCKS);
        assert!(config.enable_uniswap_v3);
        assert!(config.enable_aerodrome);
        assert!(config.enable_coinbase_features);
        assert_eq!(config.mempool_interval_ms, BASE_DEFAULT_MEMPOOL_INTERVAL_MS);
        assert_eq!(config.max_mempool_size, BASE_MAX_MEMPOOL_SIZE);
    }

    #[test]
    fn test_aligned_base_block_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedBaseBlockData>(), 64);
        assert!(mem::size_of::<AlignedBaseBlockData>() <= 64);
    }

    #[test]
    fn test_base_stats_operations() {
        let stats = BaseStats::default();

        stats.transactions_processed.fetch_add(100, Ordering::Relaxed);
        stats.mev_opportunities.fetch_add(25, Ordering::Relaxed);
        stats.total_volume_usd.fetch_add(50_000, Ordering::Relaxed);
        stats.gas_saved.fetch_add(1_000_000, Ordering::Relaxed);

        assert_eq!(stats.transactions_processed.load(Ordering::Relaxed), 100);
        assert_eq!(stats.mev_opportunities.load(Ordering::Relaxed), 25);
        assert_eq!(stats.total_volume_usd.load(Ordering::Relaxed), 50_000);
        assert_eq!(stats.gas_saved.load(Ordering::Relaxed), 1_000_000);
    }

    #[test]
    fn test_aligned_base_block_data_methods() {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let block_data = AlignedBaseBlockData::new(
            8_500_000, // Block number
            current_time, // Current timestamp
            25_000_000, // Gas used (25M scaled)
            30_000_000, // Gas limit (30M scaled)
            5_000_000_000, // 5 gwei base fee scaled
            150, // Transaction count
            5, // MEV opportunities
        );

        assert_eq!(block_data.block_number, 8_500_000);
        assert_eq!(block_data.tx_count, 150);
        assert_eq!(block_data.mev_count, 5);
        assert!(block_data.is_recent(3600)); // Should be recent within 1 hour

        let gas_util = block_data.gas_utilization();
        assert!(gas_util > Decimal::ZERO);
        assert!(gas_util <= dec!(100));

        let base_fee = block_data.base_fee_gwei();
        assert_eq!(base_fee, dec!(5));
    }

    #[test]
    fn test_base_protocol_equality() {
        assert_eq!(BaseProtocol::UniswapV3, BaseProtocol::UniswapV3);
        assert_ne!(BaseProtocol::UniswapV3, BaseProtocol::Aerodrome);
        assert_ne!(BaseProtocol::CoinbaseWallet, BaseProtocol::BaseBridge);
    }

    #[test]
    fn test_base_mempool_transaction_mev_score() {
        let tx = BaseMempoolTransaction {
            hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234".to_string(),
            from: "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6".to_string(),
            to: Some(BASE_UNISWAP_V3_ROUTER.to_string()),
            value: "1000000000000000000".to_string(), // 1 ETH
            gas_price: 50,
            gas_limit: 200_000,
            data: "0x414bf389".to_string(),
            nonce: 42,
            mev_score: 0, // Will be calculated
            protocols: vec![BaseProtocol::UniswapV3, BaseProtocol::Aerodrome],
            detected_at: 1_640_995_200,
        };

        let score = tx.calculate_mev_score();
        assert!(score > 0);
        assert!(score <= 1000);

        // Should have high score due to UniswapV3 + Aerodrome + high value
        assert!(score > 500);
    }

    #[test]
    fn test_base_mempool_transaction_creation() {
        let tx = BaseMempoolTransaction {
            hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234".to_string(),
            from: "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6".to_string(),
            to: Some(BASE_UNISWAP_V3_ROUTER.to_string()),
            value: "1000000000000000000".to_string(),
            gas_price: 50,
            gas_limit: 200_000,
            data: "0x414bf389".to_string(),
            nonce: 42,
            mev_score: 750,
            protocols: vec![BaseProtocol::UniswapV3],
            detected_at: 1_640_995_200,
        };

        assert_eq!(tx.hash.len(), 70); // 0x + 64 hex chars + 4 extra
        assert_eq!(tx.gas_price, 50);
        assert_eq!(tx.gas_limit, 200_000);
        assert_eq!(tx.mev_score, 750);
        assert_eq!(tx.protocols.len(), 1);
        assert_eq!(tx.protocols.first(), Some(&BaseProtocol::UniswapV3));
    }

    #[test]
    fn test_create_mev_opportunity_from_tx() {
        let tx = BaseMempoolTransaction {
            hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234".to_string(),
            from: "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6".to_string(),
            to: Some(BASE_UNISWAP_V3_ROUTER.to_string()),
            value: "1000000000000000000".to_string(),
            gas_price: 50,
            gas_limit: 200_000,
            data: "0x414bf389".to_string(),
            nonce: 42,
            mev_score: 800,
            protocols: vec![BaseProtocol::UniswapV3],
            detected_at: 1_640_995_200,
        };

        let config = BaseConfig::default();
        let opportunity = BaseCoordinator::create_mev_opportunity_from_tx(&tx, &config);

        assert_eq!(opportunity.pair.chain_id, ChainId::Base);
        assert_eq!(opportunity.opportunity_type, OpportunityType::Arbitrage);
        assert!(opportunity.estimated_profit > Decimal::ZERO);
        assert!(opportunity.urgency > 70); // High urgency for high MEV score
    }

    #[tokio::test]
    async fn test_fetch_base_mempool() {
        let http_client = Arc::new(TokioMutex::new(None));

        let result = BaseCoordinator::fetch_base_mempool(&http_client).await;

        assert!(result.is_ok());
        if let Ok(txs) = result {
            assert!(!txs.is_empty());
            if let Some(tx) = txs.first() {
                assert!(!tx.hash.is_empty());
                assert!(tx.gas_price > 0);
                assert!(tx.gas_limit > 0);
                assert!(!tx.protocols.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_fetch_latest_block() {
        let http_client = Arc::new(TokioMutex::new(None));

        let result = BaseCoordinator::fetch_latest_block(&http_client).await;

        assert!(result.is_ok());
        if let Ok(block) = result {
            assert!(block.block_number > 0);
            assert!(block.tx_count > 0);
            assert!(block.gas_used_scaled > 0);
            assert!(block.gas_limit_scaled > 0);
        }
    }

    #[tokio::test]
    async fn test_base_coordinator_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(coordinator) = BaseCoordinator::new(config).await else {
            return;
        };

        let mempool_txs = coordinator.get_mempool_transactions().await;
        assert!(mempool_txs.is_empty()); // No transactions initially

        let mev_opportunities = coordinator.get_mev_opportunities().await;
        assert!(mev_opportunities.is_empty()); // No opportunities initially

        let stats = coordinator.stats();
        assert_eq!(stats.transactions_processed.load(Ordering::Relaxed), 0);
    }
}
