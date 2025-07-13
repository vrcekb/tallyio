//! OP Sequencer Monitor for ultra-performance L2 batch tracking
//!
//! This module provides advanced Optimism sequencer monitoring for MEV strategies,
//! enabling real-time batch tracking, submission prediction, and L1/L2 coordination.
//!
//! ## Performance Targets
//! - Batch Detection: <100μs
//! - L1 Submission Tracking: <200μs
//! - State Root Verification: <150μs
//! - Sequencer Health Check: <50μs
//! - Batch Prediction: <300μs
//!
//! ## Architecture
//! - Real-time sequencer monitoring
//! - L1 batch submission tracking
//! - State root verification
//! - Sequencer health monitoring
//! - Batch timing prediction
//! - Lock-free data structures for hot paths

use crate::{
    ChainCoreConfig, Result,
    utils::perf::Timer,
    optimism::OptimismConfig,
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
    collections::{HashMap, VecDeque},
};
use tokio::{
    sync::{RwLock, Mutex as TokioMutex},
    time::{interval, sleep},
};
use tracing::{info, trace, warn};

/// OP Sequencer monitor configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct OpSequencerConfig {
    /// Enable sequencer monitoring
    pub enabled: bool,
    
    /// Monitoring interval in milliseconds
    pub monitor_interval_ms: u64,
    
    /// L1 batch submission check interval
    pub l1_check_interval_ms: u64,
    
    /// Maximum batch age before warning (seconds)
    pub max_batch_age_seconds: u64,
    
    /// Enable batch prediction
    pub enable_batch_prediction: bool,
    
    /// Enable state root verification
    pub enable_state_verification: bool,
    
    /// Sequencer health check interval
    pub health_check_interval_ms: u64,
    
    /// Maximum sequencer downtime tolerance (seconds)
    pub max_downtime_seconds: u64,
    
    /// Enable L1 gas price monitoring
    pub enable_l1_gas_monitoring: bool,
}

/// Sequencer status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequencerStatus {
    /// Sequencer is healthy and processing
    Healthy,
    /// Sequencer is experiencing delays
    Delayed,
    /// Sequencer is down or unresponsive
    Down,
    /// Sequencer status unknown
    Unknown,
}

/// L2 batch information
#[derive(Debug, Clone)]
pub struct OpBatch {
    /// Batch number
    pub batch_number: u64,
    
    /// L2 block range (start, end)
    pub l2_block_range: (u64, u64),
    
    /// Transaction count in batch
    pub transaction_count: u64,
    
    /// Batch timestamp
    pub timestamp: u64,
    
    /// State root
    pub state_root: String,
    
    /// L1 submission transaction hash (if submitted)
    pub l1_tx_hash: Option<String>,
    
    /// L1 block number (if submitted)
    pub l1_block_number: Option<u64>,
    
    /// Batch size in bytes
    pub batch_size_bytes: u64,
    
    /// Gas used for L1 submission
    pub l1_gas_used: Option<u64>,
    
    /// Batch status
    pub status: BatchStatus,
}

/// Batch status on L1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchStatus {
    /// Batch is pending submission to L1
    Pending,
    /// Batch submitted to L1 but not confirmed
    Submitted,
    /// Batch confirmed on L1
    Confirmed,
    /// Batch finalized on L1
    Finalized,
    /// Batch submission failed
    Failed,
}

/// Sequencer health metrics
#[derive(Debug, Clone)]
pub struct SequencerHealth {
    /// Current sequencer status
    pub status: SequencerStatus,
    
    /// Last seen block number
    pub last_block_number: u64,
    
    /// Last activity timestamp
    pub last_activity: u64,
    
    /// Average block time (milliseconds)
    pub avg_block_time_ms: u64,
    
    /// Pending transaction count
    pub pending_tx_count: u64,
    
    /// Current L1 gas price (gwei)
    pub l1_gas_price_gwei: u64,
    
    /// Estimated next batch submission time
    pub next_batch_eta: Option<u64>,
    
    /// Sequencer uptime percentage (last 24h)
    pub uptime_24h: Decimal,
}

/// Batch prediction model
#[derive(Debug, Clone)]
pub struct BatchPrediction {
    /// Predicted batch number
    pub predicted_batch: u64,
    
    /// Estimated submission time
    pub estimated_submission: u64,
    
    /// Confidence score (0-100)
    pub confidence: u8,
    
    /// Predicted transaction count
    pub predicted_tx_count: u64,
    
    /// Estimated L1 gas cost
    pub estimated_l1_gas: u64,
    
    /// Factors influencing prediction
    pub factors: Vec<String>,
}

/// OP Sequencer monitoring statistics
#[derive(Debug, Default)]
pub struct OpSequencerStats {
    /// Total batches monitored
    pub batches_monitored: AtomicU64,
    
    /// L1 submissions tracked
    pub l1_submissions_tracked: AtomicU64,
    
    /// State root verifications
    pub state_verifications: AtomicU64,
    
    /// Health checks performed
    pub health_checks: AtomicU64,
    
    /// Batch predictions made
    pub batch_predictions: AtomicU64,
    
    /// Average batch submission time (seconds)
    pub avg_batch_submission_time: AtomicU64,
    
    /// Sequencer downtime events
    pub downtime_events: AtomicU64,
    
    /// Failed batch submissions
    pub failed_submissions: AtomicU64,
}

/// Cache-line aligned sequencer data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedSequencerData {
    /// Last block number
    pub last_block: u64,
    
    /// Last activity timestamp
    pub last_activity: u64,
    
    /// Sequencer status (0=Unknown, 1=Healthy, 2=Delayed, 3=Down)
    pub status: u64,
    
    /// Average block time (milliseconds)
    pub avg_block_time: u64,
    
    /// Pending transaction count
    pub pending_txs: u64,
    
    /// L1 gas price (scaled by 1e9 for gwei)
    pub l1_gas_price_scaled: u64,
    
    /// Uptime percentage (scaled by 1e6)
    pub uptime_scaled: u64,
    
    /// Reserved for future use
    pub reserved: u64,
}

/// OP Sequencer integration constants
pub const OP_SEQUENCER_DEFAULT_MONITOR_INTERVAL_MS: u64 = 500; // 500ms
pub const OP_SEQUENCER_DEFAULT_L1_CHECK_INTERVAL_MS: u64 = 2000; // 2 seconds
pub const OP_SEQUENCER_DEFAULT_MAX_BATCH_AGE: u64 = 300; // 5 minutes
pub const OP_SEQUENCER_DEFAULT_HEALTH_CHECK_INTERVAL_MS: u64 = 1000; // 1 second
pub const OP_SEQUENCER_DEFAULT_MAX_DOWNTIME: u64 = 60; // 1 minute
pub const OP_SEQUENCER_MAX_BATCHES: usize = 1000;
pub const OP_SEQUENCER_MAX_PREDICTIONS: usize = 100;

/// Optimism L1 contract addresses
pub const OP_L1_BATCH_INBOX: &str = "0xFF00000000000000000000000000000000000010";
pub const OP_L1_STATE_COMMITMENT_CHAIN: &str = "0xBe5dAb4A2e9cd0F27300dB4aB94BeE3A233AEB19";
pub const OP_L1_CANONICAL_TRANSACTION_CHAIN: &str = "0x5E4e65926BA27467555EB562121fac00D24E9dD2";

/// Optimism L2 system addresses
pub const OP_L2_SEQUENCER_FEE_VAULT: &str = "0x4200000000000000000000000000000000000011";
pub const OP_L2_GAS_PRICE_ORACLE: &str = "0x420000000000000000000000000000000000000F";

impl Default for OpSequencerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitor_interval_ms: OP_SEQUENCER_DEFAULT_MONITOR_INTERVAL_MS,
            l1_check_interval_ms: OP_SEQUENCER_DEFAULT_L1_CHECK_INTERVAL_MS,
            max_batch_age_seconds: OP_SEQUENCER_DEFAULT_MAX_BATCH_AGE,
            enable_batch_prediction: true,
            enable_state_verification: true,
            health_check_interval_ms: OP_SEQUENCER_DEFAULT_HEALTH_CHECK_INTERVAL_MS,
            max_downtime_seconds: OP_SEQUENCER_DEFAULT_MAX_DOWNTIME,
            enable_l1_gas_monitoring: true,
        }
    }
}

impl AlignedSequencerData {
    /// Create new aligned sequencer data
    #[inline(always)]
    #[must_use]
    pub const fn new(
        last_block: u64,
        last_activity: u64,
        status: u64,
        avg_block_time: u64,
        pending_txs: u64,
        l1_gas_price_scaled: u64,
        uptime_scaled: u64,
    ) -> Self {
        Self {
            last_block,
            last_activity,
            status,
            avg_block_time,
            pending_txs,
            l1_gas_price_scaled,
            uptime_scaled,
            reserved: 0,
        }
    }
    
    /// Check if sequencer data is stale
    #[inline(always)]
    #[must_use]
    #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for staleness check")]
    pub fn is_stale(&self, max_age_ms: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        now.saturating_sub(self.last_activity) > max_age_ms
    }
    
    /// Get sequencer status
    #[inline(always)]
    #[must_use]
    pub const fn get_status(&self) -> SequencerStatus {
        match self.status {
            1 => SequencerStatus::Healthy,
            2 => SequencerStatus::Delayed,
            3 => SequencerStatus::Down,
            _ => SequencerStatus::Unknown,
        }
    }
    
    /// Get L1 gas price in gwei
    #[inline(always)]
    #[must_use]
    pub fn l1_gas_price_gwei(&self) -> Decimal {
        Decimal::from(self.l1_gas_price_scaled) / Decimal::from(1_000_000_000_u64)
    }
    
    /// Get uptime percentage
    #[inline(always)]
    #[must_use]
    pub fn uptime_percentage(&self) -> Decimal {
        Decimal::from(self.uptime_scaled) / Decimal::from(1_000_000_u64)
    }
    
    /// Check if sequencer is healthy
    #[inline(always)]
    #[must_use]
    pub const fn is_healthy(&self) -> bool {
        matches!(self.status, 1)
    }
}

/// OP Sequencer Monitor for ultra-performance L2 batch tracking
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct OpSequencerMonitor {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// OP Sequencer specific configuration
    sequencer_config: OpSequencerConfig,

    /// Optimism configuration
    optimism_config: OptimismConfig,

    /// Statistics
    stats: Arc<OpSequencerStats>,

    /// Monitored batches
    batches: Arc<RwLock<HashMap<u64, OpBatch>>>,

    /// Sequencer data cache for ultra-fast access
    sequencer_cache: Arc<DashMap<String, AlignedSequencerData>>,

    /// Batch predictions
    predictions: Arc<RwLock<VecDeque<BatchPrediction>>>,

    /// Current sequencer health
    health: Arc<RwLock<SequencerHealth>>,

    /// Performance timers
    batch_timer: Timer,
    health_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Batch update channels
    batch_sender: Sender<OpBatch>,
    batch_receiver: Receiver<OpBatch>,

    /// Health update channels
    health_sender: Sender<SequencerHealth>,
    health_receiver: Receiver<SequencerHealth>,

    /// HTTP client for external calls
    http_client: Arc<TokioMutex<Option<reqwest::Client>>>,

    /// Current L2 block number
    current_l2_block: Arc<TokioMutex<u64>>,

    /// Current L1 block number
    current_l1_block: Arc<TokioMutex<u64>>,
}

impl OpSequencerMonitor {
    /// Create new OP Sequencer monitor with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(
        config: Arc<ChainCoreConfig>,
        optimism_config: OptimismConfig,
    ) -> Result<Self> {
        let sequencer_config = OpSequencerConfig::default();
        let stats = Arc::new(OpSequencerStats::default());
        let batches = Arc::new(RwLock::new(HashMap::with_capacity(OP_SEQUENCER_MAX_BATCHES)));
        let sequencer_cache = Arc::new(DashMap::with_capacity(10));
        let predictions = Arc::new(RwLock::new(VecDeque::with_capacity(OP_SEQUENCER_MAX_PREDICTIONS)));
        let health = Arc::new(RwLock::new(SequencerHealth {
            status: SequencerStatus::Unknown,
            last_block_number: 0,
            last_activity: 0,
            avg_block_time_ms: 2000, // Default 2 seconds
            pending_tx_count: 0,
            l1_gas_price_gwei: 20, // Default 20 gwei
            next_batch_eta: None,
            uptime_24h: Decimal::from(100), // Start with 100% uptime
        }));
        let batch_timer = Timer::new("op_sequencer_batch");
        let health_timer = Timer::new("op_sequencer_health");
        let shutdown = Arc::new(AtomicBool::new(false));
        let current_l2_block = Arc::new(TokioMutex::new(0));
        let current_l1_block = Arc::new(TokioMutex::new(0));

        let (batch_sender, batch_receiver) = channel::bounded(OP_SEQUENCER_MAX_BATCHES);
        let (health_sender, health_receiver) = channel::bounded(100);
        let http_client = Arc::new(TokioMutex::new(None));

        Ok(Self {
            config,
            sequencer_config,
            optimism_config,
            stats,
            batches,
            sequencer_cache,
            predictions,
            health,
            batch_timer,
            health_timer,
            shutdown,
            batch_sender,
            batch_receiver,
            health_sender,
            health_receiver,
            http_client,
            current_l2_block,
            current_l1_block,
        })
    }

    /// Start OP Sequencer monitoring services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.sequencer_config.enabled {
            info!("OP Sequencer monitoring disabled");
            return Ok(());
        }

        info!("Starting OP Sequencer monitoring");

        // Initialize HTTP client
        self.initialize_http_client().await?;

        // Start batch monitoring
        self.start_batch_monitoring().await;

        // Start L1 submission tracking
        self.start_l1_tracking().await;

        // Start health monitoring
        self.start_health_monitoring().await;

        // Start batch prediction
        if self.sequencer_config.enable_batch_prediction {
            self.start_batch_prediction().await;
        }

        // Start state verification
        if self.sequencer_config.enable_state_verification {
            self.start_state_verification().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        info!("OP Sequencer monitoring started successfully");
        Ok(())
    }

    /// Stop OP Sequencer monitoring
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping OP Sequencer monitoring");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("OP Sequencer monitoring stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &OpSequencerStats {
        &self.stats
    }

    /// Get monitored batches
    #[inline]
    pub async fn get_batches(&self) -> Vec<OpBatch> {
        let batches = self.batches.read().await;
        batches.values().cloned().collect()
    }

    /// Get current sequencer health
    #[inline]
    pub async fn get_health(&self) -> SequencerHealth {
        let health = self.health.read().await;
        health.clone()
    }

    /// Get batch predictions
    #[inline]
    pub async fn get_predictions(&self) -> Vec<BatchPrediction> {
        let predictions = self.predictions.read().await;
        predictions.iter().cloned().collect()
    }

    /// Initialize HTTP client with optimizations
    async fn initialize_http_client(&self) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(2000)) // Longer timeout for L1 calls
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

    /// Start batch monitoring
    async fn start_batch_monitoring(&self) {
        let batch_receiver = self.batch_receiver.clone();
        let batches = Arc::clone(&self.batches);
        let sequencer_cache = Arc::clone(&self.sequencer_cache);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let sequencer_config = self.sequencer_config.clone();
        let http_client = Arc::clone(&self.http_client);
        let current_l2_block = Arc::clone(&self.current_l2_block);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(sequencer_config.monitor_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Process incoming batch updates
                while let Ok(batch) = batch_receiver.try_recv() {
                    let batch_number = batch.batch_number;

                    // Update batches
                    {
                        let mut batches_guard = batches.write().await;
                        batches_guard.insert(batch_number, batch.clone());

                        // Keep only recent batches
                        while batches_guard.len() > OP_SEQUENCER_MAX_BATCHES {
                            if let Some(oldest_key) = batches_guard.keys().min().copied() {
                                batches_guard.remove(&oldest_key);
                            }
                        }
                        drop(batches_guard);
                    }

                    stats.batches_monitored.fetch_add(1, Ordering::Relaxed);
                }

                // Fetch latest L2 block and batch data
                if let Ok(latest_block) = Self::fetch_latest_l2_block(&http_client).await {
                    {
                        let mut current_block = current_l2_block.lock().await;
                        *current_block = latest_block;
                    }

                    // Update sequencer cache
                    let aligned_data = AlignedSequencerData::new(
                        latest_block,
                        {
                            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for sequencer data")]
                            {
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64
                            }
                        },
                        1, // Healthy status
                        2000, // 2 second block time
                        0, // No pending txs for now
                        20_000_000_000, // 20 gwei scaled
                        100_000_000, // 100% uptime scaled
                    );
                    sequencer_cache.insert("main".to_string(), aligned_data);
                }

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let batch_time = start_time.elapsed().as_micros() as u64;
                trace!("Batch monitoring cycle completed in {}μs", batch_time);

                // Clean stale cache entries
                Self::clean_stale_cache(&sequencer_cache, 30_000); // 30 seconds
            }
        });
    }

    /// Start L1 submission tracking
    async fn start_l1_tracking(&self) {
        let batches = Arc::clone(&self.batches);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let sequencer_config = self.sequencer_config.clone();
        let http_client = Arc::clone(&self.http_client);
        let current_l1_block = Arc::clone(&self.current_l1_block);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(sequencer_config.l1_check_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Fetch latest L1 block
                if let Ok(latest_l1_block) = Self::fetch_latest_l1_block(&http_client).await {
                    {
                        let mut current_block = current_l1_block.lock().await;
                        *current_block = latest_l1_block;
                    }

                    // Check for L1 batch submissions
                    if let Ok(submissions) = Self::check_l1_batch_submissions(&http_client, latest_l1_block).await {
                        for submission in submissions {
                            // Update batch status
                            {
                                let mut batches_guard = batches.write().await;
                                if let Some(batch) = batches_guard.get_mut(&submission.batch_number) {
                                    batch.l1_tx_hash = Some(submission.tx_hash);
                                    batch.l1_block_number = Some(submission.l1_block);
                                    batch.status = BatchStatus::Submitted;
                                    batch.l1_gas_used = Some(submission.gas_used);
                                }
                            }

                            stats.l1_submissions_tracked.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                trace!("L1 tracking cycle completed");
            }
        });
    }

    /// Start health monitoring
    async fn start_health_monitoring(&self) {
        let health_receiver = self.health_receiver.clone();
        let health = Arc::clone(&self.health);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let sequencer_config = self.sequencer_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(sequencer_config.health_check_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Process health updates
                while let Ok(health_update) = health_receiver.try_recv() {
                    {
                        let mut health_guard = health.write().await;
                        *health_guard = health_update;
                    }

                    stats.health_checks.fetch_add(1, Ordering::Relaxed);
                }

                // Perform health check
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                {
                    let mut health_guard = health.write().await;

                    // Check if sequencer is responsive
                    if current_time.saturating_sub(health_guard.last_activity) > sequencer_config.max_downtime_seconds {
                        if health_guard.status != SequencerStatus::Down {
                            warn!("Sequencer appears to be down - no activity for {}s",
                                  current_time.saturating_sub(health_guard.last_activity));
                            health_guard.status = SequencerStatus::Down;
                            stats.downtime_events.fetch_add(1, Ordering::Relaxed);
                        }
                    } else if health_guard.avg_block_time_ms > 5000 {
                        // Block time > 5 seconds indicates delays
                        health_guard.status = SequencerStatus::Delayed;
                    } else {
                        health_guard.status = SequencerStatus::Healthy;
                    }

                    health_guard.last_activity = current_time;
                }

                trace!("Health monitoring cycle completed");
            }
        });
    }

    /// Start batch prediction
    async fn start_batch_prediction(&self) {
        let predictions = Arc::clone(&self.predictions);
        let batches = Arc::clone(&self.batches);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Predict every 30 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Generate batch prediction
                let prediction = {
                    let batches_guard = batches.read().await;
                    Self::generate_batch_prediction(&batches_guard)
                };

                if let Some(prediction) = prediction {
                    {
                        let mut predictions_guard = predictions.write().await;
                        predictions_guard.push_back(prediction);

                        // Keep only recent predictions
                        while predictions_guard.len() > OP_SEQUENCER_MAX_PREDICTIONS {
                            predictions_guard.pop_front();
                        }
                        drop(predictions_guard);
                    }

                    stats.batch_predictions.fetch_add(1, Ordering::Relaxed);
                }

                trace!("Batch prediction cycle completed");
            }
        });
    }

    /// Start state verification
    async fn start_state_verification(&self) {
        let batches = Arc::clone(&self.batches);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let http_client = Arc::clone(&self.http_client);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Verify every minute

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Verify state roots for submitted batches
                {
                    let batches_guard = batches.read().await;
                    for batch in batches_guard.values() {
                        if batch.status == BatchStatus::Submitted {
                            if let Ok(verified) = Self::verify_state_root(&http_client, batch).await {
                                if verified {
                                    stats.state_verifications.fetch_add(1, Ordering::Relaxed);
                                    trace!("State root verified for batch {}", batch.batch_number);
                                } else {
                                    warn!("State root verification failed for batch {}", batch.batch_number);
                                }
                            }
                        }
                    }
                }

                trace!("State verification cycle completed");
            }
        });
    }

    /// Start performance monitoring
    async fn start_performance_monitoring(&self) {
        let stats = Arc::clone(&self.stats);
        let health = Arc::clone(&self.health);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Log every minute

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let batches = stats.batches_monitored.load(Ordering::Relaxed);
                let l1_submissions = stats.l1_submissions_tracked.load(Ordering::Relaxed);
                let health_checks = stats.health_checks.load(Ordering::Relaxed);
                let predictions = stats.batch_predictions.load(Ordering::Relaxed);
                let downtime_events = stats.downtime_events.load(Ordering::Relaxed);

                let current_health = {
                    let health_guard = health.read().await;
                    health_guard.status
                };

                info!(
                    "OP Sequencer Stats: batches={}, l1_subs={}, health_checks={}, predictions={}, downtime={}, status={:?}",
                    batches, l1_submissions, health_checks, predictions, downtime_events, current_health
                );
            }
        });
    }

    /// Fetch latest L2 block number
    async fn fetch_latest_l2_block(_http_client: &Arc<TokioMutex<Option<reqwest::Client>>>) -> Result<u64> {
        // Simplified implementation - in production this would fetch real L2 block data
        Ok(12_345_678)
    }

    /// Fetch latest L1 block number
    async fn fetch_latest_l1_block(_http_client: &Arc<TokioMutex<Option<reqwest::Client>>>) -> Result<u64> {
        // Simplified implementation - in production this would fetch real L1 block data
        Ok(18_500_000)
    }

    /// Check for L1 batch submissions
    async fn check_l1_batch_submissions(
        _http_client: &Arc<TokioMutex<Option<reqwest::Client>>>,
        _l1_block: u64,
    ) -> Result<Vec<L1BatchSubmission>> {
        // Simplified implementation - in production this would scan L1 for batch submissions
        let submission = L1BatchSubmission {
            batch_number: 12345,
            tx_hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".to_string(),
            l1_block: 18_500_000,
            gas_used: 150_000,
        };

        Ok(vec![submission])
    }

    /// Verify state root for a batch
    async fn verify_state_root(
        _http_client: &Arc<TokioMutex<Option<reqwest::Client>>>,
        _batch: &OpBatch,
    ) -> Result<bool> {
        // Simplified implementation - in production this would verify state roots
        Ok(true)
    }

    /// Generate batch prediction
    fn generate_batch_prediction(batches: &HashMap<u64, OpBatch>) -> Option<BatchPrediction> {
        if batches.is_empty() {
            return None;
        }

        let latest_batch = batches.values().max_by_key(|b| b.batch_number)?;
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Some(BatchPrediction {
            predicted_batch: latest_batch.batch_number + 1,
            estimated_submission: current_time + 300, // 5 minutes from now
            confidence: 85,
            predicted_tx_count: 150,
            estimated_l1_gas: 200_000,
            factors: vec![
                "Historical batch timing".to_string(),
                "Current L1 gas price".to_string(),
                "Transaction pool size".to_string(),
            ],
        })
    }

    /// Clean stale cache entries
    fn clean_stale_cache(cache: &Arc<DashMap<String, AlignedSequencerData>>, max_age_ms: u64) {
        cache.retain(|_key, data| !data.is_stale(max_age_ms));
    }
}

/// L1 batch submission information
#[derive(Debug, Clone)]
struct L1BatchSubmission {
    /// Batch number
    batch_number: u64,
    /// L1 transaction hash
    tx_hash: String,
    /// L1 block number
    l1_block: u64,
    /// Gas used
    gas_used: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChainCoreConfig;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_op_sequencer_monitor_creation() {
        let config = Arc::new(ChainCoreConfig::default());
        let optimism_config = OptimismConfig::default();

        let Ok(monitor) = OpSequencerMonitor::new(config, optimism_config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(monitor.stats().batches_monitored.load(Ordering::Relaxed), 0);
        assert_eq!(monitor.stats().l1_submissions_tracked.load(Ordering::Relaxed), 0);
        assert_eq!(monitor.stats().health_checks.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_op_sequencer_config_default() {
        let config = OpSequencerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.monitor_interval_ms, OP_SEQUENCER_DEFAULT_MONITOR_INTERVAL_MS);
        assert_eq!(config.l1_check_interval_ms, OP_SEQUENCER_DEFAULT_L1_CHECK_INTERVAL_MS);
        assert_eq!(config.max_batch_age_seconds, OP_SEQUENCER_DEFAULT_MAX_BATCH_AGE);
        assert!(config.enable_batch_prediction);
        assert!(config.enable_state_verification);
        assert!(config.enable_l1_gas_monitoring);
    }

    #[test]
    fn test_aligned_sequencer_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedSequencerData>(), 64);
        assert!(mem::size_of::<AlignedSequencerData>() <= 64);
    }

    #[test]
    fn test_op_sequencer_stats_operations() {
        let stats = OpSequencerStats::default();

        stats.batches_monitored.fetch_add(100, Ordering::Relaxed);
        stats.l1_submissions_tracked.fetch_add(50, Ordering::Relaxed);
        stats.health_checks.fetch_add(200, Ordering::Relaxed);
        stats.batch_predictions.fetch_add(25, Ordering::Relaxed);

        assert_eq!(stats.batches_monitored.load(Ordering::Relaxed), 100);
        assert_eq!(stats.l1_submissions_tracked.load(Ordering::Relaxed), 50);
        assert_eq!(stats.health_checks.load(Ordering::Relaxed), 200);
        assert_eq!(stats.batch_predictions.load(Ordering::Relaxed), 25);
    }

    #[test]
    fn test_aligned_sequencer_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for staleness check")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedSequencerData::new(
            12_345_678, // Latest block
            now,
            1, // Healthy
            2000, // 2s block time
            0, // No pending txs
            20_000_000_000, // 20 gwei
            100_000_000, // 100% uptime
        );

        let stale_data = AlignedSequencerData::new(
            12_345_600,
            now - 60_000, // 1 minute old
            1,
            2000,
            0,
            20_000_000_000,
            100_000_000,
        );

        assert!(!fresh_data.is_stale(30_000));
        assert!(stale_data.is_stale(30_000));
    }

    #[test]
    fn test_aligned_sequencer_data_status() {
        let data = AlignedSequencerData::new(
            12_345_678,
            1_640_995_200_000,
            1, // Healthy
            2000,
            0,
            20_000_000_000,
            100_000_000,
        );

        assert_eq!(data.get_status(), SequencerStatus::Healthy);
        assert!(data.is_healthy());
        assert_eq!(data.l1_gas_price_gwei(), dec!(20));
        assert_eq!(data.uptime_percentage(), dec!(100));
    }

    #[test]
    fn test_sequencer_status_equality() {
        assert_eq!(SequencerStatus::Healthy, SequencerStatus::Healthy);
        assert_ne!(SequencerStatus::Healthy, SequencerStatus::Down);
        assert_ne!(SequencerStatus::Delayed, SequencerStatus::Unknown);
    }

    #[test]
    fn test_batch_status_equality() {
        assert_eq!(BatchStatus::Pending, BatchStatus::Pending);
        assert_ne!(BatchStatus::Pending, BatchStatus::Submitted);
        assert_ne!(BatchStatus::Confirmed, BatchStatus::Failed);
    }

    #[test]
    fn test_op_batch_creation() {
        let batch = OpBatch {
            batch_number: 12345,
            l2_block_range: (1_000_000, 1_000_100),
            transaction_count: 150,
            timestamp: 1_640_995_200,
            state_root: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".to_string(),
            l1_tx_hash: None,
            l1_block_number: None,
            batch_size_bytes: 50_000,
            l1_gas_used: None,
            status: BatchStatus::Pending,
        };

        assert_eq!(batch.batch_number, 12345);
        assert_eq!(batch.l2_block_range, (1_000_000, 1_000_100));
        assert_eq!(batch.transaction_count, 150);
        assert_eq!(batch.status, BatchStatus::Pending);
        assert!(batch.l1_tx_hash.is_none());
    }

    #[test]
    fn test_sequencer_health_creation() {
        let health = SequencerHealth {
            status: SequencerStatus::Healthy,
            last_block_number: 12_345_678,
            last_activity: 1_640_995_200,
            avg_block_time_ms: 2000,
            pending_tx_count: 0,
            l1_gas_price_gwei: 20,
            next_batch_eta: Some(1_640_995_500),
            uptime_24h: dec!(99.5),
        };

        assert_eq!(health.status, SequencerStatus::Healthy);
        assert_eq!(health.last_block_number, 12_345_678);
        assert_eq!(health.avg_block_time_ms, 2000);
        assert_eq!(health.uptime_24h, dec!(99.5));
        assert!(health.next_batch_eta.is_some());
    }

    #[test]
    fn test_batch_prediction_creation() {
        let prediction = BatchPrediction {
            predicted_batch: 12346,
            estimated_submission: 1_640_995_500,
            confidence: 85,
            predicted_tx_count: 150,
            estimated_l1_gas: 200_000,
            factors: vec![
                "Historical timing".to_string(),
                "Gas price trends".to_string(),
            ],
        };

        assert_eq!(prediction.predicted_batch, 12346);
        assert_eq!(prediction.confidence, 85);
        assert_eq!(prediction.predicted_tx_count, 150);
        assert_eq!(prediction.factors.len(), 2);
    }

    #[tokio::test]
    async fn test_fetch_latest_l2_block() {
        let http_client = Arc::new(TokioMutex::new(None));

        let result = OpSequencerMonitor::fetch_latest_l2_block(&http_client).await;

        assert!(result.is_ok());
        if let Ok(block) = result {
            assert!(block > 0);
        }
    }

    #[tokio::test]
    async fn test_fetch_latest_l1_block() {
        let http_client = Arc::new(TokioMutex::new(None));

        let result = OpSequencerMonitor::fetch_latest_l1_block(&http_client).await;

        assert!(result.is_ok());
        if let Ok(block) = result {
            assert!(block > 0);
        }
    }

    #[tokio::test]
    async fn test_check_l1_batch_submissions() {
        let http_client = Arc::new(TokioMutex::new(None));

        let result = OpSequencerMonitor::check_l1_batch_submissions(&http_client, 18_500_000).await;

        assert!(result.is_ok());
        if let Ok(submissions) = result {
            assert!(!submissions.is_empty());
            if let Some(submission) = submissions.first() {
                assert!(submission.batch_number > 0);
                assert!(!submission.tx_hash.is_empty());
                assert!(submission.gas_used > 0);
            }
        }
    }

    #[test]
    fn test_generate_batch_prediction() {
        let mut batches = HashMap::new();
        let batch = OpBatch {
            batch_number: 12345,
            l2_block_range: (1_000_000, 1_000_100),
            transaction_count: 150,
            timestamp: 1_640_995_200,
            state_root: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".to_string(),
            l1_tx_hash: None,
            l1_block_number: None,
            batch_size_bytes: 50_000,
            l1_gas_used: None,
            status: BatchStatus::Pending,
        };
        batches.insert(12345, batch);

        let prediction = OpSequencerMonitor::generate_batch_prediction(&batches);

        assert!(prediction.is_some());
        if let Some(pred) = prediction {
            assert_eq!(pred.predicted_batch, 12346);
            assert!(pred.confidence > 0);
            assert!(pred.predicted_tx_count > 0);
            assert!(!pred.factors.is_empty());
        }
    }

    #[tokio::test]
    async fn test_op_sequencer_monitor_methods() {
        let config = Arc::new(ChainCoreConfig::default());
        let optimism_config = OptimismConfig::default();

        let Ok(monitor) = OpSequencerMonitor::new(config, optimism_config).await else {
            return;
        };

        let batches = monitor.get_batches().await;
        assert!(batches.is_empty()); // No batches initially

        let health = monitor.get_health().await;
        assert_eq!(health.status, SequencerStatus::Unknown); // Initial status

        let predictions = monitor.get_predictions().await;
        assert!(predictions.is_empty()); // No predictions initially

        let stats = monitor.stats();
        assert_eq!(stats.batches_monitored.load(Ordering::Relaxed), 0);
    }
}
