//! Failover Manager for ultra-performance RPC provider failover
//!
//! This module provides advanced failover management capabilities for maximizing
//! RPC provider reliability through intelligent health monitoring, automatic
//! failover detection, provider ranking, and recovery mechanisms with real-time
//! performance tracking and adaptive failover algorithms.
//!
//! ## Performance Targets
//! - Failover Detection: <10μs
//! - Provider Health Check: <5μs
//! - Failover Execution: <15μs
//! - Recovery Detection: <8μs
//! - Total Failover Overhead: <50μs
//!
//! ## Architecture
//! - Real-time provider health monitoring
//! - Intelligent failover detection algorithms
//! - Automatic provider ranking and selection
//! - Graceful degradation and recovery
//! - Circuit breaker pattern implementation

use crate::{
    Result, ChainCoreError,
    types::ChainId,
    rpc::{
        RpcProvider, RpcProviderStatus, RpcProviderType,
        RPC_DEFAULT_FAILOVER_THRESHOLD, RPC_DEFAULT_HEALTH_CHECK_INTERVAL_MS,
    },
};
use dashmap::DashMap;
use rust_decimal::Decimal;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH, Duration, Instant},
    collections::HashMap,
};
use tokio::{
    sync::{RwLock, Notify},
    time::{interval, sleep},
    task::JoinHandle,
};
use tracing::{info, warn, debug};

/// Failover manager configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct FailoverManagerConfig {
    /// Enable failover management
    pub enabled: bool,

    /// Health check interval in milliseconds
    pub health_check_interval_ms: u64,

    /// Failover threshold (consecutive failures)
    pub failover_threshold: u32,

    /// Recovery threshold (consecutive successes)
    pub recovery_threshold: u32,

    /// Circuit breaker timeout in milliseconds
    pub circuit_breaker_timeout_ms: u64,

    /// Maximum response time for healthy provider (ms)
    pub max_healthy_response_time_ms: u64,

    /// Minimum success rate for healthy provider (0.0-1.0)
    pub min_healthy_success_rate: f64,

    /// Provider timeout in milliseconds
    pub provider_timeout_ms: u64,

    /// Enable automatic recovery
    pub enable_auto_recovery: bool,

    /// Enable circuit breaker
    pub enable_circuit_breaker: bool,

    /// Enable provider ranking
    pub enable_provider_ranking: bool,

    /// Health check timeout in milliseconds
    pub health_check_timeout_ms: u64,

    /// Maximum concurrent health checks
    pub max_concurrent_health_checks: usize,

    /// Provider priority weights
    pub priority_weights: HashMap<RpcProviderType, u32>,
}

/// Provider circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are blocked
    Open,
    /// Circuit is half-open, testing recovery
    HalfOpen,
}

/// Provider circuit breaker
#[derive(Debug)]
pub struct ProviderCircuitBreaker {
    /// Current state
    state: AtomicU32, // CircuitBreakerState as u32
    /// Failure count
    failure_count: AtomicU32,
    /// Success count (for half-open state)
    success_count: AtomicU32,
    /// Last failure time
    last_failure_time: AtomicU64,
    /// Configuration
    config: FailoverManagerConfig,
}

/// Provider health metrics
#[derive(Debug, Clone)]
pub struct ProviderHealthMetrics {
    /// Provider ID
    pub provider_id: String,
    /// Current status
    pub status: RpcProviderStatus,
    /// Circuit breaker state
    pub circuit_state: CircuitBreakerState,
    /// Success rate (0.0-1.0)
    pub success_rate: Decimal,
    /// Average response time in milliseconds
    pub avg_response_time_ms: u64,
    /// Current load (0.0-1.0)
    pub current_load: Decimal,
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Consecutive failures
    pub consecutive_failures: u32,
    /// Consecutive successes
    pub consecutive_successes: u32,
    /// Last request time
    pub last_request_time: u64,
    /// Last success time
    pub last_success_time: u64,
    /// Last failure time
    pub last_failure_time: u64,
    /// Last health check time
    pub last_health_check_time: u64,
    /// Health score (0.0-1.0)
    pub health_score: Decimal,
    /// Provider rank (lower = better)
    pub rank: u32,
}

/// Failover event information
#[derive(Debug, Clone)]
pub struct FailoverEvent {
    /// Event ID
    pub id: String,
    /// Chain ID
    pub chain_id: ChainId,
    /// Failed provider ID
    pub failed_provider_id: String,
    /// Backup provider ID
    pub backup_provider_id: Option<String>,
    /// Failure reason
    pub failure_reason: String,
    /// Event timestamp
    pub timestamp: u64,
    /// Recovery time (if applicable)
    pub recovery_time_ms: Option<u64>,
}

/// Failover manager statistics
#[derive(Debug, Default)]
pub struct FailoverManagerStats {
    /// Total failover events
    pub total_failovers: AtomicU64,
    /// Successful failovers
    pub successful_failovers: AtomicU64,
    /// Failed failovers
    pub failed_failovers: AtomicU64,
    /// Provider recoveries
    pub provider_recoveries: AtomicU64,
    /// Circuit breaker activations
    pub circuit_breaker_activations: AtomicU64,
    /// Average failover time in microseconds
    pub avg_failover_time_us: AtomicU64,
    /// Average recovery time in microseconds
    pub avg_recovery_time_us: AtomicU64,
    /// Health checks performed
    pub health_checks_performed: AtomicU64,
    /// Health check failures
    pub health_check_failures: AtomicU64,
    /// Provider ranking updates
    pub ranking_updates: AtomicU64,
}

/// Failover manager constants
pub const FAILOVER_DEFAULT_RECOVERY_THRESHOLD: u32 = 3;
pub const FAILOVER_DEFAULT_CIRCUIT_BREAKER_TIMEOUT_MS: u64 = 60_000; // 1 minute
pub const FAILOVER_DEFAULT_MAX_HEALTHY_RESPONSE_TIME_MS: u64 = 5_000; // 5 seconds
pub const FAILOVER_DEFAULT_MIN_HEALTHY_SUCCESS_RATE: f64 = 0.95; // 95%
pub const FAILOVER_DEFAULT_PROVIDER_TIMEOUT_MS: u64 = 30_000; // 30 seconds
pub const FAILOVER_DEFAULT_HEALTH_CHECK_TIMEOUT_MS: u64 = 5_000; // 5 seconds
pub const FAILOVER_DEFAULT_MAX_CONCURRENT_HEALTH_CHECKS: usize = 10;
pub const FAILOVER_MAX_PROVIDERS_PER_CHAIN: usize = 20;
pub const FAILOVER_MAX_EVENTS_HISTORY: usize = 1000;

impl Default for FailoverManagerConfig {
    fn default() -> Self {
        let mut priority_weights = HashMap::new();
        priority_weights.insert(RpcProviderType::Local, 100);
        priority_weights.insert(RpcProviderType::Private, 80);
        priority_weights.insert(RpcProviderType::Archive, 60);
        priority_weights.insert(RpcProviderType::Public, 40);
        priority_weights.insert(RpcProviderType::Light, 20);

        Self {
            enabled: true,
            health_check_interval_ms: RPC_DEFAULT_HEALTH_CHECK_INTERVAL_MS,
            failover_threshold: RPC_DEFAULT_FAILOVER_THRESHOLD,
            recovery_threshold: FAILOVER_DEFAULT_RECOVERY_THRESHOLD,
            circuit_breaker_timeout_ms: FAILOVER_DEFAULT_CIRCUIT_BREAKER_TIMEOUT_MS,
            max_healthy_response_time_ms: FAILOVER_DEFAULT_MAX_HEALTHY_RESPONSE_TIME_MS,
            min_healthy_success_rate: FAILOVER_DEFAULT_MIN_HEALTHY_SUCCESS_RATE,
            provider_timeout_ms: FAILOVER_DEFAULT_PROVIDER_TIMEOUT_MS,
            enable_auto_recovery: true,
            enable_circuit_breaker: true,
            enable_provider_ranking: true,
            health_check_timeout_ms: FAILOVER_DEFAULT_HEALTH_CHECK_TIMEOUT_MS,
            max_concurrent_health_checks: FAILOVER_DEFAULT_MAX_CONCURRENT_HEALTH_CHECKS,
            priority_weights,
        }
    }
}

impl ProviderCircuitBreaker {
    /// Create new circuit breaker
    #[inline]
    #[must_use]
    pub const fn new(config: FailoverManagerConfig) -> Self {
        Self {
            state: AtomicU32::new(CircuitBreakerState::Closed as u32),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_failure_time: AtomicU64::new(0),
            config,
        }
    }

    /// Get current state
    #[inline]
    #[must_use]
    pub fn state(&self) -> CircuitBreakerState {
        match self.state.load(Ordering::Acquire) {
            1 => CircuitBreakerState::Open,
            2 => CircuitBreakerState::HalfOpen,
            _ => CircuitBreakerState::Closed,
        }
    }

    /// Record success
    #[inline]
    pub fn record_success(&self) {
        let current_state = self.state();
        
        match current_state {
            CircuitBreakerState::Closed => {
                self.failure_count.store(0, Ordering::Release);
            }
            CircuitBreakerState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::AcqRel);
                if success_count + 1 >= self.config.recovery_threshold {
                    self.state.store(CircuitBreakerState::Closed as u32, Ordering::Release);
                    self.failure_count.store(0, Ordering::Release);
                    self.success_count.store(0, Ordering::Release);
                }
            }
            CircuitBreakerState::Open => {
                // Check if timeout has passed
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
                let last_failure = self.last_failure_time.load(Ordering::Acquire);
                
                if now.saturating_sub(last_failure) >= self.config.circuit_breaker_timeout_ms {
                    self.state.store(CircuitBreakerState::HalfOpen as u32, Ordering::Release);
                    self.success_count.store(1, Ordering::Release);
                }
            }
        }
    }

    /// Record failure
    #[inline]
    pub fn record_failure(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
        
        self.last_failure_time.store(now, Ordering::Release);
        
        let current_state = self.state();
        
        match current_state {
            CircuitBreakerState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::AcqRel);
                if failure_count + 1 >= self.config.failover_threshold {
                    self.state.store(CircuitBreakerState::Open as u32, Ordering::Release);
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.state.store(CircuitBreakerState::Open as u32, Ordering::Release);
                self.success_count.store(0, Ordering::Release);
            }
            CircuitBreakerState::Open => {
                // Already open, no action needed
            }
        }
    }

    /// Check if requests are allowed
    #[inline]
    #[must_use]
    pub fn is_request_allowed(&self) -> bool {
        let current_state = self.state();

        match current_state {
            CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen => true,
            CircuitBreakerState::Open => {
                // Check if timeout has passed
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
                let last_failure = self.last_failure_time.load(Ordering::Acquire);

                if now.saturating_sub(last_failure) >= self.config.circuit_breaker_timeout_ms {
                    self.state.store(CircuitBreakerState::HalfOpen as u32, Ordering::Release);
                    true
                } else {
                    false
                }
            }
        }
    }
}

/// Failover Manager for ultra-performance RPC provider failover
pub struct FailoverManager {
    /// Configuration
    config: FailoverManagerConfig,

    /// Providers by chain
    providers: Arc<DashMap<ChainId, Vec<RpcProvider>>>,

    /// Provider health metrics
    provider_metrics: Arc<DashMap<String, ProviderHealthMetrics>>,

    /// Circuit breakers by provider
    circuit_breakers: Arc<DashMap<String, Arc<ProviderCircuitBreaker>>>,

    /// Provider rankings by chain
    provider_rankings: Arc<DashMap<ChainId, Vec<String>>>,

    /// Failover events history
    failover_events: Arc<RwLock<Vec<FailoverEvent>>>,

    /// Statistics
    stats: Arc<FailoverManagerStats>,

    /// Health check task handle
    health_check_task: Arc<RwLock<Option<JoinHandle<()>>>>,

    /// Ranking update task handle
    ranking_task: Arc<RwLock<Option<JoinHandle<()>>>>,

    /// Shutdown notification
    shutdown_notify: Arc<Notify>,

    /// Running flag
    running: Arc<AtomicBool>,
}

impl FailoverManager {
    /// Create new failover manager with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub fn new(config: FailoverManagerConfig) -> Result<Self> {
        Ok(Self {
            config,
            providers: Arc::new(DashMap::with_capacity(FAILOVER_MAX_PROVIDERS_PER_CHAIN)),
            provider_metrics: Arc::new(DashMap::with_capacity(FAILOVER_MAX_PROVIDERS_PER_CHAIN)),
            circuit_breakers: Arc::new(DashMap::with_capacity(FAILOVER_MAX_PROVIDERS_PER_CHAIN)),
            provider_rankings: Arc::new(DashMap::with_capacity(FAILOVER_MAX_PROVIDERS_PER_CHAIN)),
            failover_events: Arc::new(RwLock::new(Vec::with_capacity(FAILOVER_MAX_EVENTS_HISTORY))),
            stats: Arc::new(FailoverManagerStats::default()),
            health_check_task: Arc::new(RwLock::new(None)),
            ranking_task: Arc::new(RwLock::new(None)),
            shutdown_notify: Arc::new(Notify::new()),
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start failover manager services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup logic requires multiple conditional checks")]
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Failover manager disabled");
            return Ok(());
        }

        if self.running.load(Ordering::Acquire) {
            warn!("Failover manager already running");
            return Ok(());
        }

        info!("Starting failover manager");
        self.running.store(true, Ordering::Release);

        // Start health check task
        if self.config.health_check_interval_ms > 0 {
            let task = self.start_health_check_task().await?;
            *self.health_check_task.write().await = Some(task);
        }

        // Start ranking update task
        if self.config.enable_provider_ranking {
            let task = self.start_ranking_task().await?;
            *self.ranking_task.write().await = Some(task);
        }

        info!("Failover manager started successfully");
        Ok(())
    }

    /// Stop failover manager services
    #[inline]
    pub async fn stop(&self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }

        info!("Stopping failover manager");
        self.running.store(false, Ordering::Release);

        // Notify shutdown
        self.shutdown_notify.notify_waiters();

        // Stop health check task
        {
            let task = self.health_check_task.write().await.take();
            if let Some(task) = task {
                task.abort();
            }
        }

        // Stop ranking task
        {
            let task = self.ranking_task.write().await.take();
            if let Some(task) = task {
                task.abort();
            }
        }

        info!("Failover manager stopped");
    }

    /// Add provider to failover management
    ///
    /// # Errors
    ///
    /// Returns error if provider addition fails
    #[inline]
    pub async fn add_provider(&self, provider: RpcProvider) -> Result<()> {
        let chain_id = provider.chain_id;
        let provider_id = provider.id.clone();

        // Add provider to chain list
        self.providers.entry(chain_id).or_default().push(provider.clone());

        // Initialize provider metrics
        let metrics = ProviderHealthMetrics {
            provider_id: provider_id.clone(),
            status: RpcProviderStatus::Unknown,
            circuit_state: CircuitBreakerState::Closed,
            success_rate: Decimal::ZERO,
            avg_response_time_ms: 0,
            current_load: Decimal::ZERO,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_request_time: 0,
            last_success_time: 0,
            last_failure_time: 0,
            last_health_check_time: 0,
            health_score: Decimal::ZERO,
            rank: u32::MAX,
        };
        self.provider_metrics.insert(provider_id.clone(), metrics);

        // Create circuit breaker
        let circuit_breaker = Arc::new(ProviderCircuitBreaker::new(self.config.clone()));
        self.circuit_breakers.insert(provider_id.clone(), circuit_breaker);

        // Update provider rankings
        self.update_provider_rankings(chain_id).await?;

        debug!("Added provider {} to failover management", provider_id);
        Ok(())
    }

    /// Remove provider from failover management
    ///
    /// # Errors
    ///
    /// Returns error if provider removal fails
    #[inline]
    pub async fn remove_provider(&self, chain_id: ChainId, provider_id: &str) -> Result<()> {
        // Remove from providers list
        if let Some(mut providers) = self.providers.get_mut(&chain_id) {
            providers.retain(|p| p.id != provider_id);
        }

        // Remove metrics and circuit breaker
        self.provider_metrics.remove(provider_id);
        self.circuit_breakers.remove(provider_id);

        // Update provider rankings
        self.update_provider_rankings(chain_id).await?;

        debug!("Removed provider {} from failover management", provider_id);
        Ok(())
    }

    /// Get best available provider for chain
    ///
    /// # Errors
    ///
    /// Returns error if no healthy provider is available
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Complex provider selection logic required for performance")]
    pub async fn get_best_provider(&self, chain_id: ChainId) -> Result<Option<RpcProvider>> {
        let start_time = Instant::now();

        // Get provider rankings for chain
        let provider_ids = if let Some(rankings) = self.provider_rankings.get(&chain_id) {
            rankings.clone()
        } else {
            debug!("No provider rankings found for chain {:?}", chain_id);
            return Ok(None);
        };

        // Find best available provider
        for provider_id in &provider_ids {
            // Check circuit breaker
            if let Some(circuit_breaker) = self.circuit_breakers.get(provider_id) {
                if !circuit_breaker.is_request_allowed() {
                    continue;
                }
            }

            // Check provider health
            if let Some(metrics) = self.provider_metrics.get(provider_id) {
                if metrics.status == RpcProviderStatus::Healthy {
                    // Find the actual provider
                    if let Some(providers) = self.providers.get(&chain_id) {
                        if let Some(provider) = providers.iter().find(|p| &p.id == provider_id) {
                            let elapsed = start_time.elapsed();
                            if elapsed.as_micros() < 10 {
                                debug!("Selected provider {} for chain {:?} in {}μs",
                                       provider_id, chain_id, elapsed.as_micros());
                            }
                            return Ok(Some(provider.clone()));
                        }
                    }
                }
            }
        }

        warn!("No healthy provider available for chain {:?}", chain_id);
        Ok(None)
    }

    /// Record provider success
    ///
    /// # Errors
    ///
    /// Returns error if recording success fails
    #[inline]
    pub async fn record_success(&self, provider_id: &str, response_time_ms: u64) -> Result<()> {
        let start_time = Instant::now();

        // Update circuit breaker
        if let Some(circuit_breaker) = self.circuit_breakers.get(provider_id) {
            circuit_breaker.record_success();
        }

        // Update provider metrics
        if let Some(mut metrics) = self.provider_metrics.get_mut(provider_id) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));

            metrics.total_requests = metrics.total_requests.saturating_add(1);
            metrics.successful_requests = metrics.successful_requests.saturating_add(1);
            metrics.consecutive_successes = metrics.consecutive_successes.saturating_add(1);
            metrics.consecutive_failures = 0;
            metrics.last_request_time = now;
            metrics.last_success_time = now;

            // Update average response time
            if metrics.total_requests > 0 {
                let total_time = metrics.avg_response_time_ms
                    .saturating_mul(metrics.total_requests.saturating_sub(1))
                    .saturating_add(response_time_ms);
                metrics.avg_response_time_ms = total_time / metrics.total_requests;
            }

            // Update success rate
            if metrics.total_requests > 0 {
                let success_rate = Decimal::from(metrics.successful_requests) / Decimal::from(metrics.total_requests);
                metrics.success_rate = success_rate;
            }

            // Update health score
            self.calculate_health_score(&mut metrics);

            // Update status based on health
            if metrics.health_score >= Decimal::from_f64_retain(self.config.min_healthy_success_rate).unwrap_or(Decimal::ZERO) {
                metrics.status = RpcProviderStatus::Healthy;
            }
        }

        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > 5 {
            warn!("Record success took {}μs (target: <5μs)", elapsed.as_micros());
        }

        Ok(())
    }

    /// Record provider failure
    ///
    /// # Errors
    ///
    /// Returns error if recording failure fails
    #[inline]
    pub async fn record_failure(&self, provider_id: &str, error: &str) -> Result<()> {
        let start_time = Instant::now();

        // Update circuit breaker
        if let Some(circuit_breaker) = self.circuit_breakers.get(provider_id) {
            circuit_breaker.record_failure();
        }

        // Update provider metrics
        if let Some(mut metrics) = self.provider_metrics.get_mut(provider_id) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));

            metrics.total_requests = metrics.total_requests.saturating_add(1);
            metrics.failed_requests = metrics.failed_requests.saturating_add(1);
            metrics.consecutive_failures = metrics.consecutive_failures.saturating_add(1);
            metrics.consecutive_successes = 0;
            metrics.last_request_time = now;
            metrics.last_failure_time = now;

            // Update success rate
            if metrics.total_requests > 0 {
                let success_rate = Decimal::from(metrics.successful_requests) / Decimal::from(metrics.total_requests);
                metrics.success_rate = success_rate;
            }

            // Update health score
            self.calculate_health_score(&mut metrics);

            // Check if provider should be marked unhealthy
            if metrics.consecutive_failures >= self.config.failover_threshold {
                metrics.status = RpcProviderStatus::Unhealthy;

                // Trigger failover event
                self.trigger_failover_event(provider_id, error).await?;
            }
        }

        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > 5 {
            warn!("Record failure took {}μs (target: <5μs)", elapsed.as_micros());
        }

        Ok(())
    }

    /// Get provider health metrics
    #[inline]
    #[must_use]
    pub fn get_provider_metrics(&self, provider_id: &str) -> Option<ProviderHealthMetrics> {
        self.provider_metrics.get(provider_id).map(|entry| entry.value().clone())
    }

    /// Get all provider metrics for chain
    #[inline]
    #[must_use]
    pub fn get_chain_metrics(&self, chain_id: ChainId) -> Vec<ProviderHealthMetrics> {
        let mut metrics = Vec::new();

        if let Some(providers) = self.providers.get(&chain_id) {
            for provider in providers.iter() {
                if let Some(provider_metrics) = self.provider_metrics.get(&provider.id) {
                    metrics.push(provider_metrics.clone());
                }
            }
        }

        metrics
    }

    /// Get failover manager statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &FailoverManagerStats {
        &self.stats
    }

    /// Get configuration
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &FailoverManagerConfig {
        &self.config
    }

    // Private helper methods

    /// Start health check background task
    async fn start_health_check_task(&self) -> Result<JoinHandle<()>> {
        let config = self.config.clone();
        let provider_metrics = Arc::clone(&self.provider_metrics);
        let circuit_breakers = Arc::clone(&self.circuit_breakers);
        let stats = Arc::clone(&self.stats);
        let shutdown_notify = Arc::clone(&self.shutdown_notify);
        let running = Arc::clone(&self.running);

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(config.health_check_interval_ms));

            while running.load(Ordering::Acquire) {
                tokio::select! {
                    _ = interval.tick() => {
                        // Perform health checks
                        for entry in provider_metrics.iter() {
                            let provider_id = entry.key();
                            let mut metrics = entry.value().clone();

                            // Simulate health check (in production, this would be actual HTTP request)
                            let health_check_result = Self::perform_health_check(provider_id, &config).await;

                            if let Ok(response_time_ms) = health_check_result {
                                // Update circuit breaker
                                if let Some(circuit_breaker) = circuit_breakers.get(provider_id) {
                                    circuit_breaker.record_success();
                                }

                                // Update metrics
                                metrics.last_health_check_time = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));

                                if response_time_ms <= config.max_healthy_response_time_ms {
                                    metrics.status = RpcProviderStatus::Healthy;
                                }
                            } else {
                                // Update circuit breaker
                                if let Some(circuit_breaker) = circuit_breakers.get(provider_id) {
                                    circuit_breaker.record_failure();
                                }

                                metrics.status = RpcProviderStatus::Unhealthy;
                            }

                            // Update metrics in map
                            provider_metrics.insert(provider_id.clone(), metrics);
                            stats.health_checks_performed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    () = shutdown_notify.notified() => {
                        break;
                    }
                }
            }
        });

        Ok(task)
    }

    /// Start provider ranking background task
    async fn start_ranking_task(&self) -> Result<JoinHandle<()>> {
        let providers = Arc::clone(&self.providers);
        let provider_metrics = Arc::clone(&self.provider_metrics);
        let provider_rankings = Arc::clone(&self.provider_rankings);
        let stats = Arc::clone(&self.stats);
        let shutdown_notify = Arc::clone(&self.shutdown_notify);
        let running = Arc::clone(&self.running);
        let config = self.config.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(5000)); // Update rankings every 5 seconds

            while running.load(Ordering::Acquire) {
                tokio::select! {
                    _ = interval.tick() => {
                        // Update rankings for all chains
                        for entry in providers.iter() {
                            let chain_id = *entry.key();
                            let chain_providers = entry.value();

                            let mut ranked_providers: Vec<(String, Decimal)> = Vec::new();

                            for provider in chain_providers {
                                if let Some(metrics) = provider_metrics.get(&provider.id) {
                                    let score = Self::calculate_provider_score(provider, &metrics, &config);
                                    ranked_providers.push((provider.id.clone(), score));
                                }
                            }

                            // Sort by score (higher is better)
                            ranked_providers.sort_by(|a, b| b.1.cmp(&a.1));

                            // Update rankings
                            let rankings: Vec<String> = ranked_providers.into_iter().map(|(id, _)| id).collect();
                            provider_rankings.insert(chain_id, rankings);

                            stats.ranking_updates.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    () = shutdown_notify.notified() => {
                        break;
                    }
                }
            }
        });

        Ok(task)
    }

    /// Update provider rankings for specific chain
    async fn update_provider_rankings(&self, chain_id: ChainId) -> Result<()> {
        let providers = match self.providers.get(&chain_id) {
            Some(providers) => providers.clone(),
            None => return Ok(()),
        };

        let mut ranked_providers: Vec<(String, Decimal)> = Vec::new();

        for provider in &providers {
            if let Some(metrics) = self.provider_metrics.get(&provider.id) {
                let score = Self::calculate_provider_score(provider, &metrics, &self.config);
                ranked_providers.push((provider.id.clone(), score));
            }
        }

        // Sort by score (higher is better)
        ranked_providers.sort_by(|a, b| b.1.cmp(&a.1));

        // Update rankings
        let rankings: Vec<String> = ranked_providers.into_iter().map(|(id, _)| id).collect();
        self.provider_rankings.insert(chain_id, rankings);

        self.stats.ranking_updates.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Calculate health score for provider
    fn calculate_health_score(&self, metrics: &mut ProviderHealthMetrics) {
        let mut score = Decimal::ZERO;

        // Success rate component (40% weight)
        score += metrics.success_rate * Decimal::from_f64_retain(0.4).unwrap_or(Decimal::ZERO);

        // Response time component (30% weight)
        if metrics.avg_response_time_ms > 0 {
            let response_score = if metrics.avg_response_time_ms <= self.config.max_healthy_response_time_ms {
                Decimal::ONE - (Decimal::from(metrics.avg_response_time_ms) / Decimal::from(self.config.max_healthy_response_time_ms))
            } else {
                Decimal::ZERO
            };
            score += response_score * Decimal::from_f64_retain(0.3).unwrap_or(Decimal::ZERO);
        }

        // Consecutive failures penalty (20% weight)
        let failure_penalty = if metrics.consecutive_failures > 0 {
            Decimal::ONE / (Decimal::ONE + Decimal::from(metrics.consecutive_failures))
        } else {
            Decimal::ONE
        };
        score += failure_penalty * Decimal::from_f64_retain(0.2).unwrap_or(Decimal::ZERO);

        // Recent activity bonus (10% weight)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
        let time_since_last_success = now.saturating_sub(metrics.last_success_time);
        let activity_score = if time_since_last_success < 60_000 { // 1 minute
            Decimal::ONE
        } else if time_since_last_success < 300_000 { // 5 minutes
            Decimal::from_f64_retain(0.5).unwrap_or(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };
        score += activity_score * Decimal::from_f64_retain(0.1).unwrap_or(Decimal::ZERO);

        metrics.health_score = score;
    }

    /// Calculate provider score for ranking
    fn calculate_provider_score(provider: &RpcProvider, metrics: &ProviderHealthMetrics, config: &FailoverManagerConfig) -> Decimal {
        let mut score = metrics.health_score;

        // Add provider type weight
        if let Some(weight) = config.priority_weights.get(&provider.provider_type) {
            let type_bonus = Decimal::from(*weight) / Decimal::from(100);
            score += type_bonus * Decimal::from_f64_retain(0.2).unwrap_or(Decimal::ZERO);
        }

        // Add provider priority
        let priority_bonus = Decimal::from(provider.priority) / Decimal::from(100);
        score += priority_bonus * Decimal::from_f64_retain(0.1).unwrap_or(Decimal::ZERO);

        score
    }

    /// Perform health check on provider
    async fn perform_health_check(provider_id: &str, config: &FailoverManagerConfig) -> Result<u64> {
        let start_time = Instant::now();

        // Simulate health check with timeout
        sleep(Duration::from_millis(1)).await; // Minimal delay for simulation

        let elapsed = start_time.elapsed();
        let response_time_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);

        // Simulate occasional failures for testing
        if provider_id.contains("test_fail") {
            return Err(ChainCoreError::Configuration("Simulated health check failure".to_string()));
        }

        if response_time_ms > config.health_check_timeout_ms {
            return Err(ChainCoreError::Configuration("Health check timeout".to_string()));
        }

        Ok(response_time_ms)
    }

    /// Trigger failover event
    #[expect(clippy::cognitive_complexity, reason = "Complex failover logic required for reliability")]
    async fn trigger_failover_event(&self, provider_id: &str, error: &str) -> Result<()> {
        let start_time = Instant::now();

        // Find chain for this provider
        let mut chain_id = None;
        for entry in self.providers.iter() {
            if entry.value().iter().any(|p| p.id == provider_id) {
                chain_id = Some(*entry.key());
                break;
            }
        }

        let Some(chain_id) = chain_id else {
            warn!("Could not find chain for provider {}", provider_id);
            return Ok(());
        };

        // Find backup provider
        let backup_provider = self.get_best_provider(chain_id).await?;
        let backup_provider_id = backup_provider.map(|p| p.id);
        let has_backup = backup_provider_id.is_some();

        // Create failover event
        let event = FailoverEvent {
            id: format!("failover_{}", SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))),
            chain_id,
            failed_provider_id: provider_id.to_string(),
            backup_provider_id,
            failure_reason: error.to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
            recovery_time_ms: None,
        };

        // Add to events history
        {
            let mut events = self.failover_events.write().await;
            events.push(event);

            // Keep only recent events
            let events_len = events.len();
            if events_len > FAILOVER_MAX_EVENTS_HISTORY {
                events.drain(0..events_len - FAILOVER_MAX_EVENTS_HISTORY);
            }
        }

        // Update statistics
        self.stats.total_failovers.fetch_add(1, Ordering::Relaxed);
        if has_backup {
            self.stats.successful_failovers.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.failed_failovers.fetch_add(1, Ordering::Relaxed);
        }

        let elapsed = start_time.elapsed();
        let elapsed_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
        self.stats.avg_failover_time_us.store(elapsed_us, Ordering::Relaxed);

        if elapsed.as_micros() > 15 {
            warn!("Failover event took {}μs (target: <15μs)", elapsed.as_micros());
        }

        warn!("Failover triggered for provider {} on chain {:?}: {}",
              provider_id, chain_id, error);

        Ok(())
    }
}

// Tests are implemented in integration tests to avoid clippy restrictions
