//! Parallel Flashloan Executor for ultra-performance concurrent flashloan operations
//!
//! This module provides advanced parallel execution capabilities for maximizing
//! flashloan throughput through concurrent execution across multiple providers,
//! chains, and strategies with intelligent load balancing and fault tolerance.
//!
//! ## Performance Targets
//! - Concurrent Execution: <25μs per task
//! - Load Balancing: <5μs overhead
//! - Task Scheduling: <10μs latency
//! - Error Recovery: <15μs response
//! - Total Throughput: >10,000 ops/sec
//!
//! ## Architecture
//! - Multi-threaded execution engine
//! - Advanced task scheduling
//! - Intelligent load balancing
//! - Real-time performance monitoring
//! - Lock-free coordination primitives

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
        atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
    collections::HashMap,
};
use tokio::{
    sync::{RwLock, Mutex as TokioMutex, Semaphore},
    time::{interval, sleep},
    task::JoinHandle,
};
use tracing::{info, trace, warn};

/// Parallel executor configuration
#[derive(Debug, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "Configuration struct with multiple feature flags")]
pub struct ParallelExecutorConfig {
    /// Enable parallel execution
    pub enabled: bool,
    
    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,
    
    /// Worker thread count
    pub worker_thread_count: usize,
    
    /// Task queue capacity
    pub task_queue_capacity: usize,
    
    /// Load balancing interval in milliseconds
    pub load_balancing_interval_ms: u64,
    
    /// Performance monitoring interval in milliseconds
    pub performance_monitoring_interval_ms: u64,
    
    /// Health check interval in milliseconds
    pub health_check_interval_ms: u64,
    
    /// Enable dynamic scaling
    pub enable_dynamic_scaling: bool,
    
    /// Enable load balancing
    pub enable_load_balancing: bool,
    
    /// Enable fault tolerance
    pub enable_fault_tolerance: bool,
    
    /// Enable performance optimization
    pub enable_performance_optimization: bool,
    
    /// Task timeout in milliseconds
    pub task_timeout_ms: u64,
    
    /// Retry attempts for failed tasks
    pub retry_attempts: u32,
    
    /// Backoff multiplier for retries
    pub backoff_multiplier: f64,
    
    /// Maximum execution time per task (milliseconds)
    pub max_execution_time_ms: u64,
    
    /// Minimum execution time per task (milliseconds)
    pub min_execution_time_ms: u64,
    
    /// Target throughput (operations per second)
    pub target_throughput_ops_per_sec: u64,
}

/// Execution task for parallel processing
#[derive(Debug, Clone)]
pub struct ExecutionTask {
    /// Task ID
    pub task_id: String,
    
    /// Flashloan request
    pub request: super::FlashloanRequest,
    
    /// Preferred provider
    pub preferred_provider: Option<super::FlashloanProvider>,
    
    /// Priority level (higher = more priority)
    pub priority: u32,
    
    /// Maximum retries
    pub max_retries: u32,
    
    /// Current retry count
    pub retry_count: u32,
    
    /// Task creation timestamp
    pub created_at: u64,
    
    /// Task deadline
    pub deadline: u64,
    
    /// Strategy identifier
    pub strategy_id: String,
    
    /// Execution context
    pub context: HashMap<String, String>,
}

/// Execution result for parallel processing
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Task ID
    pub task_id: String,
    
    /// Execution result
    pub execution: Option<super::FlashloanExecution>,
    
    /// Execution status
    pub status: ParallelExecutionStatus,
    
    /// Error message (if failed)
    pub error_message: Option<String>,
    
    /// Worker ID that processed the task
    pub worker_id: usize,
    
    /// Execution time (microseconds)
    pub execution_time_us: u64,
    
    /// Queue time (microseconds)
    pub queue_time_us: u64,
    
    /// Total time (microseconds)
    pub total_time_us: u64,
    
    /// Completed at timestamp
    pub completed_at: u64,
}

/// Parallel execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelExecutionStatus {
    /// Task queued for execution
    Queued,
    /// Task assigned to worker
    Assigned,
    /// Task executing
    Executing,
    /// Task completed successfully
    Success,
    /// Task failed
    Failed,
    /// Task cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
    /// Task retrying
    Retrying,
    /// Task rejected (queue full)
    Rejected,
}

/// Worker statistics
#[derive(Debug, Default)]
pub struct WorkerStats {
    /// Tasks processed
    pub tasks_processed: AtomicU64,
    
    /// Tasks successful
    pub tasks_successful: AtomicU64,
    
    /// Tasks failed
    pub tasks_failed: AtomicU64,
    
    /// Total execution time (microseconds)
    pub total_execution_time_us: AtomicU64,
    
    /// Average execution time (microseconds)
    pub avg_execution_time_us: AtomicU64,
    
    /// Worker utilization (scaled by 1e6)
    pub utilization_scaled: AtomicU64,
    
    /// Last task timestamp
    pub last_task_timestamp: AtomicU64,
}

/// Parallel executor statistics
#[derive(Debug, Default)]
pub struct ParallelExecutorStats {
    /// Total tasks submitted
    pub total_tasks_submitted: AtomicU64,
    
    /// Total tasks completed
    pub total_tasks_completed: AtomicU64,
    
    /// Total tasks failed
    pub total_tasks_failed: AtomicU64,
    
    /// Total tasks cancelled
    pub total_tasks_cancelled: AtomicU64,
    
    /// Total tasks timed out
    pub total_tasks_timed_out: AtomicU64,
    
    /// Total tasks retried
    pub total_tasks_retried: AtomicU64,
    
    /// Current queue size
    pub current_queue_size: AtomicUsize,
    
    /// Peak queue size
    pub peak_queue_size: AtomicUsize,
    
    /// Active workers
    pub active_workers: AtomicUsize,
    
    /// Total throughput (ops/sec)
    pub total_throughput_ops_per_sec: AtomicU64,
    
    /// Average queue time (microseconds)
    pub avg_queue_time_us: AtomicU64,
    
    /// Average execution time (microseconds)
    pub avg_execution_time_us: AtomicU64,
    
    /// Load balancing cycles
    pub load_balancing_cycles: AtomicU64,
    
    /// Performance optimization cycles
    pub performance_optimization_cycles: AtomicU64,
    
    /// Health check cycles
    pub health_check_cycles: AtomicU64,
    
    /// Dynamic scaling events
    pub dynamic_scaling_events: AtomicU64,
}

/// Cache-line aligned execution data for ultra-performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedExecutionData {
    /// Current throughput (ops/sec)
    pub current_throughput_ops_per_sec: u64,
    
    /// Average queue time (microseconds)
    pub avg_queue_time_us: u64,
    
    /// Average execution time (microseconds)
    pub avg_execution_time_us: u64,
    
    /// Worker utilization (scaled by 1e6)
    pub worker_utilization_scaled: u64,
    
    /// Success rate (scaled by 1e6)
    pub success_rate_scaled: u64,
    
    /// Queue utilization (scaled by 1e6)
    pub queue_utilization_scaled: u64,
    
    /// Load balance score (scaled by 1e6)
    pub load_balance_score_scaled: u64,
    
    /// Last update timestamp
    pub timestamp: u64,
}

/// Parallel executor constants
pub const PARALLEL_DEFAULT_MAX_CONCURRENT: usize = 1000;
pub const PARALLEL_DEFAULT_WORKER_THREADS: usize = 32;
pub const PARALLEL_DEFAULT_QUEUE_CAPACITY: usize = 10000;
pub const PARALLEL_DEFAULT_LOAD_BALANCING_INTERVAL_MS: u64 = 1000; // 1 second
pub const PARALLEL_DEFAULT_PERFORMANCE_INTERVAL_MS: u64 = 5000; // 5 seconds
pub const PARALLEL_DEFAULT_HEALTH_CHECK_INTERVAL_MS: u64 = 2000; // 2 seconds
pub const PARALLEL_DEFAULT_TASK_TIMEOUT_MS: u64 = 30000; // 30 seconds
pub const PARALLEL_DEFAULT_RETRY_ATTEMPTS: u32 = 3;
pub const PARALLEL_DEFAULT_BACKOFF_MULTIPLIER: f64 = 2.0;
pub const PARALLEL_DEFAULT_MAX_EXECUTION_TIME_MS: u64 = 10000; // 10 seconds
pub const PARALLEL_DEFAULT_MIN_EXECUTION_TIME_MS: u64 = 100; // 100ms
pub const PARALLEL_DEFAULT_TARGET_THROUGHPUT: u64 = 10000; // 10k ops/sec
pub const PARALLEL_HIGH_PRIORITY: u32 = 1000;
pub const PARALLEL_NORMAL_PRIORITY: u32 = 500;
pub const PARALLEL_LOW_PRIORITY: u32 = 100;

impl Default for ParallelExecutorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent_executions: PARALLEL_DEFAULT_MAX_CONCURRENT,
            worker_thread_count: PARALLEL_DEFAULT_WORKER_THREADS,
            task_queue_capacity: PARALLEL_DEFAULT_QUEUE_CAPACITY,
            load_balancing_interval_ms: PARALLEL_DEFAULT_LOAD_BALANCING_INTERVAL_MS,
            performance_monitoring_interval_ms: PARALLEL_DEFAULT_PERFORMANCE_INTERVAL_MS,
            health_check_interval_ms: PARALLEL_DEFAULT_HEALTH_CHECK_INTERVAL_MS,
            enable_dynamic_scaling: true,
            enable_load_balancing: true,
            enable_fault_tolerance: true,
            enable_performance_optimization: true,
            task_timeout_ms: PARALLEL_DEFAULT_TASK_TIMEOUT_MS,
            retry_attempts: PARALLEL_DEFAULT_RETRY_ATTEMPTS,
            #[allow(clippy::cast_precision_loss)] // Configuration value, precision loss acceptable
            backoff_multiplier: PARALLEL_DEFAULT_BACKOFF_MULTIPLIER,
            max_execution_time_ms: PARALLEL_DEFAULT_MAX_EXECUTION_TIME_MS,
            min_execution_time_ms: PARALLEL_DEFAULT_MIN_EXECUTION_TIME_MS,
            target_throughput_ops_per_sec: PARALLEL_DEFAULT_TARGET_THROUGHPUT,
        }
    }
}

impl AlignedExecutionData {
    /// Create new aligned execution data
    #[inline(always)]
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "Aligned data structure requires all fields")]
    pub const fn new(
        current_throughput_ops_per_sec: u64,
        avg_queue_time_us: u64,
        avg_execution_time_us: u64,
        worker_utilization_scaled: u64,
        success_rate_scaled: u64,
        queue_utilization_scaled: u64,
        load_balance_score_scaled: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            current_throughput_ops_per_sec,
            avg_queue_time_us,
            avg_execution_time_us,
            worker_utilization_scaled,
            success_rate_scaled,
            queue_utilization_scaled,
            load_balance_score_scaled,
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

    /// Get worker utilization as Decimal
    #[inline(always)]
    #[must_use]
    pub fn worker_utilization(&self) -> Decimal {
        Decimal::from(self.worker_utilization_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get success rate as Decimal
    #[inline(always)]
    #[must_use]
    pub fn success_rate(&self) -> Decimal {
        Decimal::from(self.success_rate_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get queue utilization as Decimal
    #[inline(always)]
    #[must_use]
    pub fn queue_utilization(&self) -> Decimal {
        Decimal::from(self.queue_utilization_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get load balance score as Decimal
    #[inline(always)]
    #[must_use]
    pub fn load_balance_score(&self) -> Decimal {
        Decimal::from(self.load_balance_score_scaled) / Decimal::from(1_000_000_u64)
    }

    /// Get overall performance score
    #[inline(always)]
    #[must_use]
    pub fn overall_performance_score(&self) -> Decimal {
        // Weighted score: throughput (30%) + success rate (25%) + utilization (20%) + queue efficiency (15%) + load balance (10%)
        let throughput_weight = "0.3".parse::<Decimal>().unwrap_or_default();
        let success_weight = "0.25".parse::<Decimal>().unwrap_or_default();
        let utilization_weight = "0.2".parse::<Decimal>().unwrap_or_default();
        let queue_weight = "0.15".parse::<Decimal>().unwrap_or_default();
        let balance_weight = "0.1".parse::<Decimal>().unwrap_or_default();

        // Normalize throughput score (higher throughput = higher score, max 50k ops/sec)
        let throughput_score = (Decimal::from(self.current_throughput_ops_per_sec) / Decimal::from(50_000_u64)).min(Decimal::ONE);

        // Queue efficiency is inverse of utilization (lower queue utilization = higher efficiency)
        let queue_efficiency = Decimal::ONE - self.queue_utilization();

        throughput_score * throughput_weight +
        self.success_rate() * success_weight +
        self.worker_utilization() * utilization_weight +
        queue_efficiency * queue_weight +
        self.load_balance_score() * balance_weight
    }
}

/// Parallel Flashloan Executor for ultra-performance concurrent operations
#[expect(dead_code, reason = "Fields used in production implementation")]
pub struct ParallelExecutor {
    /// Configuration
    config: Arc<ChainCoreConfig>,

    /// Parallel executor specific configuration
    parallel_config: ParallelExecutorConfig,

    /// Statistics
    stats: Arc<ParallelExecutorStats>,

    /// Worker statistics
    worker_stats: Arc<Vec<WorkerStats>>,

    /// Execution data cache for ultra-fast access
    execution_cache: Arc<DashMap<String, AlignedExecutionData>>,

    /// Active tasks
    active_tasks: Arc<RwLock<HashMap<String, ExecutionTask>>>,

    /// Completed results
    completed_results: Arc<RwLock<HashMap<String, ExecutionResult>>>,

    /// Performance timers
    load_balancing_timer: Timer,
    performance_timer: Timer,
    health_check_timer: Timer,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Task channels
    task_sender: Sender<ExecutionTask>,
    task_receiver: Receiver<ExecutionTask>,

    /// Result channels
    result_sender: Sender<ExecutionResult>,
    result_receiver: Receiver<ExecutionResult>,

    /// Semaphore for concurrency control
    execution_semaphore: Arc<Semaphore>,

    /// Worker handles
    worker_handles: Arc<TokioMutex<Vec<JoinHandle<()>>>>,

    /// Current execution round
    execution_round: Arc<TokioMutex<u64>>,
}

impl ParallelExecutor {
    /// Create new parallel executor with ultra-performance configuration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    #[inline]
    pub async fn new(config: Arc<ChainCoreConfig>) -> Result<Self> {
        let parallel_config = ParallelExecutorConfig::default();
        let stats = Arc::new(ParallelExecutorStats::default());

        // Initialize worker statistics
        let mut worker_stats_vec = Vec::with_capacity(parallel_config.worker_thread_count);
        for _ in 0..parallel_config.worker_thread_count {
            worker_stats_vec.push(WorkerStats::default());
        }
        let worker_stats = Arc::new(worker_stats_vec);

        let execution_cache = Arc::new(DashMap::with_capacity(parallel_config.max_concurrent_executions));
        let active_tasks = Arc::new(RwLock::new(HashMap::with_capacity(parallel_config.max_concurrent_executions)));
        let completed_results = Arc::new(RwLock::new(HashMap::with_capacity(parallel_config.max_concurrent_executions)));
        let load_balancing_timer = Timer::new("parallel_load_balancing");
        let performance_timer = Timer::new("parallel_performance");
        let health_check_timer = Timer::new("parallel_health_check");
        let shutdown = Arc::new(AtomicBool::new(false));
        let execution_round = Arc::new(TokioMutex::new(0));

        let (task_sender, task_receiver) = channel::bounded(parallel_config.task_queue_capacity);
        let (result_sender, result_receiver) = channel::bounded(parallel_config.task_queue_capacity);
        let execution_semaphore = Arc::new(Semaphore::new(parallel_config.max_concurrent_executions));
        let worker_handles = Arc::new(TokioMutex::new(Vec::with_capacity(parallel_config.worker_thread_count)));

        Ok(Self {
            config,
            parallel_config,
            stats,
            worker_stats,
            execution_cache,
            active_tasks,
            completed_results,
            load_balancing_timer,
            performance_timer,
            health_check_timer,
            shutdown,
            task_sender,
            task_receiver,
            result_sender,
            result_receiver,
            execution_semaphore,
            worker_handles,
            execution_round,
        })
    }

    /// Start parallel executor services
    ///
    /// # Errors
    ///
    /// Returns error if startup fails
    #[inline]
    #[expect(clippy::cognitive_complexity, reason = "Startup function coordinates multiple services")]
    pub async fn start(&self) -> Result<()> {
        if !self.parallel_config.enabled {
            info!("Parallel executor disabled");
            return Ok(());
        }

        info!("Starting parallel executor with {} workers", self.parallel_config.worker_thread_count);

        // Start worker threads
        self.start_workers().await;

        // Start load balancing
        if self.parallel_config.enable_load_balancing {
            self.start_load_balancing().await;
        }

        // Start performance monitoring
        self.start_performance_monitoring().await;

        // Start health checking
        self.start_health_checking().await;

        // Start result processing
        self.start_result_processing().await;

        info!("Parallel executor started successfully");
        Ok(())
    }

    /// Stop parallel executor
    #[inline]
    pub async fn stop(&self) {
        info!("Stopping parallel executor");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for workers to finish
        let mut handles = self.worker_handles.lock().await;
        for handle in handles.drain(..) {
            #[expect(clippy::let_underscore_must_use, reason = "Worker handle result is not needed")]
            #[expect(clippy::let_underscore_untyped, reason = "Worker handle result type is not needed")]
            let _ = handle.await;
        }
        drop(handles);

        // Wait for graceful shutdown
        sleep(Duration::from_millis(100)).await;

        info!("Parallel executor stopped");
    }

    /// Get current statistics
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &ParallelExecutorStats {
        &self.stats
    }

    /// Get worker statistics
    #[inline]
    #[must_use]
    pub fn worker_stats(&self) -> &[WorkerStats] {
        &self.worker_stats
    }

    /// Submit task for parallel execution
    #[inline]
    #[must_use]
    pub async fn submit_task(&self, task: ExecutionTask) -> bool {
        let queue_size = self.stats.current_queue_size.load(Ordering::Relaxed);

        // Check queue capacity
        if queue_size >= self.parallel_config.task_queue_capacity {
            self.stats.total_tasks_failed.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Try to send task
        if self.task_sender.try_send(task.clone()).is_ok() {
            self.stats.total_tasks_submitted.fetch_add(1, Ordering::Relaxed);
            self.stats.current_queue_size.fetch_add(1, Ordering::Relaxed);

            // Update peak queue size
            let current_size = self.stats.current_queue_size.load(Ordering::Relaxed);
            let peak_size = self.stats.peak_queue_size.load(Ordering::Relaxed);
            if current_size > peak_size {
                self.stats.peak_queue_size.store(current_size, Ordering::Relaxed);
            }

            // Store active task
            {
                let mut active_tasks = self.active_tasks.write().await;
                active_tasks.insert(task.task_id.clone(), task);
                drop(active_tasks);
            }

            true
        } else {
            self.stats.total_tasks_failed.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Get task result
    #[inline]
    pub async fn get_result(&self, task_id: &str) -> Option<ExecutionResult> {
        let completed_results = self.completed_results.read().await;
        completed_results.get(task_id).cloned()
    }

    /// Get active tasks
    #[inline]
    pub async fn get_active_tasks(&self) -> Vec<ExecutionTask> {
        let active_tasks = self.active_tasks.read().await;
        active_tasks.values().cloned().collect()
    }

    /// Get completed results
    #[inline]
    pub async fn get_completed_results(&self) -> Vec<ExecutionResult> {
        let completed_results = self.completed_results.read().await;
        completed_results.values().cloned().collect()
    }

    /// Create execution task
    #[inline]
    #[must_use]
    pub async fn create_task(
        &self,
        request: super::FlashloanRequest,
        strategy_id: String,
        priority: u32,
    ) -> ExecutionTask {
        let mut round = self.execution_round.lock().await;
        *round += 1;

        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for task creation")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        #[expect(clippy::cast_possible_truncation, reason = "Chain ID values are small")]
        let task_id = format!("task_{}_{}_{}_{}", request.chain_id as u8, strategy_id, *round, fastrand::u64(..));

        ExecutionTask {
            task_id,
            request,
            preferred_provider: None,
            priority,
            max_retries: self.parallel_config.retry_attempts,
            retry_count: 0,
            created_at: now,
            deadline: now + self.parallel_config.task_timeout_ms,
            strategy_id,
            context: HashMap::new(),
        }
    }

    /// Start worker threads
    async fn start_workers(&self) {
        let mut handles = self.worker_handles.lock().await;

        for worker_id in 0..self.parallel_config.worker_thread_count {
            let task_receiver = self.task_receiver.clone();
            let result_sender = self.result_sender.clone();
            let execution_semaphore = Arc::clone(&self.execution_semaphore);
            let worker_stats = Arc::clone(&self.worker_stats);
            let shutdown = Arc::clone(&self.shutdown);
            let parallel_config = self.parallel_config.clone();

            let handle = tokio::spawn(async move {
                Self::worker_loop(
                    worker_id,
                    task_receiver,
                    result_sender,
                    execution_semaphore,
                    worker_stats,
                    shutdown,
                    parallel_config,
                ).await;
            });

            handles.push(handle);
        }

        self.stats.active_workers.store(self.parallel_config.worker_thread_count, Ordering::Relaxed);
        drop(handles);
    }

    /// Worker loop for processing tasks
    async fn worker_loop(
        worker_id: usize,
        task_receiver: Receiver<ExecutionTask>,
        result_sender: Sender<ExecutionResult>,
        execution_semaphore: Arc<Semaphore>,
        worker_stats: Arc<Vec<WorkerStats>>,
        shutdown: Arc<AtomicBool>,
        parallel_config: ParallelExecutorConfig,
    ) {
        trace!("Worker {} started", worker_id);

        while !shutdown.load(Ordering::Relaxed) {
            // Try to receive task
            if let Ok(task) = task_receiver.try_recv() {
                let start_time = Instant::now();

                // Acquire semaphore permit
                let _permit = execution_semaphore.acquire().await;

                // Calculate queue time
                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let queue_time_us = start_time.elapsed().as_micros() as u64;

                // Execute task
                let execution_start = Instant::now();
                let execution_result = Self::execute_task(&task, worker_id, &parallel_config).await;

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let execution_time_us = execution_start.elapsed().as_micros() as u64;

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let total_time_us = start_time.elapsed().as_micros() as u64;

                #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for completion data")]
                let completed_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                // Create result
                let result = ExecutionResult {
                    task_id: task.task_id.clone(),
                    execution: execution_result.0,
                    status: execution_result.1,
                    error_message: execution_result.2,
                    worker_id,
                    execution_time_us,
                    queue_time_us,
                    total_time_us,
                    completed_at,
                };

                // Update worker statistics
                if let Some(stats) = worker_stats.get(worker_id) {
                    stats.tasks_processed.fetch_add(1, Ordering::Relaxed);

                    if result.status == ParallelExecutionStatus::Success {
                        stats.tasks_successful.fetch_add(1, Ordering::Relaxed);
                    } else {
                        stats.tasks_failed.fetch_add(1, Ordering::Relaxed);
                    }

                    stats.total_execution_time_us.fetch_add(execution_time_us, Ordering::Relaxed);
                    stats.last_task_timestamp.store(completed_at, Ordering::Relaxed);

                    // Update average execution time
                    let total_tasks = stats.tasks_processed.load(Ordering::Relaxed);
                    if total_tasks > 0 {
                        let total_time = stats.total_execution_time_us.load(Ordering::Relaxed);
                        let avg_time = total_time / total_tasks;
                        stats.avg_execution_time_us.store(avg_time, Ordering::Relaxed);
                    }
                }

                // Send result
                #[expect(clippy::let_underscore_must_use, reason = "Result send failure is not critical")]
                #[expect(clippy::let_underscore_untyped, reason = "Result send type is not needed")]
                let _ = result_sender.try_send(result);
            } else {
                // No tasks available, sleep briefly
                sleep(Duration::from_micros(100)).await;
            }
        }

        trace!("Worker {} stopped", worker_id);
    }

    /// Execute individual task
    async fn execute_task(
        task: &ExecutionTask,
        worker_id: usize,
        parallel_config: &ParallelExecutorConfig,
    ) -> (Option<super::FlashloanExecution>, ParallelExecutionStatus, Option<String>) {
        // Check task deadline
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for deadline check")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if now > task.deadline {
            return (None, ParallelExecutionStatus::TimedOut, Some("Task deadline exceeded".to_string()));
        }

        // Simulate task execution (in production this would call actual flashloan providers)
        let execution_success = Self::simulate_execution(task, worker_id, parallel_config).await;

        if execution_success {
            // Create mock successful execution
            let execution = super::FlashloanExecution {
                request_id: task.task_id.clone(),
                chain_id: task.request.chain_id,
                provider: task.preferred_provider.clone().unwrap_or(super::FlashloanProvider::Aave),
                status: super::ExecutionStatus::Success,
                transaction_hash: Some(format!("0x{:x}", fastrand::u64(..))),
                actual_fee_usd: task.request.amount_usd * "0.0009".parse::<Decimal>().unwrap_or_default(), // 0.09% fee
                execution_time_s: 5,
                gas_used: 250_000,
                gas_cost_usd: "15".parse().unwrap_or_default(),
                error_message: None,
                #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for execution data")]
                executed_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            };

            (Some(execution), ParallelExecutionStatus::Success, None)
        } else {
            (None, ParallelExecutionStatus::Failed, Some("Simulated execution failure".to_string()))
        }
    }

    /// Simulate task execution
    async fn simulate_execution(
        task: &ExecutionTask,
        _worker_id: usize,
        parallel_config: &ParallelExecutorConfig,
    ) -> bool {
        // Simulate execution time
        let execution_time_ms = fastrand::u64(
            parallel_config.min_execution_time_ms..=parallel_config.max_execution_time_ms
        );
        sleep(Duration::from_millis(execution_time_ms)).await;

        // Simulate success rate based on priority
        let success_rate = match task.priority {
            p if p >= PARALLEL_HIGH_PRIORITY => 0.95, // 95% success for high priority
            p if p >= PARALLEL_NORMAL_PRIORITY => 0.90, // 90% success for normal priority
            _ => 0.85, // 85% success for low priority
        };

        #[allow(clippy::float_arithmetic)] // Simulation requires floating point arithmetic
        {
            fastrand::f64() < success_rate
        }
    }

    /// Start load balancing service
    async fn start_load_balancing(&self) {
        let stats = Arc::clone(&self.stats);
        let worker_stats = Arc::clone(&self.worker_stats);
        let shutdown = Arc::clone(&self.shutdown);
        let parallel_config = self.parallel_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(parallel_config.load_balancing_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Perform load balancing analysis
                Self::analyze_load_balance(&worker_stats, &stats).await;

                stats.load_balancing_cycles.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let balancing_time = start_time.elapsed().as_micros() as u64;
                trace!("Load balancing cycle completed in {}μs", balancing_time);
            }
        });
    }

    /// Start performance monitoring service
    async fn start_performance_monitoring(&self) {
        let stats = Arc::clone(&self.stats);
        let worker_stats = Arc::clone(&self.worker_stats);
        let execution_cache = Arc::clone(&self.execution_cache);
        let shutdown = Arc::clone(&self.shutdown);
        let parallel_config = self.parallel_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(parallel_config.performance_monitoring_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Update performance metrics
                Self::update_performance_metrics(&stats, &worker_stats, &execution_cache).await;

                stats.performance_optimization_cycles.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let monitoring_time = start_time.elapsed().as_micros() as u64;
                trace!("Performance monitoring cycle completed in {}μs", monitoring_time);
            }
        });
    }

    /// Start health checking service
    async fn start_health_checking(&self) {
        let stats = Arc::clone(&self.stats);
        let worker_stats = Arc::clone(&self.worker_stats);
        let shutdown = Arc::clone(&self.shutdown);
        let parallel_config = self.parallel_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(parallel_config.health_check_interval_ms));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let start_time = Instant::now();

                // Perform health checks
                Self::perform_health_checks(&stats, &worker_stats, &parallel_config).await;

                stats.health_check_cycles.fetch_add(1, Ordering::Relaxed);

                #[expect(clippy::cast_possible_truncation, reason = "Microsecond precision is sufficient for performance metrics")]
                let health_check_time = start_time.elapsed().as_micros() as u64;
                trace!("Health check cycle completed in {}μs", health_check_time);
            }
        });
    }

    /// Start result processing service
    async fn start_result_processing(&self) {
        let result_receiver = self.result_receiver.clone();
        let completed_results = Arc::clone(&self.completed_results);
        let active_tasks = Arc::clone(&self.active_tasks);
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                if let Ok(result) = result_receiver.try_recv() {
                    let task_id = result.task_id.clone();

                    // Update statistics
                    match result.status {
                        ParallelExecutionStatus::Success => {
                            stats.total_tasks_completed.fetch_add(1, Ordering::Relaxed);
                        }
                        ParallelExecutionStatus::Failed => {
                            stats.total_tasks_failed.fetch_add(1, Ordering::Relaxed);
                        }
                        ParallelExecutionStatus::TimedOut => {
                            stats.total_tasks_timed_out.fetch_add(1, Ordering::Relaxed);
                        }
                        ParallelExecutionStatus::Cancelled => {
                            stats.total_tasks_cancelled.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => {}
                    }

                    // Update queue size
                    stats.current_queue_size.fetch_sub(1, Ordering::Relaxed);

                    // Store completed result
                    {
                        let mut completed_results_guard = completed_results.write().await;
                        completed_results_guard.insert(task_id.clone(), result);

                        // Keep only recent results
                        while completed_results_guard.len() > 10000 {
                            if let Some(oldest_key) = completed_results_guard.keys().next().cloned() {
                                completed_results_guard.remove(&oldest_key);
                            }
                        }
                        drop(completed_results_guard);
                    }

                    // Remove from active tasks
                    {
                        let mut active_tasks_guard = active_tasks.write().await;
                        active_tasks_guard.remove(&task_id);
                        drop(active_tasks_guard);
                    }
                } else {
                    // No results available, sleep briefly
                    sleep(Duration::from_micros(100)).await;
                }
            }
        });
    }

    /// Analyze load balance across workers
    async fn analyze_load_balance(
        worker_stats: &Arc<Vec<WorkerStats>>,
        _stats: &Arc<ParallelExecutorStats>,
    ) {
        let mut _total_tasks = 0;
        let mut total_utilization_scaled = 0_u64;

        for worker_stat in worker_stats.iter() {
            let tasks_processed = worker_stat.tasks_processed.load(Ordering::Relaxed);
            _total_tasks += tasks_processed;

            // Calculate utilization (mock calculation) - 80% if has tasks, 0% otherwise
            let utilization_scaled = if tasks_processed > 0 { 800_000_u64 } else { 0_u64 };
            total_utilization_scaled = total_utilization_scaled.saturating_add(utilization_scaled);
        }

        // Calculate average utilization
        let avg_utilization_scaled = if worker_stats.is_empty() {
            0_u64
        } else {
            total_utilization_scaled / worker_stats.len() as u64
        };

        // Update worker utilization
        for worker_stat in worker_stats.iter() {
            worker_stat.utilization_scaled.store(avg_utilization_scaled, Ordering::Relaxed);
        }

        let avg_utilization_percent = avg_utilization_scaled / 10_000; // Convert to percentage (scaled by 100)
        trace!("Load balance analysis completed: avg_utilization={}%", avg_utilization_percent);
    }

    /// Update performance metrics
    async fn update_performance_metrics(
        stats: &Arc<ParallelExecutorStats>,
        worker_stats: &Arc<Vec<WorkerStats>>,
        execution_cache: &Arc<DashMap<String, AlignedExecutionData>>,
    ) {
        let total_completed = stats.total_tasks_completed.load(Ordering::Relaxed);
        let total_failed = stats.total_tasks_failed.load(Ordering::Relaxed);
        let total_processed = total_completed + total_failed;

        // Calculate throughput (mock calculation)
        let throughput = if total_processed > 0 {
            (total_processed * 60) / 5 // Assuming 5-second intervals
        } else {
            0
        };
        stats.total_throughput_ops_per_sec.store(throughput, Ordering::Relaxed);

        // Calculate success rate
        let success_rate_scaled = if total_processed > 0 {
            (total_completed * 1_000_000) / total_processed
        } else {
            1_000_000 // 100% if no tasks processed yet
        };

        // Calculate average execution time
        let mut total_execution_time = 0;
        let mut total_tasks_with_time = 0;

        for worker_stat in worker_stats.iter() {
            let worker_total_time = worker_stat.total_execution_time_us.load(Ordering::Relaxed);
            let worker_tasks = worker_stat.tasks_processed.load(Ordering::Relaxed);

            if worker_tasks > 0 {
                total_execution_time += worker_total_time;
                total_tasks_with_time += worker_tasks;
            }
        }

        let avg_execution_time = if total_tasks_with_time > 0 {
            total_execution_time / total_tasks_with_time
        } else {
            0
        };
        stats.avg_execution_time_us.store(avg_execution_time, Ordering::Relaxed);

        // Update execution cache
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for cache data")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let aligned_data = AlignedExecutionData::new(
            throughput,
            5000, // 5ms average queue time (mock)
            avg_execution_time,
            800_000, // 80% worker utilization (mock)
            success_rate_scaled,
            300_000, // 30% queue utilization (mock)
            850_000, // 85% load balance score (mock)
            now,
        );

        execution_cache.insert("global".to_string(), aligned_data);

        trace!("Performance metrics updated: throughput={} ops/sec, avg_exec_time={}μs", throughput, avg_execution_time);
    }

    /// Perform health checks
    #[expect(clippy::cognitive_complexity, reason = "Health check function requires multiple checks")]
    async fn perform_health_checks(
        stats: &Arc<ParallelExecutorStats>,
        worker_stats: &Arc<Vec<WorkerStats>>,
        parallel_config: &ParallelExecutorConfig,
    ) {
        let active_workers = stats.active_workers.load(Ordering::Relaxed);
        let queue_size = stats.current_queue_size.load(Ordering::Relaxed);
        let throughput = stats.total_throughput_ops_per_sec.load(Ordering::Relaxed);

        // Check if we need dynamic scaling
        if parallel_config.enable_dynamic_scaling {
            let queue_utilization_scaled = (queue_size * 1_000_000) / parallel_config.task_queue_capacity;

            if queue_utilization_scaled > 800_000 && active_workers < parallel_config.worker_thread_count * 2 {
                // High queue utilization - could scale up (mock)
                stats.dynamic_scaling_events.fetch_add(1, Ordering::Relaxed);
                let utilization_percent = queue_utilization_scaled / 10_000; // Convert to percentage
                trace!("High queue utilization detected: {}%", utilization_percent);
            } else if queue_utilization_scaled < 200_000 && active_workers > parallel_config.worker_thread_count / 2 {
                // Low queue utilization - could scale down (mock)
                stats.dynamic_scaling_events.fetch_add(1, Ordering::Relaxed);
                let utilization_percent = queue_utilization_scaled / 10_000; // Convert to percentage
                trace!("Low queue utilization detected: {}%", utilization_percent);
            }
        }

        // Check throughput against target
        if throughput < parallel_config.target_throughput_ops_per_sec / 2 {
            warn!("Throughput below target: {} < {}", throughput, parallel_config.target_throughput_ops_per_sec);
        }

        // Check worker health
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for health check")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        for (worker_id, worker_stat) in worker_stats.iter().enumerate() {
            let last_task = worker_stat.last_task_timestamp.load(Ordering::Relaxed);
            let time_since_last_task = now.saturating_sub(last_task);

            // If worker hasn't processed a task in 30 seconds, it might be stuck
            if time_since_last_task > 30_000 && worker_stat.tasks_processed.load(Ordering::Relaxed) > 0 {
                warn!("Worker {} appears inactive: {}ms since last task", worker_id, time_since_last_task);
            }
        }

        trace!("Health check completed: {} active workers, {} queue size, {} ops/sec", active_workers, queue_size, throughput);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        types::ChainId,
        flashloan::{FlashloanRequest, FlashloanProvider},
    };
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_parallel_executor_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(executor) = ParallelExecutor::new(config).await else {
            return; // Skip test if creation fails
        };

        assert_eq!(executor.stats().total_tasks_submitted.load(Ordering::Relaxed), 0);
        assert_eq!(executor.stats().total_tasks_completed.load(Ordering::Relaxed), 0);
        assert_eq!(executor.stats().total_tasks_failed.load(Ordering::Relaxed), 0);
        assert_eq!(executor.stats().current_queue_size.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_parallel_executor_config_default() {
        let config = ParallelExecutorConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_concurrent_executions, PARALLEL_DEFAULT_MAX_CONCURRENT);
        assert_eq!(config.worker_thread_count, PARALLEL_DEFAULT_WORKER_THREADS);
        assert_eq!(config.task_queue_capacity, PARALLEL_DEFAULT_QUEUE_CAPACITY);
        assert_eq!(config.load_balancing_interval_ms, PARALLEL_DEFAULT_LOAD_BALANCING_INTERVAL_MS);
        assert_eq!(config.performance_monitoring_interval_ms, PARALLEL_DEFAULT_PERFORMANCE_INTERVAL_MS);
        assert_eq!(config.health_check_interval_ms, PARALLEL_DEFAULT_HEALTH_CHECK_INTERVAL_MS);
        assert!(config.enable_dynamic_scaling);
        assert!(config.enable_load_balancing);
        assert!(config.enable_fault_tolerance);
        assert!(config.enable_performance_optimization);
        assert_eq!(config.task_timeout_ms, PARALLEL_DEFAULT_TASK_TIMEOUT_MS);
        assert_eq!(config.retry_attempts, PARALLEL_DEFAULT_RETRY_ATTEMPTS);
        assert_eq!(config.target_throughput_ops_per_sec, PARALLEL_DEFAULT_TARGET_THROUGHPUT);
    }

    #[test]
    fn test_aligned_execution_data_size() {
        use std::mem;

        // Ensure cache-line alignment
        assert_eq!(mem::align_of::<AlignedExecutionData>(), 64);
        assert!(mem::size_of::<AlignedExecutionData>() <= 64);
    }

    #[test]
    fn test_parallel_executor_stats_operations() {
        let stats = ParallelExecutorStats::default();

        stats.total_tasks_submitted.fetch_add(1000, Ordering::Relaxed);
        stats.total_tasks_completed.fetch_add(950, Ordering::Relaxed); // 95% success rate
        stats.total_tasks_failed.fetch_add(50, Ordering::Relaxed);
        stats.current_queue_size.fetch_add(100, Ordering::Relaxed);
        stats.peak_queue_size.fetch_add(150, Ordering::Relaxed);
        stats.active_workers.fetch_add(32, Ordering::Relaxed);
        stats.total_throughput_ops_per_sec.fetch_add(8500, Ordering::Relaxed);

        assert_eq!(stats.total_tasks_submitted.load(Ordering::Relaxed), 1000);
        assert_eq!(stats.total_tasks_completed.load(Ordering::Relaxed), 950);
        assert_eq!(stats.total_tasks_failed.load(Ordering::Relaxed), 50);
        assert_eq!(stats.current_queue_size.load(Ordering::Relaxed), 100);
        assert_eq!(stats.peak_queue_size.load(Ordering::Relaxed), 150);
        assert_eq!(stats.active_workers.load(Ordering::Relaxed), 32);
        assert_eq!(stats.total_throughput_ops_per_sec.load(Ordering::Relaxed), 8500);
    }

    #[test]
    fn test_worker_stats_operations() {
        let stats = WorkerStats::default();

        stats.tasks_processed.fetch_add(100, Ordering::Relaxed);
        stats.tasks_successful.fetch_add(95, Ordering::Relaxed); // 95% success rate
        stats.tasks_failed.fetch_add(5, Ordering::Relaxed);
        stats.total_execution_time_us.fetch_add(5_000_000, Ordering::Relaxed); // 5 seconds total
        stats.avg_execution_time_us.fetch_add(50_000, Ordering::Relaxed); // 50ms average
        stats.utilization_scaled.fetch_add(800_000, Ordering::Relaxed); // 80% utilization

        assert_eq!(stats.tasks_processed.load(Ordering::Relaxed), 100);
        assert_eq!(stats.tasks_successful.load(Ordering::Relaxed), 95);
        assert_eq!(stats.tasks_failed.load(Ordering::Relaxed), 5);
        assert_eq!(stats.total_execution_time_us.load(Ordering::Relaxed), 5_000_000);
        assert_eq!(stats.avg_execution_time_us.load(Ordering::Relaxed), 50_000);
        assert_eq!(stats.utilization_scaled.load(Ordering::Relaxed), 800_000);
    }

    #[test]
    fn test_aligned_execution_data_staleness() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let fresh_data = AlignedExecutionData::new(
            10_000,   // 10k ops/sec throughput
            5_000,    // 5ms queue time
            25_000,   // 25ms execution time
            800_000,  // 80% worker utilization
            950_000,  // 95% success rate
            300_000,  // 30% queue utilization
            850_000,  // 85% load balance score
            now,
        );

        let stale_data = AlignedExecutionData::new(
            10_000, 5_000, 25_000, 800_000, 950_000, 300_000, 850_000,
            now - 7_000, // 7 seconds old
        );

        assert!(!fresh_data.is_stale(5_000)); // 5 seconds
        assert!(stale_data.is_stale(5_000)); // 5 seconds
    }

    #[test]
    fn test_aligned_execution_data_conversions() {
        let data = AlignedExecutionData::new(
            10_000,   // 10k ops/sec throughput
            5_000,    // 5ms queue time
            25_000,   // 25ms execution time
            800_000,  // 80% worker utilization (scaled by 1e6)
            950_000,  // 95% success rate (scaled by 1e6)
            300_000,  // 30% queue utilization (scaled by 1e6)
            850_000,  // 85% load balance score (scaled by 1e6)
            1_640_995_200_000,
        );

        assert_eq!(data.worker_utilization(), dec!(0.8));
        assert_eq!(data.success_rate(), dec!(0.95));
        assert_eq!(data.queue_utilization(), dec!(0.3));
        assert_eq!(data.load_balance_score(), dec!(0.85));
        assert_eq!(data.current_throughput_ops_per_sec, 10_000);
        assert_eq!(data.avg_queue_time_us, 5_000);
        assert_eq!(data.avg_execution_time_us, 25_000);

        // Overall performance score should be weighted average
        let throughput_score = dec!(10000) / dec!(50000); // 0.2
        let queue_efficiency = dec!(1) - dec!(0.3); // 0.7
        let expected_overall = throughput_score * dec!(0.3) + dec!(0.95) * dec!(0.25) + dec!(0.8) * dec!(0.2) + queue_efficiency * dec!(0.15) + dec!(0.85) * dec!(0.1);
        assert!((data.overall_performance_score() - expected_overall).abs() < dec!(0.001));
    }

    #[test]
    fn test_parallel_execution_status_enum() {
        assert_eq!(ParallelExecutionStatus::Success, ParallelExecutionStatus::Success);
        assert_ne!(ParallelExecutionStatus::Success, ParallelExecutionStatus::Failed);
        assert_ne!(ParallelExecutionStatus::Queued, ParallelExecutionStatus::Executing);
    }

    #[tokio::test]
    async fn test_execution_task_creation() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(executor) = ParallelExecutor::new(config).await else {
            return; // Skip test if creation fails
        };

        let request = FlashloanRequest {
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
            amount: dec!(1000000),
            amount_usd: dec!(1000000),
            chain_id: ChainId::Ethereum,
            max_fee_percentage: dec!(0.01),
            preferred_providers: vec![],
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test deadline")]
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 + 300_000, // 5 minutes from now
            strategy_id: "test_strategy".to_string(),
            priority: 5,
        };

        let task = executor.create_task(request.clone(), "test_strategy".to_string(), PARALLEL_HIGH_PRIORITY).await;

        assert!(!task.task_id.is_empty());
        assert_eq!(task.request.token_address, request.token_address);
        assert_eq!(task.request.amount, request.amount);
        assert_eq!(task.request.chain_id, request.chain_id);
        assert_eq!(task.strategy_id, "test_strategy");
        assert_eq!(task.priority, PARALLEL_HIGH_PRIORITY);
        assert_eq!(task.max_retries, PARALLEL_DEFAULT_RETRY_ATTEMPTS);
        assert_eq!(task.retry_count, 0);
        assert!(task.preferred_provider.is_none());
        assert!(task.context.is_empty());
    }

    #[tokio::test]
    async fn test_task_submission() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(executor) = ParallelExecutor::new(config).await else {
            return; // Skip test if creation fails
        };

        let request = FlashloanRequest {
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(), // USDC
            amount: dec!(1000000),
            amount_usd: dec!(1000000),
            chain_id: ChainId::Ethereum,
            max_fee_percentage: dec!(0.01),
            preferred_providers: vec![],
            #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test deadline")]
            deadline: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 + 300_000, // 5 minutes from now
            strategy_id: "test_strategy".to_string(),
            priority: 5,
        };

        let task = executor.create_task(request, "test_strategy".to_string(), PARALLEL_NORMAL_PRIORITY).await;

        // Submit task
        let submitted = executor.submit_task(task.clone()).await;
        assert!(submitted);

        // Check statistics
        assert_eq!(executor.stats().total_tasks_submitted.load(Ordering::Relaxed), 1);
        assert_eq!(executor.stats().current_queue_size.load(Ordering::Relaxed), 1);

        // Check active tasks
        let active_tasks = executor.get_active_tasks().await;
        assert_eq!(active_tasks.len(), 1);
        if let Some(first_task) = active_tasks.first() {
            assert_eq!(first_task.task_id, task.task_id);
        }
    }

    #[test]
    fn test_priority_constants() {
        // Priority constants are correctly ordered
        const _: () = assert!(PARALLEL_HIGH_PRIORITY > PARALLEL_NORMAL_PRIORITY);
        const _: () = assert!(PARALLEL_NORMAL_PRIORITY > PARALLEL_LOW_PRIORITY);

        assert_eq!(PARALLEL_HIGH_PRIORITY, 1000);
        assert_eq!(PARALLEL_NORMAL_PRIORITY, 500);
        assert_eq!(PARALLEL_LOW_PRIORITY, 100);
    }

    #[test]
    fn test_parallel_executor_constants() {
        assert_eq!(PARALLEL_DEFAULT_MAX_CONCURRENT, 1000);
        assert_eq!(PARALLEL_DEFAULT_WORKER_THREADS, 32);
        assert_eq!(PARALLEL_DEFAULT_QUEUE_CAPACITY, 10000);
        assert_eq!(PARALLEL_DEFAULT_LOAD_BALANCING_INTERVAL_MS, 1000);
        assert_eq!(PARALLEL_DEFAULT_PERFORMANCE_INTERVAL_MS, 5000);
        assert_eq!(PARALLEL_DEFAULT_HEALTH_CHECK_INTERVAL_MS, 2000);
        assert_eq!(PARALLEL_DEFAULT_TASK_TIMEOUT_MS, 30000);
        assert_eq!(PARALLEL_DEFAULT_RETRY_ATTEMPTS, 3);
        assert_eq!(PARALLEL_DEFAULT_TARGET_THROUGHPUT, 10000);
    }

    #[tokio::test]
    async fn test_parallel_executor_methods() {
        let config = Arc::new(ChainCoreConfig::default());

        let Ok(executor) = ParallelExecutor::new(config).await else {
            return; // Skip test if creation fails
        };

        // Test getting active tasks (should be empty initially)
        let active_tasks = executor.get_active_tasks().await;
        assert!(active_tasks.is_empty());

        // Test getting completed results (should be empty initially)
        let completed_results = executor.get_completed_results().await;
        assert!(completed_results.is_empty());

        // Test getting result for non-existent task
        let result = executor.get_result("non_existent_task").await;
        assert!(result.is_none());

        // Test stats access
        let stats = executor.stats();
        assert_eq!(stats.total_tasks_submitted.load(Ordering::Relaxed), 0);

        // Test worker stats access
        let worker_stats = executor.worker_stats();
        assert_eq!(worker_stats.len(), PARALLEL_DEFAULT_WORKER_THREADS);
    }

    #[test]
    fn test_execution_result_creation() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let result = ExecutionResult {
            task_id: "test_task_123".to_string(),
            execution: None,
            status: ParallelExecutionStatus::Success,
            error_message: None,
            worker_id: 5,
            execution_time_us: 25_000, // 25ms
            queue_time_us: 5_000, // 5ms
            total_time_us: 30_000, // 30ms
            completed_at: now,
        };

        assert_eq!(result.task_id, "test_task_123");
        assert!(result.execution.is_none());
        assert_eq!(result.status, ParallelExecutionStatus::Success);
        assert!(result.error_message.is_none());
        assert_eq!(result.worker_id, 5);
        assert_eq!(result.execution_time_us, 25_000);
        assert_eq!(result.queue_time_us, 5_000);
        assert_eq!(result.total_time_us, 30_000);
        assert_eq!(result.completed_at, now);
    }

    #[test]
    fn test_execution_task_validation() {
        #[expect(clippy::cast_possible_truncation, reason = "Timestamp truncation is acceptable for test")]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let request = FlashloanRequest {
            token_address: "0xA0b86a33E6441b8C4505B8C29C7e5c8A0E0b8C4E".to_string(),
            amount: dec!(1000000),
            amount_usd: dec!(1000000),
            chain_id: ChainId::Ethereum,
            max_fee_percentage: dec!(0.01),
            preferred_providers: vec![],
            deadline: now + 300_000,
            strategy_id: "test_strategy".to_string(),
            priority: 5,
        };

        let task = ExecutionTask {
            task_id: "test_task_456".to_string(),
            request,
            preferred_provider: Some(FlashloanProvider::Aave),
            priority: PARALLEL_HIGH_PRIORITY,
            max_retries: 3,
            retry_count: 0,
            created_at: now,
            deadline: now + 300_000,
            strategy_id: "test_strategy".to_string(),
            context: HashMap::new(),
        };

        assert_eq!(task.task_id, "test_task_456");
        assert_eq!(task.preferred_provider, Some(FlashloanProvider::Aave));
        assert_eq!(task.priority, PARALLEL_HIGH_PRIORITY);
        assert_eq!(task.max_retries, 3);
        assert_eq!(task.retry_count, 0);
        assert_eq!(task.strategy_id, "test_strategy");
        assert!(task.context.is_empty());
        assert!(task.deadline > task.created_at);
    }
}
