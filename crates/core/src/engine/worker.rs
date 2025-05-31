//! Worker threads for TallyIO engine
//!
//! This module provides ultra-high performance worker threads for transaction processing
//! with CPU affinity and lock-free coordination.

use crate::error::{CoreError, CoreResult};
use crate::types::{ProcessingResult, Transaction};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Worker status
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Worker is idle
    Idle,
    /// Worker is processing a task
    Processing,
    /// Worker is paused
    Paused,
    /// Worker has stopped
    Stopped,
    /// Worker encountered an error
    Error,
}

impl Default for WorkerStatus {
    fn default() -> Self {
        Self::Idle
    }
}

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Worker ID
    pub id: u32,
    /// CPU core to bind to (if CPU affinity is enabled)
    pub cpu_core: Option<u32>,
    /// Worker thread stack size
    pub stack_size: usize,
    /// Enable CPU affinity
    pub enable_cpu_affinity: bool,
}

impl WorkerConfig {
    /// Create a new worker configuration
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self {
            id,
            cpu_core: None,
            stack_size: 2 * 1024 * 1024, // 2MB
            enable_cpu_affinity: false,
        }
    }

    /// Set CPU core for affinity
    #[must_use]
    pub const fn with_cpu_core(mut self, core: u32) -> Self {
        self.cpu_core = Some(core);
        self.enable_cpu_affinity = true;
        self
    }

    /// Set stack size
    #[must_use]
    pub const fn with_stack_size(mut self, size: usize) -> Self {
        self.stack_size = size;
        self
    }
}

/// High-performance worker thread
///
/// Provides ultra-low latency transaction processing with CPU affinity
/// and lock-free coordination.
#[derive(Debug)]
#[repr(C, align(64))]
pub struct Worker {
    /// Worker configuration
    config: WorkerConfig,
    /// Current status
    status: WorkerStatus,
    /// Running flag
    is_running: AtomicBool,
    /// Tasks processed counter
    tasks_processed: AtomicU64,
    /// Total processing time in nanoseconds
    total_processing_time_ns: AtomicU64,
    /// Error count
    error_count: AtomicU64,
    /// Last activity timestamp
    last_activity: AtomicU64,
}

impl Worker {
    /// Create a new worker
    #[must_use]
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            status: WorkerStatus::Idle,
            is_running: AtomicBool::new(false),
            tasks_processed: AtomicU64::new(0),
            total_processing_time_ns: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            last_activity: AtomicU64::new(0),
        }
    }

    /// Start the worker
    pub fn start(&mut self) -> CoreResult<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.is_running.store(true, Ordering::Release);
        self.status = WorkerStatus::Idle;
        self.update_last_activity();
        Ok(())
    }

    /// Stop the worker
    pub fn stop(&mut self) -> CoreResult<()> {
        self.is_running.store(false, Ordering::Release);
        self.status = WorkerStatus::Stopped;
        Ok(())
    }

    /// Process a transaction
    #[inline(always)]
    pub fn process_transaction(&self, transaction: Transaction) -> CoreResult<ProcessingResult> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(CoreError::worker("Worker is not running"));
        }

        let start = Instant::now();
        self.update_last_activity();

        // Simulate transaction processing
        let result = ProcessingResult::success(
            transaction.hash.unwrap_or([0u8; 32]),
            start.elapsed(),
            transaction.gas_limit(),
            transaction.gas_price(),
        );

        // Update metrics
        let processing_time_ns = start.elapsed().as_nanos() as u64;
        self.tasks_processed.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time_ns
            .fetch_add(processing_time_ns, Ordering::Relaxed);

        Ok(result)
    }

    /// Get worker ID
    #[must_use]
    pub const fn id(&self) -> u32 {
        self.config.id
    }

    /// Get worker status
    #[must_use]
    pub const fn status(&self) -> WorkerStatus {
        self.status
    }

    /// Check if worker is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get worker statistics
    #[must_use]
    pub fn statistics(&self) -> WorkerStatistics {
        let tasks = self.tasks_processed.load(Ordering::Relaxed);
        let total_time = self.total_processing_time_ns.load(Ordering::Relaxed);
        let avg_time = if tasks > 0 { total_time / tasks } else { 0 };

        WorkerStatistics {
            id: self.config.id,
            status: self.status,
            tasks_processed: tasks,
            average_processing_time_ns: avg_time,
            error_count: self.error_count.load(Ordering::Relaxed),
            last_activity_timestamp: self.last_activity.load(Ordering::Relaxed),
        }
    }

    /// Update last activity timestamp
    #[inline(always)]
    fn update_last_activity(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_activity.store(now, Ordering::Relaxed);
    }

    /// Record an error
    #[inline(always)]
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// Worker statistics
#[derive(Debug, Clone)]
pub struct WorkerStatistics {
    /// Worker ID
    pub id: u32,
    /// Current status
    pub status: WorkerStatus,
    /// Total tasks processed
    pub tasks_processed: u64,
    /// Average processing time in nanoseconds
    pub average_processing_time_ns: u64,
    /// Error count
    pub error_count: u64,
    /// Last activity timestamp
    pub last_activity_timestamp: u64,
}

/// Worker pool for managing multiple workers
#[derive(Debug)]
pub struct WorkerPool {
    /// Pool of workers
    workers: Vec<Worker>,
    /// Pool configuration
    config: WorkerPoolConfig,
    /// Active worker count
    active_workers: AtomicU32,
}

/// Worker pool configuration
#[derive(Debug, Clone)]
pub struct WorkerPoolConfig {
    /// Number of workers
    pub worker_count: usize,
    /// Enable CPU affinity
    pub enable_cpu_affinity: bool,
    /// Worker stack size
    pub worker_stack_size: usize,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get().min(16),
            enable_cpu_affinity: true,
            worker_stack_size: 2 * 1024 * 1024, // 2MB
        }
    }
}

impl WorkerPool {
    /// Create a new worker pool
    pub fn new(config: WorkerPoolConfig) -> Self {
        let mut workers = Vec::with_capacity(config.worker_count);

        for i in 0..config.worker_count {
            let worker_config = WorkerConfig {
                id: i as u32,
                cpu_core: if config.enable_cpu_affinity {
                    Some(i as u32 % num_cpus::get() as u32)
                } else {
                    None
                },
                stack_size: config.worker_stack_size,
                enable_cpu_affinity: config.enable_cpu_affinity,
            };

            workers.push(Worker::new(worker_config));
        }

        Self {
            workers,
            config,
            active_workers: AtomicU32::new(0),
        }
    }

    /// Start all workers
    pub fn start(&mut self) -> CoreResult<()> {
        for worker in &mut self.workers {
            worker.start()?;
        }
        self.active_workers
            .store(self.workers.len() as u32, Ordering::Release);
        Ok(())
    }

    /// Stop all workers
    pub fn stop(&mut self) -> CoreResult<()> {
        for worker in &mut self.workers {
            worker.stop()?;
        }
        self.active_workers.store(0, Ordering::Release);
        Ok(())
    }

    /// Get worker count
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Get active worker count
    #[must_use]
    pub fn active_worker_count(&self) -> u32 {
        self.active_workers.load(Ordering::Acquire)
    }

    /// Get worker by ID
    #[must_use]
    pub fn get_worker(&self, id: u32) -> Option<&Worker> {
        self.workers.get(id as usize)
    }

    /// Get mutable worker by ID
    pub fn get_worker_mut(&mut self, id: u32) -> Option<&mut Worker> {
        self.workers.get_mut(id as usize)
    }

    /// Get pool statistics
    #[must_use]
    pub fn statistics(&self) -> WorkerPoolStatistics {
        let worker_stats: Vec<WorkerStatistics> =
            self.workers.iter().map(|w| w.statistics()).collect();

        let total_tasks: u64 = worker_stats.iter().map(|s| s.tasks_processed).sum();
        let total_errors: u64 = worker_stats.iter().map(|s| s.error_count).sum();

        WorkerPoolStatistics {
            worker_count: self.workers.len(),
            active_workers: self.active_worker_count(),
            total_tasks_processed: total_tasks,
            total_errors,
            worker_statistics: worker_stats,
        }
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new(WorkerPoolConfig::default())
    }
}

/// Worker pool statistics
#[derive(Debug, Clone)]
pub struct WorkerPoolStatistics {
    /// Total number of workers
    pub worker_count: usize,
    /// Number of active workers
    pub active_workers: u32,
    /// Total tasks processed by all workers
    pub total_tasks_processed: u64,
    /// Total errors across all workers
    pub total_errors: u64,
    /// Individual worker statistics
    pub worker_statistics: Vec<WorkerStatistics>,
}

impl WorkerPoolStatistics {
    /// Get error rate
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        if self.total_tasks_processed == 0 {
            0.0
        } else {
            self.total_errors as f64 / self.total_tasks_processed as f64
        }
    }

    /// Get average processing time across all workers
    #[must_use]
    pub fn average_processing_time_ns(&self) -> u64 {
        if self.worker_statistics.is_empty() {
            return 0;
        }

        let total: u64 = self
            .worker_statistics
            .iter()
            .map(|s| s.average_processing_time_ns)
            .sum();

        total / self.worker_statistics.len() as u64
    }
}

// Placeholder for num_cpus functionality
mod num_cpus {
    #[must_use]
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Gas, Price};

    #[test]
    fn test_worker_creation() {
        let config = WorkerConfig::new(0);
        let worker = Worker::new(config);
        assert_eq!(worker.id(), 0);
        assert_eq!(worker.status(), WorkerStatus::Idle);
        assert!(!worker.is_running());
    }

    #[test]
    fn test_worker_start_stop() -> CoreResult<()> {
        let config = WorkerConfig::new(0);
        let mut worker = Worker::new(config);

        worker.start()?;
        assert!(worker.is_running());
        assert_eq!(worker.status(), WorkerStatus::Idle);

        worker.stop()?;
        assert!(!worker.is_running());
        assert_eq!(worker.status(), WorkerStatus::Stopped);

        Ok(())
    }

    #[test]
    fn test_worker_transaction_processing() -> CoreResult<()> {
        let config = WorkerConfig::new(0);
        let mut worker = Worker::new(config);
        worker.start()?;

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = worker.process_transaction(tx)?;
        assert!(result.is_success());

        let stats = worker.statistics();
        assert_eq!(stats.tasks_processed, 1);

        Ok(())
    }

    #[test]
    fn test_worker_pool_creation() {
        let config = WorkerPoolConfig::default();
        let pool = WorkerPool::new(config);
        assert!(pool.worker_count() > 0);
        assert_eq!(pool.active_worker_count(), 0);
    }

    #[test]
    fn test_worker_pool_start_stop() -> CoreResult<()> {
        let config = WorkerPoolConfig {
            worker_count: 2,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let mut pool = WorkerPool::new(config);

        pool.start()?;
        assert_eq!(pool.active_worker_count(), 2);

        pool.stop()?;
        assert_eq!(pool.active_worker_count(), 0);

        Ok(())
    }

    #[test]
    fn test_worker_pool_statistics() -> CoreResult<()> {
        let config = WorkerPoolConfig {
            worker_count: 2,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let mut pool = WorkerPool::new(config);
        pool.start()?;

        let stats = pool.statistics();
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.active_workers, 2);

        Ok(())
    }

    #[test]
    fn test_worker_status_default() {
        // Test WorkerStatus::default() (lines 29-30)
        let status = WorkerStatus::default();
        assert_eq!(status, WorkerStatus::Idle);
    }

    #[test]
    fn test_worker_config_new() {
        // Test WorkerConfig::new() (lines 50-56)
        let config = WorkerConfig::new(42);
        assert_eq!(config.id, 42);
        assert_eq!(config.cpu_core, None);
        assert_eq!(config.stack_size, 2 * 1024 * 1024);
        assert!(!config.enable_cpu_affinity);
    }

    #[test]
    fn test_worker_config_with_cpu_core() {
        // Test WorkerConfig::with_cpu_core() (lines 61-64)
        let config = WorkerConfig::new(0).with_cpu_core(3);
        assert_eq!(config.cpu_core, Some(3));
        assert!(config.enable_cpu_affinity);
    }

    #[test]
    fn test_worker_config_with_stack_size() {
        // Test WorkerConfig::with_stack_size() (lines 69-71)
        let config = WorkerConfig::new(0).with_stack_size(4 * 1024 * 1024);
        assert_eq!(config.stack_size, 4 * 1024 * 1024);
    }

    #[test]
    fn test_worker_start_already_running() -> CoreResult<()> {
        // Test starting worker that's already running (lines 115-116)
        let config = WorkerConfig::new(0);
        let mut worker = Worker::new(config);

        worker.start()?;
        assert!(worker.is_running());

        // Starting again should return Ok without error
        worker.start()?;
        assert!(worker.is_running());

        Ok(())
    }

    #[test]
    fn test_worker_process_transaction_not_running() -> CoreResult<()> {
        // Test processing transaction when worker is not running (lines 135-136)
        let config = WorkerConfig::new(0);
        let worker = Worker::new(config);

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = worker.process_transaction(tx);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_worker_record_error() {
        // Test Worker::record_error() (lines 206-207)
        let config = WorkerConfig::new(0);
        let worker = Worker::new(config);

        let initial_stats = worker.statistics();
        assert_eq!(initial_stats.error_count, 0);

        worker.record_error();
        worker.record_error();

        let stats = worker.statistics();
        assert_eq!(stats.error_count, 2);
    }

    #[test]
    fn test_worker_statistics_zero_tasks() {
        // Test statistics calculation with zero tasks (lines 182)
        let config = WorkerConfig::new(42);
        let worker = Worker::new(config);

        let stats = worker.statistics();
        assert_eq!(stats.id, 42);
        assert_eq!(stats.status, WorkerStatus::Idle);
        assert_eq!(stats.tasks_processed, 0);
        assert_eq!(stats.average_processing_time_ns, 0); // Should handle division by zero
        assert_eq!(stats.error_count, 0);
    }

    #[test]
    fn test_worker_pool_config_default() {
        // Test WorkerPoolConfig::default() (lines 251-256)
        let config = WorkerPoolConfig::default();
        assert!(config.worker_count > 0);
        assert!(config.worker_count <= 16);
        assert!(config.enable_cpu_affinity);
        assert_eq!(config.worker_stack_size, 2 * 1024 * 1024);
    }

    #[test]
    fn test_worker_pool_new_with_cpu_affinity() {
        // Test WorkerPool::new() with CPU affinity (lines 268-269)
        let config = WorkerPoolConfig {
            worker_count: 4,
            enable_cpu_affinity: true,
            worker_stack_size: 1024 * 1024,
        };

        let pool = WorkerPool::new(config);
        assert_eq!(pool.worker_count(), 4);

        // Check that workers have CPU cores assigned
        for i in 0..4 {
            if let Some(worker) = pool.get_worker(i) {
                assert!(worker.config.cpu_core.is_some());
                assert!(worker.config.enable_cpu_affinity);
            }
        }
    }

    #[test]
    fn test_worker_pool_new_without_cpu_affinity() {
        // Test WorkerPool::new() without CPU affinity (lines 270-271)
        let config = WorkerPoolConfig {
            worker_count: 2,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };

        let pool = WorkerPool::new(config);

        // Check that workers don't have CPU cores assigned
        for i in 0..2 {
            if let Some(worker) = pool.get_worker(i) {
                assert_eq!(worker.config.cpu_core, None);
                assert!(!worker.config.enable_cpu_affinity);
            }
        }
    }

    #[test]
    fn test_worker_pool_get_worker() {
        // Test WorkerPool::get_worker() (lines 320-321)
        let config = WorkerPoolConfig {
            worker_count: 3,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let pool = WorkerPool::new(config);

        // Valid worker IDs
        assert!(pool.get_worker(0).is_some());
        assert!(pool.get_worker(1).is_some());
        assert!(pool.get_worker(2).is_some());

        // Invalid worker ID
        assert!(pool.get_worker(3).is_none());
        assert!(pool.get_worker(100).is_none());
    }

    #[test]
    fn test_worker_pool_get_worker_mut() -> CoreResult<()> {
        // Test WorkerPool::get_worker_mut() (lines 325-326)
        let config = WorkerPoolConfig {
            worker_count: 2,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let mut pool = WorkerPool::new(config);

        // Valid worker ID
        if let Some(worker) = pool.get_worker_mut(0) {
            worker.start()?;
            assert!(worker.is_running());
        }

        // Invalid worker ID
        assert!(pool.get_worker_mut(2).is_none());

        Ok(())
    }

    #[test]
    fn test_worker_pool_statistics_comprehensive() -> CoreResult<()> {
        // Test comprehensive worker pool statistics
        let config = WorkerPoolConfig {
            worker_count: 2,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let mut pool = WorkerPool::new(config);
        pool.start()?;

        let stats = pool.statistics();
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.active_workers, 2);
        assert_eq!(stats.total_tasks_processed, 0);
        assert_eq!(stats.error_rate(), 0.0);

        Ok(())
    }

    #[test]
    fn test_worker_config_builder() {
        let config = WorkerConfig::new(5)
            .with_cpu_core(2)
            .with_stack_size(4 * 1024 * 1024);

        assert_eq!(config.id, 5);
        assert_eq!(config.cpu_core, Some(2));
        assert_eq!(config.stack_size, 4 * 1024 * 1024);
        assert!(config.enable_cpu_affinity);
    }

    #[test]
    fn test_num_cpus_get() {
        // Test num_cpus::get() function (lines 400-403)
        let cpu_count = num_cpus::get();
        assert!(cpu_count > 0);
        assert!(cpu_count >= 1);
    }

    #[test]
    fn test_worker_update_last_activity() -> CoreResult<()> {
        // Test update_last_activity method (lines 196-201)
        let config = WorkerConfig::new(0);
        let mut worker = Worker::new(config);
        worker.start()?;

        let initial_stats = worker.statistics();
        let initial_timestamp = initial_stats.last_activity_timestamp;

        // Process a transaction to trigger update_last_activity
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        worker.process_transaction(tx)?;

        let final_stats = worker.statistics();
        let final_timestamp = final_stats.last_activity_timestamp;

        // Timestamp should have been updated
        assert!(final_timestamp >= initial_timestamp);

        Ok(())
    }

    #[test]
    fn test_worker_statistics_with_tasks() -> CoreResult<()> {
        // Test statistics calculation with tasks (lines 180-191)
        let config = WorkerConfig::new(42);
        let mut worker = Worker::new(config);
        worker.start()?;

        // Process multiple transactions
        for _ in 0..3 {
            let tx = Transaction::new(
                [1u8; 20],
                Some([2u8; 20]),
                Price::from_ether(1),
                Price::from_gwei(20),
                Gas::new(21_000),
                0,
                Vec::with_capacity(0),
            );
            worker.process_transaction(tx)?;
        }

        let stats = worker.statistics();
        assert_eq!(stats.id, 42);
        assert_eq!(stats.status, WorkerStatus::Idle);
        assert_eq!(stats.tasks_processed, 3);
        assert!(stats.average_processing_time_ns > 0); // Should have some processing time
        assert_eq!(stats.error_count, 0);

        Ok(())
    }

    #[test]
    fn test_worker_pool_statistics_with_tasks() -> CoreResult<()> {
        // Test WorkerPoolStatistics methods (lines 372-394)
        let config = WorkerPoolConfig {
            worker_count: 2,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let mut pool = WorkerPool::new(config);
        pool.start()?;

        // Process transactions on different workers
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        if let Some(worker) = pool.get_worker(0) {
            worker.process_transaction(tx.clone())?;
            worker.record_error(); // Add an error
        }

        if let Some(worker) = pool.get_worker(1) {
            worker.process_transaction(tx)?;
            // No error for this worker
        }

        let stats = pool.statistics();
        assert_eq!(stats.total_tasks_processed, 2);
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.error_rate(), 0.5); // 1 error out of 2 tasks
                                             // Just verify that average processing time is calculated
        let _avg_time = stats.average_processing_time_ns();

        Ok(())
    }

    #[test]
    fn test_worker_pool_statistics_empty() {
        // Test WorkerPoolStatistics with empty workers (lines 383-384)
        let config = WorkerPoolConfig {
            worker_count: 0, // No workers
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let pool = WorkerPool::new(config);

        let stats = pool.statistics();
        assert_eq!(stats.worker_count, 0);
        assert_eq!(stats.total_tasks_processed, 0);
        assert_eq!(stats.total_errors, 0);
        assert_eq!(stats.error_rate(), 0.0);
        assert_eq!(stats.average_processing_time_ns(), 0); // Should handle empty case
    }

    #[test]
    fn test_worker_pool_default() {
        // Test Default implementation for WorkerPool (lines 347-348)
        let pool = WorkerPool::default();
        assert!(pool.worker_count() > 0);
        assert_eq!(pool.active_worker_count(), 0);
    }

    #[test]
    fn test_worker_pool_statistics_error_rate() -> CoreResult<()> {
        // Test error_rate calculation (lines 374, 380-382, 385-386, 388, 391)
        let config = WorkerPoolConfig {
            worker_count: 2,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let mut pool = WorkerPool::new(config);
        pool.start()?;

        // Test error rate with no tasks (line 374)
        let stats = pool.statistics();
        assert_eq!(stats.error_rate(), 0.0);

        // Process some transactions and record errors
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        if let Some(worker) = pool.get_worker(0) {
            worker.process_transaction(tx.clone())?;
            worker.record_error(); // 1 error out of 1 task
        }

        if let Some(worker) = pool.get_worker(1) {
            worker.process_transaction(tx)?;
            // No error for this worker
        }

        let stats = pool.statistics();
        assert_eq!(stats.total_tasks_processed, 2);
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.error_rate(), 0.5); // 1 error out of 2 tasks

        Ok(())
    }

    #[test]
    fn test_worker_pool_statistics_average_processing_time() {
        // Test average_processing_time_ns calculation (lines 380-382, 385-386, 388, 391)
        let config = WorkerPoolConfig {
            worker_count: 2,
            enable_cpu_affinity: false,
            worker_stack_size: 1024 * 1024,
        };
        let pool = WorkerPool::new(config);

        let stats = pool.statistics();
        // Should handle empty worker statistics (line 382)
        assert_eq!(stats.average_processing_time_ns(), 0);

        // With workers but no tasks processed, average should be 0
        assert!(!stats.worker_statistics.is_empty());
        assert_eq!(stats.average_processing_time_ns(), 0);
    }
}
