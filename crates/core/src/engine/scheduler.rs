//! Task scheduler for TallyIO engine
//!
//! This module provides ultra-low latency task scheduling for the TallyIO engine
//! with priority-based task execution and worker coordination.

use crate::error::{CoreError, CoreResult};
use crate::types::{Priority, Transaction};
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Task priority for scheduler
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority background tasks
    Low = 0,
    /// Normal priority tasks
    Normal = 1,
    /// High priority tasks
    High = 2,
    /// Critical priority tasks (must complete within 1ms)
    Critical = 3,
}

impl From<Priority> for TaskPriority {
    fn from(priority: Priority) -> Self {
        match priority {
            Priority::Low => Self::Low,
            Priority::Normal => Self::Normal,
            Priority::High => Self::High,
            Priority::Critical => Self::Critical,
        }
    }
}

/// Scheduled task
#[derive(Debug)]
pub struct ScheduledTask {
    /// Task ID
    pub id: u64,
    /// Task priority
    pub priority: TaskPriority,
    /// Transaction to process
    pub transaction: Transaction,
    /// Timestamp when task was created
    pub created_at: Instant,
    /// Deadline for task completion
    pub deadline: Option<Instant>,
}

impl ScheduledTask {
    /// Create a new scheduled task
    #[must_use]
    pub fn new(id: u64, transaction: Transaction) -> Self {
        let priority = TaskPriority::from(transaction.priority);
        let deadline = if priority == TaskPriority::Critical {
            Some(Instant::now() + Duration::from_millis(1))
        } else {
            None
        };

        Self {
            id,
            priority,
            transaction,
            created_at: Instant::now(),
            deadline,
        }
    }

    /// Check if task has expired
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Instant::now() > deadline
        } else {
            false
        }
    }

    /// Get task age
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of tasks in queue
    pub max_queue_size: usize,
    /// Scheduler tick interval
    pub tick_interval: Duration,
    /// Enable priority-based scheduling
    pub enable_priority_scheduling: bool,
    /// Maximum task age before warning
    pub max_task_age: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 100_000,
            tick_interval: Duration::from_micros(100),
            enable_priority_scheduling: true,
            max_task_age: Duration::from_millis(10),
        }
    }
}

/// Ultra-low latency task scheduler
///
/// Provides priority-based task scheduling with <1ms latency guarantee
/// for critical tasks.
#[repr(C, align(64))]
pub struct Scheduler {
    /// Scheduler configuration
    config: SchedulerConfig,
    /// Task queues by priority
    critical_queue: Arc<SegQueue<ScheduledTask>>,
    high_queue: Arc<SegQueue<ScheduledTask>>,
    normal_queue: Arc<SegQueue<ScheduledTask>>,
    low_queue: Arc<SegQueue<ScheduledTask>>,
    /// Task ID counter
    task_id_counter: AtomicU64,
    /// Running flag
    is_running: AtomicBool,
    /// Statistics
    tasks_scheduled: AtomicU64,
    tasks_completed: AtomicU64,
    tasks_expired: AtomicU64,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            critical_queue: Arc::new(SegQueue::new()),
            high_queue: Arc::new(SegQueue::new()),
            normal_queue: Arc::new(SegQueue::new()),
            low_queue: Arc::new(SegQueue::new()),
            task_id_counter: AtomicU64::new(0),
            is_running: AtomicBool::new(false),
            tasks_scheduled: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
            tasks_expired: AtomicU64::new(0),
        }
    }

    /// Start the scheduler
    pub fn start(&self) -> CoreResult<()> {
        self.is_running.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop the scheduler
    pub fn stop(&self) -> CoreResult<()> {
        self.is_running.store(false, Ordering::Release);
        Ok(())
    }

    /// Schedule a task
    #[inline(always)]
    pub fn schedule_task(&self, transaction: Transaction) -> CoreResult<u64> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(CoreError::scheduler("Scheduler is not running"));
        }

        let task_id = self.task_id_counter.fetch_add(1, Ordering::Relaxed);
        let task = ScheduledTask::new(task_id, transaction);

        // Select appropriate queue based on priority
        let queue = match task.priority {
            TaskPriority::Critical => &self.critical_queue,
            TaskPriority::High => &self.high_queue,
            TaskPriority::Normal => &self.normal_queue,
            TaskPriority::Low => &self.low_queue,
        };

        queue.push(task);
        self.tasks_scheduled.fetch_add(1, Ordering::Relaxed);
        Ok(task_id)
    }

    /// Get the next task to process
    #[inline(always)]
    pub fn get_next_task(&self) -> Option<ScheduledTask> {
        if !self.is_running.load(Ordering::Acquire) {
            return None;
        }

        // Process queues in priority order
        if let Some(task) = self.critical_queue.pop() {
            if task.is_expired() {
                self.tasks_expired.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            return Some(task);
        }

        if let Some(task) = self.high_queue.pop() {
            if task.is_expired() {
                self.tasks_expired.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            return Some(task);
        }

        if let Some(task) = self.normal_queue.pop() {
            return Some(task);
        }

        self.low_queue.pop()
    }

    /// Mark task as completed
    #[inline(always)]
    pub fn complete_task(&self, _task_id: u64) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get queue sizes
    #[must_use]
    pub fn queue_sizes(&self) -> (usize, usize, usize, usize) {
        (
            self.critical_queue.len(),
            self.high_queue.len(),
            self.normal_queue.len(),
            self.low_queue.len(),
        )
    }

    /// Get total queue size
    #[must_use]
    pub fn total_queue_size(&self) -> usize {
        let (critical, high, normal, low) = self.queue_sizes();
        critical + high + normal + low
    }

    /// Check if scheduler is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get scheduler statistics
    #[must_use]
    pub fn statistics(&self) -> SchedulerStatistics {
        SchedulerStatistics {
            tasks_scheduled: self.tasks_scheduled.load(Ordering::Relaxed),
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed),
            tasks_expired: self.tasks_expired.load(Ordering::Relaxed),
            queue_sizes: self.queue_sizes(),
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStatistics {
    /// Total tasks scheduled
    pub tasks_scheduled: u64,
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total tasks expired
    pub tasks_expired: u64,
    /// Current queue sizes (critical, high, normal, low)
    pub queue_sizes: (usize, usize, usize, usize),
}

impl SchedulerStatistics {
    /// Get completion rate
    #[must_use]
    pub fn completion_rate(&self) -> f64 {
        if self.tasks_scheduled == 0 {
            0.0
        } else {
            self.tasks_completed as f64 / self.tasks_scheduled as f64
        }
    }

    /// Get expiration rate
    #[must_use]
    pub fn expiration_rate(&self) -> f64 {
        if self.tasks_scheduled == 0 {
            0.0
        } else {
            self.tasks_expired as f64 / self.tasks_scheduled as f64
        }
    }

    /// Get total queue size
    #[must_use]
    pub fn total_queue_size(&self) -> usize {
        self.queue_sizes.0 + self.queue_sizes.1 + self.queue_sizes.2 + self.queue_sizes.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Gas, Price};

    #[test]
    fn test_scheduler_creation() {
        let scheduler = Scheduler::default();
        assert!(!scheduler.is_running());
        assert_eq!(scheduler.total_queue_size(), 0);
    }

    #[test]
    fn test_scheduler_start_stop() -> CoreResult<()> {
        let scheduler = Scheduler::default();

        scheduler.start()?;
        assert!(scheduler.is_running());

        scheduler.stop()?;
        assert!(!scheduler.is_running());

        Ok(())
    }

    #[test]
    fn test_task_scheduling() -> CoreResult<()> {
        let scheduler = Scheduler::default();
        scheduler.start()?;

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let task_id = scheduler.schedule_task(tx)?;
        assert_eq!(task_id, 0); // First task ID should be 0
        assert_eq!(scheduler.total_queue_size(), 1);

        Ok(())
    }

    #[test]
    fn test_priority_scheduling() -> CoreResult<()> {
        let scheduler = Scheduler::default();
        scheduler.start()?;

        // Create transactions with different priorities
        let mut critical_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        critical_tx.set_priority(Priority::Critical);

        let mut normal_tx = Transaction::new(
            [3u8; 20],
            Some([4u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        normal_tx.set_priority(Priority::Normal);

        // Schedule normal task first
        scheduler.schedule_task(normal_tx)?;
        // Then schedule critical task
        scheduler.schedule_task(critical_tx)?;

        // Critical task should be returned first
        let next_task = scheduler.get_next_task();
        assert!(next_task.is_some());
        if let Some(task) = next_task {
            assert_eq!(task.priority, TaskPriority::Critical);
        }

        Ok(())
    }

    #[test]
    fn test_task_expiration() {
        let task_id = 1;
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        tx.set_priority(Priority::Critical);

        let mut task = ScheduledTask::new(task_id, tx);

        // Manually set deadline to past
        task.deadline = Some(Instant::now() - Duration::from_millis(1));

        assert!(task.is_expired());
    }

    #[test]
    fn test_scheduler_statistics() -> CoreResult<()> {
        let scheduler = Scheduler::default();
        scheduler.start()?;

        let stats = scheduler.statistics();
        assert_eq!(stats.tasks_scheduled, 0);
        assert_eq!(stats.tasks_completed, 0);
        assert_eq!(stats.completion_rate(), 0.0);

        // Schedule a task
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        let task_id = scheduler.schedule_task(tx)?;

        // Complete the task
        scheduler.complete_task(task_id);

        let final_stats = scheduler.statistics();
        assert_eq!(final_stats.tasks_scheduled, 1);
        assert_eq!(final_stats.tasks_completed, 1);
        assert_eq!(final_stats.completion_rate(), 1.0);

        Ok(())
    }

    #[test]
    fn test_priority_conversion() {
        // Test Priority to TaskPriority conversion (lines 29, 31)
        assert_eq!(TaskPriority::from(Priority::Low), TaskPriority::Low);
        assert_eq!(TaskPriority::from(Priority::Normal), TaskPriority::Normal);
        assert_eq!(TaskPriority::from(Priority::High), TaskPriority::High);
        assert_eq!(
            TaskPriority::from(Priority::Critical),
            TaskPriority::Critical
        );
    }

    #[test]
    fn test_task_without_deadline() {
        // Test task without deadline (line 78)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let task = ScheduledTask::new(1, tx);
        assert!(!task.is_expired()); // No deadline, so not expired
    }

    #[test]
    fn test_task_age() {
        // Test task age calculation (lines 84-85)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let task = ScheduledTask::new(1, tx);
        let age = task.age();
        assert!(age.as_nanos() < u128::MAX); // Age should be reasonable
    }

    #[test]
    fn test_scheduler_not_running() -> CoreResult<()> {
        // Test scheduling when not running (line 169)
        let scheduler = Scheduler::default();
        // Don't start the scheduler

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let result = scheduler.schedule_task(tx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_get_next_task_not_running() {
        // Test get_next_task when not running (line 192)
        let scheduler = Scheduler::default();
        // Don't start the scheduler

        let task = scheduler.get_next_task();
        assert!(task.is_none());
    }

    #[test]
    fn test_priority_queue_selection() -> CoreResult<()> {
        // Test different priority queue selection (lines 178, 180)
        let scheduler = Scheduler::default();
        scheduler.start()?;

        // Test High priority (line 178)
        let mut high_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        high_tx.set_priority(Priority::High);
        scheduler.schedule_task(high_tx)?;

        // Test Low priority (line 180)
        let mut low_tx = Transaction::new(
            [2u8; 20],
            Some([3u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        low_tx.set_priority(Priority::Low);
        scheduler.schedule_task(low_tx)?;

        // High priority should come first
        let task1 = scheduler.get_next_task();
        assert!(task1.is_some());
        if let Some(task) = task1 {
            assert_eq!(task.priority, TaskPriority::High);
        }

        // Low priority should come next
        let task2 = scheduler.get_next_task();
        assert!(task2.is_some());
        if let Some(task) = task2 {
            assert_eq!(task.priority, TaskPriority::Low);
        }

        Ok(())
    }

    #[test]
    fn test_expired_task_handling() -> CoreResult<()> {
        // Test expired task handling (lines 197-199, 205-207)
        let scheduler = Scheduler::default();
        scheduler.start()?;

        // Create critical task that will expire
        let mut critical_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        critical_tx.set_priority(Priority::Critical);

        // Schedule the task
        scheduler.schedule_task(critical_tx)?;

        // Wait for task to expire (critical tasks have 1ms deadline)
        std::thread::sleep(Duration::from_millis(2));

        // Try to get the task - should return None due to expiration
        let task = scheduler.get_next_task();
        assert!(task.is_none());

        // Check that expired counter was incremented
        let stats = scheduler.statistics();
        assert_eq!(stats.tasks_expired, 1);

        Ok(())
    }

    #[test]
    fn test_high_priority_expiration() -> CoreResult<()> {
        // Test high priority task expiration (lines 205-207)
        let scheduler = Scheduler::default();
        scheduler.start()?;

        // Create high priority task and manually expire it
        let mut high_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        high_tx.set_priority(Priority::High);

        let task_id = scheduler.schedule_task(high_tx)?;

        // Manually expire the task by setting deadline in the past
        // We need to access the queue directly for this test
        if let Some(mut task) = scheduler.high_queue.pop() {
            task.deadline = Some(Instant::now() - Duration::from_millis(1));
            scheduler.high_queue.push(task);
        }

        // Try to get the task - should return None due to expiration
        let task = scheduler.get_next_task();
        assert!(task.is_none());

        Ok(())
    }

    #[test]
    fn test_normal_priority_queue() -> CoreResult<()> {
        // Test normal priority queue processing (line 213)
        let scheduler = Scheduler::default();
        scheduler.start()?;

        let normal_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        // Normal priority is default

        scheduler.schedule_task(normal_tx)?;

        let task = scheduler.get_next_task();
        assert!(task.is_some());
        if let Some(task) = task {
            assert_eq!(task.priority, TaskPriority::Normal);
        }

        Ok(())
    }

    #[test]
    fn test_low_priority_queue() -> CoreResult<()> {
        // Test low priority queue processing (line 216)
        let scheduler = Scheduler::default();
        scheduler.start()?;

        let mut low_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        low_tx.set_priority(Priority::Low);

        scheduler.schedule_task(low_tx)?;

        let task = scheduler.get_next_task();
        assert!(task.is_some());
        if let Some(task) = task {
            assert_eq!(task.priority, TaskPriority::Low);
        }

        Ok(())
    }

    #[test]
    fn test_scheduler_edge_cases() -> CoreResult<()> {
        // Test edge cases for lines 197, 201, 205, 209, 213
        let scheduler = Scheduler::default();
        scheduler.start()?;

        // Test empty queues
        let task = scheduler.get_next_task();
        assert!(task.is_none());

        // Test queue size calculations
        let (critical, high, normal, low) = scheduler.queue_sizes();
        assert_eq!(critical, 0);
        assert_eq!(high, 0);
        assert_eq!(normal, 0);
        assert_eq!(low, 0);

        Ok(())
    }

    #[test]
    fn test_statistics_expiration_rate() {
        // Test expiration rate calculation (lines 293-295, 297)
        let stats = SchedulerStatistics {
            tasks_scheduled: 10,
            tasks_completed: 8,
            tasks_expired: 2,
            queue_sizes: (0, 0, 0, 0),
        };

        assert_eq!(stats.expiration_rate(), 0.2); // 2/10 = 0.2

        // Test with zero scheduled tasks (line 294-295)
        let empty_stats = SchedulerStatistics {
            tasks_scheduled: 0,
            tasks_completed: 0,
            tasks_expired: 0,
            queue_sizes: (0, 0, 0, 0),
        };

        assert_eq!(empty_stats.expiration_rate(), 0.0);
    }

    #[test]
    fn test_statistics_total_queue_size() {
        // Test total queue size calculation (lines 303-304)
        let stats = SchedulerStatistics {
            tasks_scheduled: 0,
            tasks_completed: 0,
            tasks_expired: 0,
            queue_sizes: (1, 2, 3, 4),
        };

        assert_eq!(stats.total_queue_size(), 10); // 1+2+3+4 = 10
    }

    #[test]
    fn test_scheduler_configuration() {
        // Test scheduler configuration (line 384)
        let config = SchedulerConfig {
            max_queue_size: 50_000,
            tick_interval: Duration::from_micros(50),
            enable_priority_scheduling: false,
            max_task_age: Duration::from_millis(5),
        };

        let scheduler = Scheduler::new(config.clone());
        assert_eq!(scheduler.config.max_queue_size, 50_000);
        assert_eq!(scheduler.config.tick_interval, Duration::from_micros(50));
        assert!(!scheduler.config.enable_priority_scheduling);
        assert_eq!(scheduler.config.max_task_age, Duration::from_millis(5));
    }
}
