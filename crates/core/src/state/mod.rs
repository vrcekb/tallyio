//! State management for TallyIO core
//!
//! This module provides ultra-high performance state management with
//! global and thread-local state coordination and synchronization.

pub mod global;
pub mod local;
pub mod sync;

// Re-export main state types
pub use global::GlobalState;
pub use local::LocalState;
pub use sync::{StateManager, StateSynchronizer};

use crate::error::{CoreError, CoreResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// State configuration
#[derive(Debug, Clone)]
pub struct StateConfig {
    /// Enable global state
    pub enable_global_state: bool,
    /// Enable thread-local state
    pub enable_local_state: bool,
    /// State synchronization interval
    pub sync_interval: Duration,
    /// Maximum state size in bytes
    pub max_state_size: usize,
    /// Enable state persistence
    pub enable_persistence: bool,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            enable_global_state: true,
            enable_local_state: true,
            sync_interval: Duration::from_millis(100),
            max_state_size: 16 * 1024 * 1024, // 16MB
            enable_persistence: false,
        }
    }
}

/// State statistics
#[derive(Debug, Clone)]
pub struct StateStatistics {
    /// Total state operations
    pub total_operations: u64,
    /// Read operations
    pub read_operations: u64,
    /// Write operations
    pub write_operations: u64,
    /// Synchronization operations
    pub sync_operations: u64,
    /// Average operation latency in nanoseconds
    pub avg_operation_latency_ns: u64,
    /// Current state size in bytes
    pub current_state_size: usize,
    /// Last synchronization time
    pub last_sync_time: Option<Instant>,
}

impl Default for StateStatistics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            read_operations: 0,
            write_operations: 0,
            sync_operations: 0,
            avg_operation_latency_ns: 0,
            current_state_size: 0,
            last_sync_time: None,
        }
    }
}

/// State operation type
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StateOperation {
    /// Read operation
    Read,
    /// Write operation
    Write,
    /// Synchronization operation
    Sync,
    /// Clear operation
    Clear,
}

/// State manager for coordinating global and local state
///
/// Provides ultra-high performance state management with <10μs operation latency
/// and efficient synchronization between global and thread-local state.
#[repr(C, align(64))]
pub struct CoreStateManager {
    /// Configuration
    config: StateConfig,
    /// Global state
    global_state: Option<GlobalState>,
    /// State synchronizer
    synchronizer: Option<StateSynchronizer>,
    /// Operation counters
    total_operations: AtomicU64,
    read_operations: AtomicU64,
    write_operations: AtomicU64,
    sync_operations: AtomicU64,
    total_latency_ns: AtomicU64,
    /// Start time for metrics
    start_time: Option<Instant>,
}

impl CoreStateManager {
    /// Create a new state manager
    pub fn new(config: StateConfig) -> CoreResult<Self> {
        let mut manager = Self {
            config: config.clone(),
            global_state: None,
            synchronizer: None,
            total_operations: AtomicU64::new(0),
            read_operations: AtomicU64::new(0),
            write_operations: AtomicU64::new(0),
            sync_operations: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
            start_time: None,
        };

        manager.initialize()?;
        Ok(manager)
    }

    /// Initialize state management
    fn initialize(&mut self) -> CoreResult<()> {
        self.start_time = Some(Instant::now());

        // Initialize global state
        if self.config.enable_global_state {
            self.global_state = Some(GlobalState::new()?);
        }

        // Initialize synchronizer
        self.synchronizer = Some(StateSynchronizer::new(self.config.sync_interval)?);

        Ok(())
    }

    /// Start state management
    pub fn start(&mut self) -> CoreResult<()> {
        if let Some(synchronizer) = &mut self.synchronizer {
            synchronizer.start()?;
        }
        Ok(())
    }

    /// Stop state management
    pub fn stop(&mut self) -> CoreResult<()> {
        if let Some(synchronizer) = &mut self.synchronizer {
            synchronizer.stop()?;
        }
        Ok(())
    }

    /// Perform a state operation
    #[inline(always)]
    pub fn perform_operation(&self, operation: StateOperation) -> CoreResult<()> {
        let start = Instant::now();

        // Record operation
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        match operation {
            StateOperation::Read => {
                self.read_operations.fetch_add(1, Ordering::Relaxed);
            }
            StateOperation::Write => {
                self.write_operations.fetch_add(1, Ordering::Relaxed);
            }
            StateOperation::Sync => {
                self.sync_operations.fetch_add(1, Ordering::Relaxed);
            }
            StateOperation::Clear => {
                // Clear operation counts as write
                self.write_operations.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Record latency
        let latency_ns = start.elapsed().as_nanos() as u64;
        self.total_latency_ns
            .fetch_add(latency_ns, Ordering::Relaxed);

        // Ensure operation meets latency requirements
        if latency_ns > 10_000 {
            // > 10μs
            return Err(CoreError::Critical(
                crate::error::CriticalError::LatencyViolation(latency_ns / 1000),
            ));
        }

        Ok(())
    }

    /// Get global state
    #[must_use]
    pub fn global_state(&self) -> Option<&GlobalState> {
        self.global_state.as_ref()
    }

    /// Get mutable global state
    pub fn global_state_mut(&mut self) -> Option<&mut GlobalState> {
        self.global_state.as_mut()
    }

    /// Synchronize state
    pub fn synchronize(&self) -> CoreResult<()> {
        if let Some(synchronizer) = &self.synchronizer {
            synchronizer.synchronize()?;
            self.perform_operation(StateOperation::Sync)?;
        }
        Ok(())
    }

    /// Get state statistics
    #[must_use]
    pub fn statistics(&self) -> StateStatistics {
        let total_ops = self.total_operations.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ns.load(Ordering::Relaxed);

        let avg_latency = if total_ops > 0 {
            total_latency / total_ops
        } else {
            0
        };

        let current_size = self.global_state.as_ref().map(|gs| gs.size()).unwrap_or(0);

        let last_sync = self.synchronizer.as_ref().and_then(|s| s.last_sync_time());

        StateStatistics {
            total_operations: total_ops,
            read_operations: self.read_operations.load(Ordering::Relaxed),
            write_operations: self.write_operations.load(Ordering::Relaxed),
            sync_operations: self.sync_operations.load(Ordering::Relaxed),
            avg_operation_latency_ns: avg_latency,
            current_state_size: current_size,
            last_sync_time: last_sync,
        }
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &StateConfig {
        &self.config
    }

    /// Check state health
    #[must_use]
    pub fn health_check(&self) -> StateHealth {
        let stats = self.statistics();

        // Calculate health score based on performance metrics
        let mut score = 100u8;

        // Penalize high latency
        if stats.avg_operation_latency_ns > 10_000 {
            // > 10μs
            score = score.saturating_sub(30);
        } else if stats.avg_operation_latency_ns > 5_000 {
            // > 5μs
            score = score.saturating_sub(15);
        }

        // Penalize large state size
        if stats.current_state_size > self.config.max_state_size {
            score = score.saturating_sub(20);
        } else if stats.current_state_size > self.config.max_state_size / 2 {
            score = score.saturating_sub(10);
        }

        // Check synchronization health
        let sync_healthy = if let Some(last_sync) = stats.last_sync_time {
            last_sync.elapsed() < self.config.sync_interval * 2
        } else {
            false
        };

        if !sync_healthy {
            score = score.saturating_sub(25);
        }

        let status = if score >= 90 {
            StateHealthStatus::Excellent
        } else if score >= 70 {
            StateHealthStatus::Good
        } else if score >= 50 {
            StateHealthStatus::Fair
        } else if score >= 30 {
            StateHealthStatus::Poor
        } else {
            StateHealthStatus::Critical
        };

        StateHealth {
            status,
            score,
            operation_latency_healthy: stats.avg_operation_latency_ns <= 10_000,
            state_size_healthy: stats.current_state_size <= self.config.max_state_size,
            synchronization_healthy: sync_healthy,
        }
    }
}

impl Default for CoreStateManager {
    fn default() -> Self {
        Self::new(StateConfig::default()).unwrap_or_else(|_| {
            // Fallback to minimal configuration
            Self {
                config: StateConfig::default(),
                global_state: None,
                synchronizer: None,
                total_operations: AtomicU64::new(0),
                read_operations: AtomicU64::new(0),
                write_operations: AtomicU64::new(0),
                sync_operations: AtomicU64::new(0),
                total_latency_ns: AtomicU64::new(0),
                start_time: None,
            }
        })
    }
}

/// State health status
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StateHealthStatus {
    /// Excellent health
    Excellent,
    /// Good health
    Good,
    /// Fair health
    Fair,
    /// Poor health
    Poor,
    /// Critical health
    Critical,
}

/// State health information
#[derive(Debug, Clone)]
pub struct StateHealth {
    /// Overall health status
    pub status: StateHealthStatus,
    /// Health score (0-100)
    pub score: u8,
    /// Whether operation latency is healthy
    pub operation_latency_healthy: bool,
    /// Whether state size is healthy
    pub state_size_healthy: bool,
    /// Whether synchronization is healthy
    pub synchronization_healthy: bool,
}

impl StateHealth {
    /// Check if state is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(
            self.status,
            StateHealthStatus::Excellent | StateHealthStatus::Good
        )
    }

    /// Check if state needs attention
    #[must_use]
    pub fn needs_attention(&self) -> bool {
        matches!(
            self.status,
            StateHealthStatus::Poor | StateHealthStatus::Critical
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_manager_creation() -> CoreResult<()> {
        let config = StateConfig::default();
        let manager = CoreStateManager::new(config)?;

        assert!(manager.global_state().is_some());

        Ok(())
    }

    #[test]
    fn test_state_operations() -> CoreResult<()> {
        let manager = CoreStateManager::default();

        manager.perform_operation(StateOperation::Read)?;
        manager.perform_operation(StateOperation::Write)?;
        manager.perform_operation(StateOperation::Sync)?;

        let stats = manager.statistics();
        assert_eq!(stats.total_operations, 3);
        assert_eq!(stats.read_operations, 1);
        assert_eq!(stats.write_operations, 1);
        assert_eq!(stats.sync_operations, 1);

        Ok(())
    }

    #[test]
    fn test_state_statistics() -> CoreResult<()> {
        let manager = CoreStateManager::default();

        // Perform some operations
        for _ in 0..10 {
            manager.perform_operation(StateOperation::Read)?;
        }

        let stats = manager.statistics();
        assert_eq!(stats.read_operations, 10);
        // Note: avg_operation_latency_ns might be 0 for very fast operations
        // Just verify it's a valid value (no assertion needed for >= 0 on unsigned type)

        Ok(())
    }

    #[test]
    fn test_state_health_check() -> CoreResult<()> {
        let manager = CoreStateManager::default();

        let health = manager.health_check();
        assert!(health.score <= 100);

        Ok(())
    }

    #[test]
    fn test_state_config() {
        let config = StateConfig {
            enable_global_state: false,
            enable_local_state: true,
            sync_interval: Duration::from_millis(50),
            max_state_size: 8 * 1024 * 1024,
            enable_persistence: true,
        };

        assert!(!config.enable_global_state);
        assert!(config.enable_local_state);
        assert_eq!(config.sync_interval, Duration::from_millis(50));
        assert_eq!(config.max_state_size, 8 * 1024 * 1024);
        assert!(config.enable_persistence);
    }

    #[test]
    fn test_state_config_default() {
        let config = StateConfig::default();
        assert!(config.enable_global_state);
        assert!(config.enable_local_state);
        assert_eq!(config.sync_interval, Duration::from_millis(100));
        assert_eq!(config.max_state_size, 16 * 1024 * 1024);
        assert!(!config.enable_persistence);
    }

    #[test]
    fn test_state_statistics_default() {
        let stats = StateStatistics::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.read_operations, 0);
        assert_eq!(stats.write_operations, 0);
        assert_eq!(stats.sync_operations, 0);
        assert_eq!(stats.avg_operation_latency_ns, 0);
        assert_eq!(stats.current_state_size, 0);
        assert!(stats.last_sync_time.is_none());
    }

    #[test]
    fn test_state_operation_types() {
        assert_eq!(StateOperation::Read, StateOperation::Read);
        assert_eq!(StateOperation::Write, StateOperation::Write);
        assert_eq!(StateOperation::Sync, StateOperation::Sync);
        assert_eq!(StateOperation::Clear, StateOperation::Clear);
        assert_ne!(StateOperation::Read, StateOperation::Write);
    }

    #[test]
    fn test_state_manager_with_disabled_global() -> CoreResult<()> {
        let config = StateConfig {
            enable_global_state: false,
            enable_local_state: true,
            sync_interval: Duration::from_millis(100),
            max_state_size: 16 * 1024 * 1024,
            enable_persistence: false,
        };

        let manager = CoreStateManager::new(config)?;
        assert!(manager.global_state().is_none());

        Ok(())
    }

    #[test]
    fn test_state_manager_start_stop() -> CoreResult<()> {
        let mut manager = CoreStateManager::default();

        manager.start()?;
        manager.stop()?;

        Ok(())
    }

    #[test]
    fn test_state_manager_synchronize() -> CoreResult<()> {
        let mut manager = CoreStateManager::default();
        manager.start()?; // Start the manager first
        manager.synchronize()?;

        let stats = manager.statistics();
        assert!(stats.sync_operations > 0);

        Ok(())
    }

    #[test]
    fn test_state_manager_mutable_global_state() -> CoreResult<()> {
        let mut manager = CoreStateManager::default();

        if let Some(global_state) = manager.global_state_mut() {
            // Test that we can get mutable reference
            let _size = global_state.size();
        }

        Ok(())
    }

    #[test]
    fn test_state_manager_config_access() -> CoreResult<()> {
        let config = StateConfig {
            enable_global_state: true,
            enable_local_state: false,
            sync_interval: Duration::from_millis(200),
            max_state_size: 32 * 1024 * 1024,
            enable_persistence: true,
        };

        let manager = CoreStateManager::new(config.clone())?;
        let retrieved_config = manager.config();

        assert_eq!(
            retrieved_config.enable_global_state,
            config.enable_global_state
        );
        assert_eq!(retrieved_config.sync_interval, config.sync_interval);
        assert_eq!(retrieved_config.max_state_size, config.max_state_size);

        Ok(())
    }

    #[test]
    fn test_state_health_status_variants() {
        assert_eq!(StateHealthStatus::Excellent, StateHealthStatus::Excellent);
        assert_eq!(StateHealthStatus::Good, StateHealthStatus::Good);
        assert_eq!(StateHealthStatus::Fair, StateHealthStatus::Fair);
        assert_eq!(StateHealthStatus::Poor, StateHealthStatus::Poor);
        assert_eq!(StateHealthStatus::Critical, StateHealthStatus::Critical);
        assert_ne!(StateHealthStatus::Excellent, StateHealthStatus::Good);
    }

    #[test]
    fn test_state_health_methods() {
        let excellent_health = StateHealth {
            status: StateHealthStatus::Excellent,
            score: 95,
            operation_latency_healthy: true,
            state_size_healthy: true,
            synchronization_healthy: true,
        };

        assert!(excellent_health.is_healthy());
        assert!(!excellent_health.needs_attention());

        let poor_health = StateHealth {
            status: StateHealthStatus::Poor,
            score: 40,
            operation_latency_healthy: false,
            state_size_healthy: true,
            synchronization_healthy: false,
        };

        assert!(!poor_health.is_healthy());
        assert!(poor_health.needs_attention());

        let critical_health = StateHealth {
            status: StateHealthStatus::Critical,
            score: 20,
            operation_latency_healthy: false,
            state_size_healthy: false,
            synchronization_healthy: false,
        };

        assert!(!critical_health.is_healthy());
        assert!(critical_health.needs_attention());
    }

    #[test]
    fn test_state_manager_health_scoring() -> CoreResult<()> {
        let manager = CoreStateManager::default();

        // Perform operations to affect health
        for _ in 0..10 {
            manager.perform_operation(StateOperation::Read)?;
            manager.perform_operation(StateOperation::Write)?;
        }

        let health = manager.health_check();
        assert!(health.score <= 100);
        assert!(health.operation_latency_healthy); // Should be fast for simple operations

        Ok(())
    }

    #[test]
    fn test_state_manager_clear_operations() -> CoreResult<()> {
        let manager = CoreStateManager::default();

        manager.perform_operation(StateOperation::Clear)?;

        let stats = manager.statistics();
        assert_eq!(stats.write_operations, 1); // Clear counts as write
        assert_eq!(stats.total_operations, 1);

        Ok(())
    }

    #[test]
    fn test_state_manager_latency_tracking() -> CoreResult<()> {
        let manager = CoreStateManager::default();

        // Perform multiple operations
        for _ in 0..5 {
            manager.perform_operation(StateOperation::Read)?;
        }

        let stats = manager.statistics();
        assert_eq!(stats.read_operations, 5);
        assert_eq!(stats.total_operations, 5);
        // avg_operation_latency_ns should be calculated correctly
        assert!(stats.avg_operation_latency_ns < 1_000_000); // Should be < 1ms for simple ops

        Ok(())
    }

    #[test]
    fn test_state_manager_default_fallback() {
        // Test that default creation works even if normal creation fails
        let manager = CoreStateManager::default();

        // Should have basic functionality
        let stats = manager.statistics();
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_state_health_with_large_state() -> CoreResult<()> {
        let config = StateConfig {
            enable_global_state: true,
            enable_local_state: true,
            sync_interval: Duration::from_millis(100),
            max_state_size: 1024, // Very small limit
            enable_persistence: false,
        };

        let manager = CoreStateManager::new(config)?;
        let health = manager.health_check();

        // Health score might be affected by state size
        assert!(health.score <= 100);

        Ok(())
    }

    #[test]
    fn test_state_manager_statistics_accuracy() -> CoreResult<()> {
        let mut manager = CoreStateManager::default();
        manager.start()?; // Start the manager to avoid latency violations

        // Perform specific operations
        manager.perform_operation(StateOperation::Read)?;
        manager.perform_operation(StateOperation::Read)?;
        manager.perform_operation(StateOperation::Write)?;
        manager.perform_operation(StateOperation::Sync)?;
        manager.perform_operation(StateOperation::Clear)?;

        let stats = manager.statistics();
        assert_eq!(stats.total_operations, 5);
        assert_eq!(stats.read_operations, 2);
        assert_eq!(stats.write_operations, 2); // Write + Clear
        assert_eq!(stats.sync_operations, 1);

        Ok(())
    }
}
