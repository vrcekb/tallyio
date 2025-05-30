//! State synchronization for TallyIO core
//!
//! This module provides synchronization between global and thread-local state
//! to ensure consistency while maintaining ultra-high performance.

use crate::error::{CoreError, CoreResult};
use crate::state::{GlobalState, LocalState};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// State synchronization configuration
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Synchronization interval
    pub sync_interval: Duration,
    /// Enable bidirectional sync
    pub bidirectional: bool,
    /// Sync batch size
    pub batch_size: usize,
    /// Enable sync statistics
    pub enable_statistics: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            sync_interval: Duration::from_millis(100),
            bidirectional: true,
            batch_size: 100,
            enable_statistics: true,
        }
    }
}

/// State synchronizer
///
/// Coordinates synchronization between global and thread-local state
/// to maintain consistency while preserving performance.
#[derive(Debug)]
#[repr(C, align(64))]
pub struct StateSynchronizer {
    /// Configuration
    config: SyncConfig,
    /// Running flag
    is_running: AtomicBool,
    /// Synchronization statistics
    sync_count: AtomicU64,
    sync_errors: AtomicU64,
    last_sync_time: AtomicU64,
    total_sync_time_ns: AtomicU64,
}

impl StateSynchronizer {
    /// Create a new state synchronizer
    pub fn new(sync_interval: Duration) -> CoreResult<Self> {
        let config = SyncConfig {
            sync_interval,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new state synchronizer with configuration
    pub fn with_config(config: SyncConfig) -> CoreResult<Self> {
        Ok(Self {
            config,
            is_running: AtomicBool::new(false),
            sync_count: AtomicU64::new(0),
            sync_errors: AtomicU64::new(0),
            last_sync_time: AtomicU64::new(0),
            total_sync_time_ns: AtomicU64::new(0),
        })
    }

    /// Start the synchronizer
    pub fn start(&mut self) -> CoreResult<()> {
        self.is_running.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop the synchronizer
    pub fn stop(&mut self) -> CoreResult<()> {
        self.is_running.store(false, Ordering::Release);
        Ok(())
    }

    /// Check if synchronizer is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Perform synchronization
    pub fn synchronize(&self) -> CoreResult<()> {
        if !self.is_running() {
            return Err(CoreError::state("Synchronizer is not running"));
        }

        let start = Instant::now();

        // Simulate synchronization work
        // In a real implementation, this would:
        // 1. Collect changes from thread-local states
        // 2. Apply changes to global state
        // 3. Distribute global changes to thread-local states
        // 4. Handle conflicts and ensure consistency

        // Record synchronization
        self.sync_count.fetch_add(1, Ordering::Relaxed);

        let sync_time_ns = start.elapsed().as_nanos() as u64;
        self.total_sync_time_ns
            .fetch_add(sync_time_ns, Ordering::Relaxed);

        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_sync_time.store(now_millis, Ordering::Relaxed);

        Ok(())
    }

    /// Synchronize specific local state with global state
    pub fn sync_local_to_global(
        &self,
        _local_state: &LocalState,
        _global_state: &GlobalState,
    ) -> CoreResult<SyncResult> {
        if !self.is_running() {
            return Err(CoreError::state("Synchronizer is not running"));
        }

        let start = Instant::now();

        // Simulate synchronization from local to global
        // In a real implementation, this would:
        // 1. Extract new/modified data from local state
        // 2. Apply changes to global state
        // 3. Handle conflicts
        // 4. Update local state with any global changes

        let sync_time = start.elapsed();

        Ok(SyncResult {
            items_synced: 0,
            conflicts_resolved: 0,
            sync_time,
            success: true,
        })
    }

    /// Synchronize global state to specific local state
    pub fn sync_global_to_local(
        &self,
        _global_state: &GlobalState,
        _local_state: &LocalState,
    ) -> CoreResult<SyncResult> {
        if !self.is_running() {
            return Err(CoreError::state("Synchronizer is not running"));
        }

        let start = Instant::now();

        // Simulate synchronization from global to local
        // In a real implementation, this would:
        // 1. Extract relevant changes from global state
        // 2. Apply changes to local state
        // 3. Handle cache evictions if needed
        // 4. Update local statistics

        let sync_time = start.elapsed();

        Ok(SyncResult {
            items_synced: 0,
            conflicts_resolved: 0,
            sync_time,
            success: true,
        })
    }

    /// Get last synchronization time
    #[must_use]
    pub fn last_sync_time(&self) -> Option<Instant> {
        let millis = self.last_sync_time.load(Ordering::Relaxed);
        if millis == 0 {
            None
        } else {
            // Convert from stored milliseconds back to Instant
            // This is approximate since we can't perfectly reconstruct the original Instant
            let now = Instant::now();
            let stored_system_time = std::time::UNIX_EPOCH + Duration::from_millis(millis);
            let current_system_time = std::time::SystemTime::now();

            if let Ok(elapsed) = current_system_time.duration_since(stored_system_time) {
                Some(now - elapsed)
            } else {
                Some(now) // Fallback to current time
            }
        }
    }

    /// Get synchronization statistics
    #[must_use]
    pub fn statistics(&self) -> SyncStatistics {
        let sync_count = self.sync_count.load(Ordering::Relaxed);
        let total_time = self.total_sync_time_ns.load(Ordering::Relaxed);
        let avg_time = if sync_count > 0 {
            total_time / sync_count
        } else {
            0
        };

        SyncStatistics {
            sync_count,
            sync_errors: self.sync_errors.load(Ordering::Relaxed),
            average_sync_time_ns: avg_time,
            last_sync_time: self.last_sync_time(),
            is_running: self.is_running(),
        }
    }

    /// Record a synchronization error
    #[inline(always)]
    pub fn record_error(&self) {
        self.sync_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &SyncConfig {
        &self.config
    }

    /// Check if synchronization is needed
    #[must_use]
    pub fn needs_sync(&self) -> bool {
        if let Some(last_sync) = self.last_sync_time() {
            Instant::now().duration_since(last_sync) >= self.config.sync_interval
        } else {
            true // Never synced before
        }
    }
}

/// State manager that coordinates global and local state
#[derive(Debug)]
pub struct StateManager {
    /// Global state
    global_state: Arc<GlobalState>,
    /// State synchronizer
    synchronizer: StateSynchronizer,
}

impl StateManager {
    /// Create a new state manager
    pub fn new() -> CoreResult<Self> {
        let global_state = Arc::new(GlobalState::new()?);
        let synchronizer = StateSynchronizer::new(Duration::from_millis(100))?;

        Ok(Self {
            global_state,
            synchronizer,
        })
    }

    /// Create a new state manager with configuration
    pub fn with_config(sync_config: SyncConfig) -> CoreResult<Self> {
        let global_state = Arc::new(GlobalState::new()?);
        let synchronizer = StateSynchronizer::with_config(sync_config)?;

        Ok(Self {
            global_state,
            synchronizer,
        })
    }

    /// Start the state manager
    pub fn start(&mut self) -> CoreResult<()> {
        self.synchronizer.start()
    }

    /// Stop the state manager
    pub fn stop(&mut self) -> CoreResult<()> {
        self.synchronizer.stop()
    }

    /// Get global state
    #[must_use]
    pub fn global_state(&self) -> &Arc<GlobalState> {
        &self.global_state
    }

    /// Create a new local state
    #[must_use]
    pub fn create_local_state(&self) -> LocalState {
        LocalState::new()
    }

    /// Synchronize a local state with global state
    pub fn sync_local_state(&self, local_state: &LocalState) -> CoreResult<SyncResult> {
        if self.synchronizer.config.bidirectional {
            // Sync both directions
            let _to_global = self
                .synchronizer
                .sync_local_to_global(local_state, &self.global_state)?;
            let from_global = self
                .synchronizer
                .sync_global_to_local(&self.global_state, local_state)?;
            Ok(from_global)
        } else {
            // Only sync from global to local
            self.synchronizer
                .sync_global_to_local(&self.global_state, local_state)
        }
    }

    /// Perform global synchronization
    pub fn synchronize(&self) -> CoreResult<()> {
        self.synchronizer.synchronize()
    }

    /// Get synchronization statistics
    #[must_use]
    pub fn sync_statistics(&self) -> SyncStatistics {
        self.synchronizer.statistics()
    }

    /// Check if synchronization is needed
    #[must_use]
    pub fn needs_sync(&self) -> bool {
        self.synchronizer.needs_sync()
    }
}

impl Default for StateManager {
    fn default() -> Self {
        // Try to create normally first
        if let Ok(manager) = Self::new() {
            return manager;
        }

        // Fallback: create minimal working implementation
        // This should only happen in extreme error conditions
        let global_state = Arc::new(GlobalState::new().unwrap_or_else(|_| {
            // Create a minimal global state that won't fail
            // In practice, this fallback should never be needed
            std::process::abort();
        }));

        let synchronizer =
            StateSynchronizer::new(Duration::from_millis(100)).unwrap_or_else(|_| {
                // Create a minimal synchronizer that won't fail
                // In practice, this fallback should never be needed
                std::process::abort();
            });

        Self {
            global_state,
            synchronizer,
        }
    }
}

/// Result of a synchronization operation
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of items synchronized
    pub items_synced: usize,
    /// Number of conflicts resolved
    pub conflicts_resolved: usize,
    /// Time taken for synchronization
    pub sync_time: Duration,
    /// Whether synchronization was successful
    pub success: bool,
}

/// Synchronization statistics
#[derive(Debug, Clone)]
pub struct SyncStatistics {
    /// Total number of synchronizations performed
    pub sync_count: u64,
    /// Number of synchronization errors
    pub sync_errors: u64,
    /// Average synchronization time in nanoseconds
    pub average_sync_time_ns: u64,
    /// Last synchronization time
    pub last_sync_time: Option<Instant>,
    /// Whether synchronizer is running
    pub is_running: bool,
}

impl SyncStatistics {
    /// Get synchronization error rate
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        if self.sync_count == 0 {
            0.0
        } else {
            self.sync_errors as f64 / self.sync_count as f64
        }
    }

    /// Get average synchronization time in milliseconds
    #[must_use]
    pub fn average_sync_time_ms(&self) -> f64 {
        self.average_sync_time_ns as f64 / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchronizer_creation() -> CoreResult<()> {
        let sync = StateSynchronizer::new(Duration::from_millis(100))?;
        assert!(!sync.is_running());
        Ok(())
    }

    #[test]
    fn test_synchronizer_start_stop() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(100))?;

        sync.start()?;
        assert!(sync.is_running());

        sync.stop()?;
        assert!(!sync.is_running());

        Ok(())
    }

    #[test]
    fn test_synchronization() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(100))?;
        sync.start()?;

        sync.synchronize()?;

        let stats = sync.statistics();
        assert_eq!(stats.sync_count, 1);
        assert!(stats.last_sync_time.is_some());

        Ok(())
    }

    #[test]
    fn test_state_manager() -> CoreResult<()> {
        let mut manager = StateManager::new()?;
        manager.start()?;

        let local_state = manager.create_local_state();
        let _result = manager.sync_local_state(&local_state)?;

        manager.stop()?;
        Ok(())
    }

    #[test]
    fn test_sync_statistics() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(100))?;
        sync.start()?;

        // Perform some synchronizations
        for _ in 0..3 {
            sync.synchronize()?;
        }

        let stats = sync.statistics();
        assert_eq!(stats.sync_count, 3);
        assert_eq!(stats.error_rate(), 0.0);

        Ok(())
    }

    #[test]
    fn test_needs_sync() -> CoreResult<()> {
        let sync = StateSynchronizer::new(Duration::from_millis(1))?; // Very short interval

        assert!(sync.needs_sync()); // Never synced before

        Ok(())
    }

    #[test]
    fn test_sync_config_default() {
        let config = SyncConfig::default();
        assert_eq!(config.sync_interval, Duration::from_millis(100));
        assert!(config.bidirectional);
        assert_eq!(config.batch_size, 100);
        assert!(config.enable_statistics);
    }

    #[test]
    fn test_sync_config_custom() {
        let config = SyncConfig {
            sync_interval: Duration::from_millis(50),
            bidirectional: false,
            batch_size: 200,
            enable_statistics: false,
        };

        assert_eq!(config.sync_interval, Duration::from_millis(50));
        assert!(!config.bidirectional);
        assert_eq!(config.batch_size, 200);
        assert!(!config.enable_statistics);
    }

    #[test]
    fn test_synchronizer_with_config() -> CoreResult<()> {
        let config = SyncConfig {
            sync_interval: Duration::from_millis(200),
            bidirectional: false,
            batch_size: 50,
            enable_statistics: true,
        };

        let sync = StateSynchronizer::with_config(config.clone())?;
        let retrieved_config = sync.config();

        assert_eq!(retrieved_config.sync_interval, config.sync_interval);
        assert_eq!(retrieved_config.bidirectional, config.bidirectional);
        assert_eq!(retrieved_config.batch_size, config.batch_size);
        assert_eq!(retrieved_config.enable_statistics, config.enable_statistics);

        Ok(())
    }

    #[test]
    fn test_synchronizer_not_running_operations() -> CoreResult<()> {
        let sync = StateSynchronizer::new(Duration::from_millis(100))?;

        // Should fail when not running
        assert!(sync.synchronize().is_err());

        let local_state = LocalState::new();
        let global_state = GlobalState::new()?;

        assert!(sync
            .sync_local_to_global(&local_state, &global_state)
            .is_err());
        assert!(sync
            .sync_global_to_local(&global_state, &local_state)
            .is_err());

        Ok(())
    }

    #[test]
    fn test_synchronizer_local_to_global() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(100))?;
        sync.start()?;

        let local_state = LocalState::new();
        let global_state = GlobalState::new()?;

        let result = sync.sync_local_to_global(&local_state, &global_state)?;
        assert!(result.success);
        assert_eq!(result.items_synced, 0); // Simulation returns 0
        assert_eq!(result.conflicts_resolved, 0);

        Ok(())
    }

    #[test]
    fn test_synchronizer_global_to_local() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(100))?;
        sync.start()?;

        let local_state = LocalState::new();
        let global_state = GlobalState::new()?;

        let result = sync.sync_global_to_local(&global_state, &local_state)?;
        assert!(result.success);
        assert_eq!(result.items_synced, 0); // Simulation returns 0
        assert_eq!(result.conflicts_resolved, 0);

        Ok(())
    }

    #[test]
    fn test_synchronizer_error_recording() -> CoreResult<()> {
        let sync = StateSynchronizer::new(Duration::from_millis(100))?;

        // Record some errors
        sync.record_error();
        sync.record_error();
        sync.record_error();

        let stats = sync.statistics();
        assert_eq!(stats.sync_errors, 3);

        Ok(())
    }

    #[test]
    fn test_synchronizer_last_sync_time() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(100))?;
        sync.start()?;

        // Initially no sync time
        assert!(sync.last_sync_time().is_none());

        // After sync, should have time
        sync.synchronize()?;
        assert!(sync.last_sync_time().is_some());

        Ok(())
    }

    #[test]
    fn test_synchronizer_needs_sync_after_sync() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(1000))?; // Long interval
        sync.start()?;

        assert!(sync.needs_sync()); // Never synced

        sync.synchronize()?;
        assert!(!sync.needs_sync()); // Just synced, shouldn't need sync yet

        Ok(())
    }

    #[test]
    fn test_state_manager_with_config() -> CoreResult<()> {
        let config = SyncConfig {
            sync_interval: Duration::from_millis(50),
            bidirectional: false,
            batch_size: 25,
            enable_statistics: false,
        };

        let mut manager = StateManager::with_config(config)?;
        manager.start()?; // Start the synchronizer
        let local_state = manager.create_local_state();

        // Test sync with unidirectional config
        let _result = manager.sync_local_state(&local_state)?;

        Ok(())
    }

    #[test]
    fn test_state_manager_bidirectional_sync() -> CoreResult<()> {
        let config = SyncConfig {
            sync_interval: Duration::from_millis(100),
            bidirectional: true,
            batch_size: 100,
            enable_statistics: true,
        };

        let mut manager = StateManager::with_config(config)?;
        manager.start()?; // Start the synchronizer
        let local_state = manager.create_local_state();

        // Test bidirectional sync
        let result = manager.sync_local_state(&local_state)?;
        assert!(result.success);

        Ok(())
    }

    #[test]
    fn test_state_manager_global_state_access() -> CoreResult<()> {
        let manager = StateManager::new()?;
        let global_state = manager.global_state();

        // Should be able to access global state
        let _size = global_state.size();

        Ok(())
    }

    #[test]
    fn test_state_manager_synchronize() -> CoreResult<()> {
        let mut manager = StateManager::new()?;
        manager.start()?; // Start the synchronizer
        manager.synchronize()?;

        let stats = manager.sync_statistics();
        assert_eq!(stats.sync_count, 1);

        Ok(())
    }

    #[test]
    fn test_state_manager_needs_sync() -> CoreResult<()> {
        let mut manager = StateManager::new()?;
        manager.start()?; // Start the synchronizer

        assert!(manager.needs_sync()); // Never synced

        manager.synchronize()?;
        // Might or might not need sync depending on timing

        Ok(())
    }

    #[test]
    fn test_state_manager_default() {
        let manager = StateManager::default();
        let _global_state = manager.global_state();
        let _local_state = manager.create_local_state();
    }

    #[test]
    fn test_sync_result_fields() {
        let result = SyncResult {
            items_synced: 42,
            conflicts_resolved: 3,
            sync_time: Duration::from_millis(10),
            success: true,
        };

        assert_eq!(result.items_synced, 42);
        assert_eq!(result.conflicts_resolved, 3);
        assert_eq!(result.sync_time, Duration::from_millis(10));
        assert!(result.success);
    }

    #[test]
    fn test_sync_statistics_methods() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(100))?;
        sync.start()?;

        // Record some errors
        sync.record_error();
        sync.record_error();

        // Perform some syncs
        sync.synchronize()?;
        sync.synchronize()?;
        sync.synchronize()?;

        let stats = sync.statistics();
        assert_eq!(stats.sync_count, 3);
        assert_eq!(stats.sync_errors, 2);
        assert_eq!(stats.error_rate(), 2.0 / 3.0);
        assert!(stats.average_sync_time_ms() >= 0.0);
        assert!(stats.is_running);

        Ok(())
    }

    #[test]
    fn test_sync_statistics_error_rate_edge_cases() {
        let stats = SyncStatistics {
            sync_count: 0,
            sync_errors: 0,
            average_sync_time_ns: 0,
            last_sync_time: None,
            is_running: false,
        };

        assert_eq!(stats.error_rate(), 0.0); // No syncs, no errors

        let stats_with_errors = SyncStatistics {
            sync_count: 0,
            sync_errors: 5,
            average_sync_time_ns: 0,
            last_sync_time: None,
            is_running: false,
        };

        assert_eq!(stats_with_errors.error_rate(), 0.0); // No syncs, so rate is 0
    }

    #[test]
    fn test_sync_statistics_time_conversion() {
        let stats = SyncStatistics {
            sync_count: 1,
            sync_errors: 0,
            average_sync_time_ns: 1_500_000, // 1.5 ms in nanoseconds
            last_sync_time: None,
            is_running: true,
        };

        assert_eq!(stats.average_sync_time_ms(), 1.5);
    }

    #[test]
    fn test_synchronizer_multiple_start_stop() -> CoreResult<()> {
        let mut sync = StateSynchronizer::new(Duration::from_millis(100))?;

        // Multiple start/stop cycles
        sync.start()?;
        assert!(sync.is_running());

        sync.stop()?;
        assert!(!sync.is_running());

        sync.start()?;
        assert!(sync.is_running());

        sync.stop()?;
        assert!(!sync.is_running());

        Ok(())
    }

    #[test]
    fn test_state_manager_start_stop() -> CoreResult<()> {
        let mut manager = StateManager::new()?;

        manager.start()?;
        manager.stop()?;

        Ok(())
    }
}
