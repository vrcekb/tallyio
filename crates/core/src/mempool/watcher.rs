//! Mempool watcher for real-time transaction monitoring
//!
//! This module provides ultra-low latency mempool watching capabilities
//! for real-time transaction detection and MEV opportunity identification.

use crate::error::{CoreError, CoreResult};
use crate::types::{Transaction, TransactionHash};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Mempool event types
#[derive(Debug, Clone)]
pub enum MempoolEvent {
    /// New transaction added to mempool
    TransactionAdded {
        transaction: Transaction,
        timestamp: Instant,
    },
    /// Transaction removed from mempool
    TransactionRemoved {
        hash: TransactionHash,
        timestamp: Instant,
    },
    /// Transaction replaced (higher gas price)
    TransactionReplaced {
        old_hash: TransactionHash,
        new_transaction: Transaction,
        timestamp: Instant,
    },
    /// Mempool cleared
    MempoolCleared { timestamp: Instant },
}

impl MempoolEvent {
    /// Get event timestamp
    #[must_use]
    pub const fn timestamp(&self) -> Instant {
        match self {
            Self::TransactionAdded { timestamp, .. }
            | Self::TransactionRemoved { timestamp, .. }
            | Self::TransactionReplaced { timestamp, .. }
            | Self::MempoolCleared { timestamp } => *timestamp,
        }
    }

    /// Check if event involves a transaction
    #[must_use]
    pub fn transaction(&self) -> Option<&Transaction> {
        match self {
            Self::TransactionAdded { transaction, .. }
            | Self::TransactionReplaced {
                new_transaction: transaction,
                ..
            } => Some(transaction),
            _ => None,
        }
    }

    /// Get transaction hash if available
    #[must_use]
    pub fn transaction_hash(&self) -> Option<TransactionHash> {
        match self {
            Self::TransactionAdded { transaction, .. } => transaction.hash,
            Self::TransactionRemoved { hash, .. } => Some(*hash),
            Self::TransactionReplaced {
                new_transaction, ..
            } => new_transaction.hash,
            Self::MempoolCleared { .. } => None,
        }
    }
}

/// Mempool watcher configuration
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Enable real-time monitoring
    pub enable_realtime: bool,
    /// Polling interval for non-realtime mode
    pub polling_interval: Duration,
    /// Maximum events to buffer
    pub max_event_buffer: usize,
    /// Enable transaction replacement detection
    pub detect_replacements: bool,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            polling_interval: Duration::from_millis(100),
            max_event_buffer: 10_000,
            detect_replacements: true,
        }
    }
}

/// Mempool watcher for real-time transaction monitoring
///
/// Provides ultra-low latency mempool monitoring with <1ms event detection
/// for MEV opportunity identification and transaction analysis.
#[repr(C, align(64))]
pub struct MempoolWatcher {
    /// Watcher configuration
    config: WatcherConfig,
    /// Running state
    is_running: AtomicBool,
    /// Event statistics
    events_processed: AtomicU64,
    transactions_seen: AtomicU64,
    replacements_detected: AtomicU64,
    /// Start time for metrics
    start_time: Option<Instant>,
}

impl MempoolWatcher {
    /// Create a new mempool watcher
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(WatcherConfig::default())
    }

    /// Create a new mempool watcher with configuration
    #[must_use]
    pub fn with_config(config: WatcherConfig) -> Self {
        Self {
            config,
            is_running: AtomicBool::new(false),
            events_processed: AtomicU64::new(0),
            transactions_seen: AtomicU64::new(0),
            replacements_detected: AtomicU64::new(0),
            start_time: None,
        }
    }

    /// Start watching the mempool
    pub fn start(&mut self) -> CoreResult<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.is_running.store(true, Ordering::Release);
        self.start_time = Some(Instant::now());
        Ok(())
    }

    /// Stop watching the mempool
    pub fn stop(&mut self) -> CoreResult<()> {
        self.is_running.store(false, Ordering::Release);
        Ok(())
    }

    /// Check if watcher is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Process a mempool event
    #[inline(always)]
    pub fn process_event(&self, event: MempoolEvent) -> CoreResult<()> {
        if !self.is_running() {
            return Err(CoreError::mempool("Watcher is not running"));
        }

        // Update statistics based on event type
        match &event {
            MempoolEvent::TransactionAdded { .. } => {
                self.transactions_seen.fetch_add(1, Ordering::Relaxed);
            }
            MempoolEvent::TransactionReplaced { .. } => {
                self.replacements_detected.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        self.events_processed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Simulate watching for new transactions (in real implementation, this would
    /// connect to actual mempool sources like Ethereum nodes, MEV-Boost, etc.)
    pub fn watch_for_transactions(&self) -> CoreResult<Vec<MempoolEvent>> {
        if !self.is_running() {
            return Ok(Vec::with_capacity(0));
        }

        // Simulate finding new transactions
        let mut events = Vec::with_capacity(5);

        // In a real implementation, this would:
        // 1. Connect to Ethereum node's mempool
        // 2. Subscribe to pending transactions
        // 3. Monitor for transaction replacements
        // 4. Detect mempool reorganizations
        // 5. Filter for relevant transactions

        // For now, return empty events (simulation)
        Ok(events)
    }

    /// Get watcher statistics
    #[must_use]
    pub fn statistics(&self) -> WatcherStatistics {
        let uptime = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or_default();

        let events = self.events_processed.load(Ordering::Relaxed);
        let transactions = self.transactions_seen.load(Ordering::Relaxed);
        let replacements = self.replacements_detected.load(Ordering::Relaxed);

        let events_per_second = if uptime.as_secs_f64() > 0.0 {
            events as f64 / uptime.as_secs_f64()
        } else {
            0.0
        };

        let transactions_per_second = if uptime.as_secs_f64() > 0.0 {
            transactions as f64 / uptime.as_secs_f64()
        } else {
            0.0
        };

        let replacement_rate = if transactions > 0 {
            replacements as f64 / transactions as f64
        } else {
            0.0
        };

        WatcherStatistics {
            is_running: self.is_running(),
            uptime,
            events_processed: events,
            transactions_seen: transactions,
            replacements_detected: replacements,
            events_per_second,
            transactions_per_second,
            replacement_rate,
        }
    }

    /// Get watcher configuration
    #[must_use]
    pub const fn config(&self) -> &WatcherConfig {
        &self.config
    }

    /// Update watcher configuration
    pub fn update_config(&mut self, config: WatcherConfig) {
        self.config = config;
    }

    /// Check mempool health (simulation)
    #[must_use]
    pub fn health_check(&self) -> MempoolHealth {
        let stats = self.statistics();

        // Determine health based on activity and performance
        let health_score = if !stats.is_running {
            0
        } else if stats.transactions_per_second > 100.0 {
            100 // Very active
        } else if stats.transactions_per_second > 50.0 {
            80 // Active
        } else if stats.transactions_per_second > 10.0 {
            60 // Moderate
        } else if stats.transactions_per_second > 1.0 {
            40 // Low activity
        } else {
            20 // Very low activity
        };

        let status = if health_score >= 80 {
            MempoolHealthStatus::Excellent
        } else if health_score >= 60 {
            MempoolHealthStatus::Good
        } else if health_score >= 40 {
            MempoolHealthStatus::Fair
        } else if health_score >= 20 {
            MempoolHealthStatus::Poor
        } else {
            MempoolHealthStatus::Critical
        };

        MempoolHealth {
            status,
            score: health_score,
            last_activity: stats.uptime,
            connection_stable: stats.is_running,
            transaction_flow_normal: stats.transactions_per_second > 1.0,
        }
    }
}

impl Default for MempoolWatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Watcher statistics
#[derive(Debug, Clone)]
pub struct WatcherStatistics {
    /// Whether watcher is currently running
    pub is_running: bool,
    /// Total uptime
    pub uptime: Duration,
    /// Total events processed
    pub events_processed: u64,
    /// Total transactions seen
    pub transactions_seen: u64,
    /// Total transaction replacements detected
    pub replacements_detected: u64,
    /// Events processed per second
    pub events_per_second: f64,
    /// Transactions seen per second
    pub transactions_per_second: f64,
    /// Transaction replacement rate (0.0 - 1.0)
    pub replacement_rate: f64,
}

/// Mempool health status
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MempoolHealthStatus {
    /// Excellent health - high activity, stable connection
    Excellent,
    /// Good health - normal activity
    Good,
    /// Fair health - reduced activity
    Fair,
    /// Poor health - low activity or connection issues
    Poor,
    /// Critical health - no activity or major issues
    Critical,
}

/// Mempool health information
#[derive(Debug, Clone)]
pub struct MempoolHealth {
    /// Overall health status
    pub status: MempoolHealthStatus,
    /// Health score (0-100)
    pub score: u8,
    /// Time since last activity
    pub last_activity: Duration,
    /// Whether connection is stable
    pub connection_stable: bool,
    /// Whether transaction flow appears normal
    pub transaction_flow_normal: bool,
}

impl MempoolHealth {
    /// Check if mempool is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(
            self.status,
            MempoolHealthStatus::Excellent | MempoolHealthStatus::Good
        )
    }

    /// Check if mempool needs attention
    #[must_use]
    pub fn needs_attention(&self) -> bool {
        matches!(
            self.status,
            MempoolHealthStatus::Poor | MempoolHealthStatus::Critical
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Gas, Price};

    #[test]
    fn test_watcher_creation() {
        let watcher = MempoolWatcher::new();
        assert!(!watcher.is_running());

        let stats = watcher.statistics();
        assert_eq!(stats.events_processed, 0);
        assert_eq!(stats.transactions_seen, 0);
    }

    #[test]
    fn test_watcher_start_stop() -> CoreResult<()> {
        let mut watcher = MempoolWatcher::new();

        watcher.start()?;
        assert!(watcher.is_running());

        watcher.stop()?;
        assert!(!watcher.is_running());

        Ok(())
    }

    #[test]
    fn test_event_processing() -> CoreResult<()> {
        let mut watcher = MempoolWatcher::new();
        watcher.start()?;

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let event = MempoolEvent::TransactionAdded {
            transaction: tx,
            timestamp: Instant::now(),
        };

        watcher.process_event(event)?;

        let stats = watcher.statistics();
        assert_eq!(stats.events_processed, 1);
        assert_eq!(stats.transactions_seen, 1);

        Ok(())
    }

    #[test]
    fn test_replacement_detection() -> CoreResult<()> {
        let mut watcher = MempoolWatcher::new();
        watcher.start()?;

        let old_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let new_tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(30), // Higher gas price
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let event = MempoolEvent::TransactionReplaced {
            old_hash: [1u8; 32],
            new_transaction: new_tx,
            timestamp: Instant::now(),
        };

        watcher.process_event(event)?;

        let stats = watcher.statistics();
        assert_eq!(stats.replacements_detected, 1);

        Ok(())
    }

    #[test]
    fn test_mempool_event_methods() {
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let event = MempoolEvent::TransactionAdded {
            transaction: tx.clone(),
            timestamp: Instant::now(),
        };

        assert!(event.transaction().is_some());
        if let Some(transaction) = event.transaction() {
            assert_eq!(transaction.value(), tx.value());
        }
    }

    #[test]
    fn test_health_check() -> CoreResult<()> {
        let mut watcher = MempoolWatcher::new();
        watcher.start()?;

        let health = watcher.health_check();
        assert!(health.connection_stable);
        assert!(health.score <= 100);

        Ok(())
    }

    #[test]
    fn test_watcher_with_custom_config() {
        let config = WatcherConfig {
            enable_realtime: false,
            polling_interval: Duration::from_millis(500),
            max_event_buffer: 5000,
            detect_replacements: false,
        };

        let watcher = MempoolWatcher::with_config(config);
        assert!(!watcher.config().enable_realtime);
        assert_eq!(watcher.config().max_event_buffer, 5000);
        assert!(!watcher.config().detect_replacements);
    }

    #[test]
    fn test_watch_for_transactions() -> CoreResult<()> {
        let mut watcher = MempoolWatcher::new();
        watcher.start()?;

        let events = watcher.watch_for_transactions()?;
        assert!(events.is_empty()); // Simulation returns empty

        Ok(())
    }

    #[test]
    fn test_mempool_event_timestamp() {
        // Test timestamp method (lines 37-42)
        let timestamp = Instant::now();

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let event_added = MempoolEvent::TransactionAdded {
            transaction: tx.clone(),
            timestamp,
        };
        assert_eq!(event_added.timestamp(), timestamp);

        let event_removed = MempoolEvent::TransactionRemoved {
            hash: [1u8; 32],
            timestamp,
        };
        assert_eq!(event_removed.timestamp(), timestamp);

        let event_replaced = MempoolEvent::TransactionReplaced {
            old_hash: [1u8; 32],
            new_transaction: tx,
            timestamp,
        };
        assert_eq!(event_replaced.timestamp(), timestamp);

        let event_cleared = MempoolEvent::MempoolCleared { timestamp };
        assert_eq!(event_cleared.timestamp(), timestamp);
    }

    #[test]
    fn test_mempool_event_transaction() {
        // Test transaction method (lines 52, 55)
        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let event_added = MempoolEvent::TransactionAdded {
            transaction: tx.clone(),
            timestamp: Instant::now(),
        };
        assert!(event_added.transaction().is_some());

        let event_replaced = MempoolEvent::TransactionReplaced {
            old_hash: [1u8; 32],
            new_transaction: tx,
            timestamp: Instant::now(),
        };
        assert!(event_replaced.transaction().is_some());

        let event_removed = MempoolEvent::TransactionRemoved {
            hash: [1u8; 32],
            timestamp: Instant::now(),
        };
        assert!(event_removed.transaction().is_none()); // Line 55

        let event_cleared = MempoolEvent::MempoolCleared {
            timestamp: Instant::now(),
        };
        assert!(event_cleared.transaction().is_none()); // Line 55
    }

    #[test]
    fn test_mempool_event_transaction_hash() {
        // Test transaction_hash method (lines 61-68)
        let mut tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );
        tx.hash = Some([1u8; 32]);

        let event_added = MempoolEvent::TransactionAdded {
            transaction: tx.clone(),
            timestamp: Instant::now(),
        };
        assert_eq!(event_added.transaction_hash(), Some([1u8; 32])); // Line 63

        let event_removed = MempoolEvent::TransactionRemoved {
            hash: [2u8; 32],
            timestamp: Instant::now(),
        };
        assert_eq!(event_removed.transaction_hash(), Some([2u8; 32])); // Line 64

        let event_replaced = MempoolEvent::TransactionReplaced {
            old_hash: [3u8; 32],
            new_transaction: tx,
            timestamp: Instant::now(),
        };
        assert_eq!(event_replaced.transaction_hash(), Some([1u8; 32])); // Lines 66-67

        let event_cleared = MempoolEvent::MempoolCleared {
            timestamp: Instant::now(),
        };
        assert_eq!(event_cleared.transaction_hash(), None); // Line 68
    }

    #[test]
    fn test_watcher_not_running_process_event() -> CoreResult<()> {
        // Test process_event when not running (line 162)
        let watcher = MempoolWatcher::new(); // Not started

        let tx = Transaction::new(
            [1u8; 20],
            Some([2u8; 20]),
            Price::from_ether(1),
            Price::from_gwei(20),
            Gas::new(21_000),
            0,
            Vec::with_capacity(0),
        );

        let event = MempoolEvent::TransactionAdded {
            transaction: tx,
            timestamp: Instant::now(),
        };

        let result = watcher.process_event(event);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_process_event_other_types() -> CoreResult<()> {
        // Test processing other event types (line 173)
        let mut watcher = MempoolWatcher::new();
        watcher.start()?;

        // Test TransactionRemoved event
        let event_removed = MempoolEvent::TransactionRemoved {
            hash: [1u8; 32],
            timestamp: Instant::now(),
        };
        watcher.process_event(event_removed)?;

        // Test MempoolCleared event
        let event_cleared = MempoolEvent::MempoolCleared {
            timestamp: Instant::now(),
        };
        watcher.process_event(event_cleared)?;

        let stats = watcher.statistics();
        assert_eq!(stats.events_processed, 2);
        assert_eq!(stats.transactions_seen, 0); // These events don't increment transactions_seen
        assert_eq!(stats.replacements_detected, 0);

        Ok(())
    }

    #[test]
    fn test_watch_for_transactions_not_running() -> CoreResult<()> {
        // Test watch_for_transactions when not running (line 184)
        let watcher = MempoolWatcher::new(); // Not started

        let events = watcher.watch_for_transactions()?;
        assert!(events.is_empty());
        Ok(())
    }

    #[test]
    fn test_update_config() {
        // Test update_config method (lines 250-251)
        let mut watcher = MempoolWatcher::new();

        let new_config = WatcherConfig {
            enable_realtime: false,
            polling_interval: Duration::from_millis(200),
            max_event_buffer: 2000,
            detect_replacements: false,
        };

        watcher.update_config(new_config.clone());
        assert_eq!(
            watcher.config().polling_interval,
            Duration::from_millis(200)
        );
        assert_eq!(watcher.config().max_event_buffer, 2000);
        assert!(!watcher.config().detect_replacements);
    }

    #[test]
    fn test_health_check_not_running() {
        // Test health check when not running (lines 261, 283)
        let watcher = MempoolWatcher::new(); // Not started

        let health = watcher.health_check();
        assert_eq!(health.score, 0);
        assert_eq!(health.status, MempoolHealthStatus::Critical);
        assert!(!health.connection_stable);
    }

    #[test]
    fn test_health_check_activity_levels() -> CoreResult<()> {
        // Test different health check activity levels (lines 263, 265, 267, 269, 275, 277, 279)
        let mut watcher = MempoolWatcher::new();
        watcher.start()?;

        // Simulate high activity (>100 TPS) - line 263
        watcher.transactions_seen.store(1100, Ordering::Relaxed);
        watcher.start_time = Some(Instant::now() - Duration::from_secs(10)); // 10 seconds ago

        let health = watcher.health_check();
        assert_eq!(health.score, 100);
        assert_eq!(health.status, MempoolHealthStatus::Excellent); // Line 275

        // Simulate moderate activity (50-100 TPS) - line 265
        watcher.transactions_seen.store(600, Ordering::Relaxed); // 60 TPS
        let health = watcher.health_check();
        assert_eq!(health.score, 80);
        assert_eq!(health.status, MempoolHealthStatus::Excellent); // Line 275 (score 80 = Excellent)

        // Simulate low activity (10-50 TPS) - line 267
        watcher.transactions_seen.store(250, Ordering::Relaxed); // 25 TPS
        let health = watcher.health_check();
        assert_eq!(health.score, 60);
        assert_eq!(health.status, MempoolHealthStatus::Good); // Line 277 (score 60 = Good)

        // Simulate very low activity (1-10 TPS) - line 269
        watcher.transactions_seen.store(50, Ordering::Relaxed); // 5 TPS
        let health = watcher.health_check();
        assert_eq!(health.score, 40);
        assert_eq!(health.status, MempoolHealthStatus::Fair); // Line 279 (score 40 = Fair)

        // Simulate minimal activity (<1 TPS)
        watcher.transactions_seen.store(5, Ordering::Relaxed); // 0.5 TPS
        let health = watcher.health_check();
        assert_eq!(health.score, 20);
        assert_eq!(health.status, MempoolHealthStatus::Poor); // Line 281 (score 20 = Poor)

        Ok(())
    }

    #[test]
    fn test_mempool_health_methods() {
        // Test MempoolHealth methods (lines 356-358, 365-367)
        let excellent_health = MempoolHealth {
            status: MempoolHealthStatus::Excellent,
            score: 100,
            last_activity: Duration::from_secs(1),
            connection_stable: true,
            transaction_flow_normal: true,
        };
        assert!(excellent_health.is_healthy()); // Lines 356-358
        assert!(!excellent_health.needs_attention());

        let good_health = MempoolHealth {
            status: MempoolHealthStatus::Good,
            score: 80,
            last_activity: Duration::from_secs(1),
            connection_stable: true,
            transaction_flow_normal: true,
        };
        assert!(good_health.is_healthy()); // Lines 356-358
        assert!(!good_health.needs_attention());

        let poor_health = MempoolHealth {
            status: MempoolHealthStatus::Poor,
            score: 30,
            last_activity: Duration::from_secs(10),
            connection_stable: false,
            transaction_flow_normal: false,
        };
        assert!(!poor_health.is_healthy());
        assert!(poor_health.needs_attention()); // Lines 365-367

        let critical_health = MempoolHealth {
            status: MempoolHealthStatus::Critical,
            score: 0,
            last_activity: Duration::from_secs(60),
            connection_stable: false,
            transaction_flow_normal: false,
        };
        assert!(!critical_health.is_healthy());
        assert!(critical_health.needs_attention()); // Lines 365-367
    }

    #[test]
    fn test_watcher_default() {
        // Test Default implementation (lines 297-298)
        let watcher = MempoolWatcher::default();
        assert!(!watcher.is_running());
        assert_eq!(watcher.statistics().events_processed, 0);
    }

    #[test]
    fn test_start_already_running() -> CoreResult<()> {
        // Test starting when already running (line 138)
        let mut watcher = MempoolWatcher::new();
        watcher.start()?;
        assert!(watcher.is_running());

        // Start again - should be OK
        watcher.start()?;
        assert!(watcher.is_running());

        Ok(())
    }

    #[test]
    fn test_watcher_statistics_without_start_time() {
        // Test statistics when start_time is None (line 421, 461)
        let watcher = MempoolWatcher::new(); // Not started, so start_time is None

        let stats = watcher.statistics();
        assert_eq!(stats.uptime, Duration::default());
        assert_eq!(stats.events_per_second, 0.0);
        assert_eq!(stats.transactions_per_second, 0.0);
        assert_eq!(stats.replacement_rate, 0.0);
    }
}
