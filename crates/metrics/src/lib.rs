//! `TallyIO` Metrics - Metrics and monitoring

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),

    #[error("Collection error: {0}")]
    Collection(String),
}

pub type MetricsResult<T> = Result<T, MetricsError>;

/// Placeholder for metrics functionality
pub struct MetricsManager {
    metric_count: std::sync::atomic::AtomicU64,
}

impl MetricsManager {
    /// Create new metrics manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> MetricsResult<Self> {
        Ok(Self {
            metric_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Record a metric
    ///
    /// # Errors
    /// Returns error if metric recording fails
    pub fn record_metric(&self, metric: &str) -> MetricsResult<String> {
        self.metric_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("Recorded metric: {metric}"))
    }
}

impl Default for MetricsManager {
    fn default() -> Self {
        // Use match instead of expect to comply with zero-panic policy
        #[allow(clippy::option_if_let_else)] // Result, not Option
        match Self::new() {
            Ok(manager) => manager,
            Err(_) => {
                // This should never happen in normal circumstances
                // If it does, it's a programming error
                std::process::abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_metrics_manager_creation() -> MetricsResult<()> {
        let manager = MetricsManager::new()?;
        assert_eq!(
            manager
                .metric_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn test_metrics_manager_default() {
        let manager = MetricsManager::default();
        assert_eq!(
            manager
                .metric_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_record_metric() -> MetricsResult<()> {
        let manager = MetricsManager::new()?;
        let result = manager.record_metric("test_data")?;

        // Verify operation was processed
        assert_eq!(result, "Recorded metric: test_data");
        assert_eq!(
            manager
                .metric_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        Ok(())
    }

    #[test]
    fn test_metrics_latency_requirement() -> MetricsResult<()> {
        let manager = MetricsManager::new()?;
        let start = Instant::now();

        manager.record_metric("latency_test")?;

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1,
            "metrics operation took {}ms, must be <1ms",
            duration.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_operations() -> MetricsResult<()> {
        let manager = MetricsManager::new()?;

        for i in 0..10 {
            manager.record_metric(&format!("operation_{}", i))?;
        }

        assert_eq!(
            manager
                .metric_count
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
        Ok(())
    }

    #[test]
    fn test_concurrent_operations() -> MetricsResult<()> {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(MetricsManager::new()?);
        let mut handles = vec![];

        for i in 0..5 {
            let manager_clone = Arc::clone(&manager);
            let handle =
                thread::spawn(move || manager_clone.record_metric(&format!("concurrent_{}", i)));
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap()?;
        }

        assert_eq!(
            manager
                .metric_count
                .load(std::sync::atomic::Ordering::Relaxed),
            5
        );
        Ok(())
    }
}
