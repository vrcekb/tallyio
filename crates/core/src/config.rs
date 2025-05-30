//! Core configuration for TallyIO engine
//!
//! This module provides configuration structures for the core engine
//! with performance-optimized defaults and validation.

use crate::error::{CoreError, CoreResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Core engine configuration
///
/// This structure contains all configuration parameters for the TallyIO core engine.
/// All values are validated to ensure optimal performance and system stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    /// Number of worker threads for processing
    pub worker_threads: usize,
    /// Enable CPU affinity for worker threads
    pub enable_cpu_affinity: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Maximum queue size for transactions
    pub max_queue_size: usize,
    /// Latency warning threshold in microseconds
    pub latency_warning_threshold_us: u64,
    /// Latency critical threshold in microseconds
    pub latency_critical_threshold_us: u64,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Batch size for transaction processing
    pub batch_size: usize,
    /// Scheduler tick interval
    pub scheduler_tick_interval: Duration,
    /// Worker thread stack size in bytes
    pub worker_stack_size: usize,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get().min(16),
            enable_cpu_affinity: true,
            memory_pool_size: 64 * 1024 * 1024, // 64MB
            max_queue_size: 100_000,
            latency_warning_threshold_us: 500,
            latency_critical_threshold_us: 1000,
            enable_simd: cfg!(feature = "simd"),
            batch_size: 100,
            scheduler_tick_interval: Duration::from_micros(100),
            worker_stack_size: 2 * 1024 * 1024, // 2MB
        }
    }
}

impl CoreConfig {
    /// Create a new configuration builder
    #[must_use]
    pub fn builder() -> CoreConfigBuilder {
        CoreConfigBuilder::new()
    }

    /// Validate the configuration
    ///
    /// Ensures all configuration values are within acceptable ranges
    /// for optimal performance and system stability.
    pub fn validate(&self) -> CoreResult<()> {
        if self.worker_threads == 0 {
            return Err(CoreError::config("worker_threads must be greater than 0"));
        }

        if self.worker_threads > 64 {
            return Err(CoreError::config("worker_threads should not exceed 64"));
        }

        if self.memory_pool_size < 1024 * 1024 {
            return Err(CoreError::config("memory_pool_size must be at least 1MB"));
        }

        if self.max_queue_size < 1000 {
            return Err(CoreError::config("max_queue_size must be at least 1000"));
        }

        if self.latency_critical_threshold_us < self.latency_warning_threshold_us {
            return Err(CoreError::config(
                "latency_critical_threshold_us must be >= latency_warning_threshold_us",
            ));
        }

        if self.batch_size == 0 {
            return Err(CoreError::config("batch_size must be greater than 0"));
        }

        if self.worker_stack_size < 512 * 1024 {
            return Err(CoreError::config(
                "worker_stack_size must be at least 512KB",
            ));
        }

        Ok(())
    }

    /// Get optimal configuration for the current system
    #[must_use]
    pub fn optimal() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            worker_threads: cpu_count.min(16),
            enable_cpu_affinity: cpu_count <= 32, // Only enable on smaller systems
            memory_pool_size: 128 * 1024 * 1024,  // 128MB for optimal performance
            max_queue_size: 200_000,
            latency_warning_threshold_us: 250,
            latency_critical_threshold_us: 500,
            enable_simd: true,
            batch_size: 200,
            scheduler_tick_interval: Duration::from_micros(50),
            worker_stack_size: 4 * 1024 * 1024, // 4MB for complex operations
        }
    }

    /// Get minimal configuration for testing
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            worker_threads: 1,
            enable_cpu_affinity: false,
            memory_pool_size: 1024 * 1024, // 1MB
            max_queue_size: 1000,
            latency_warning_threshold_us: 1000,
            latency_critical_threshold_us: 2000,
            enable_simd: false,
            batch_size: 10,
            scheduler_tick_interval: Duration::from_millis(1),
            worker_stack_size: 512 * 1024, // 512KB
        }
    }
}

/// Builder for CoreConfig
#[derive(Debug)]
pub struct CoreConfigBuilder {
    config: CoreConfig,
}

impl CoreConfigBuilder {
    /// Create a new builder with default values
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CoreConfig::default(),
        }
    }

    /// Set the number of worker threads
    #[must_use]
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.config.worker_threads = threads;
        self
    }

    /// Enable or disable CPU affinity
    #[must_use]
    pub fn enable_cpu_affinity(mut self, enable: bool) -> Self {
        self.config.enable_cpu_affinity = enable;
        self
    }

    /// Set the memory pool size
    #[must_use]
    pub fn memory_pool_size(mut self, size: usize) -> Self {
        self.config.memory_pool_size = size;
        self
    }

    /// Set the maximum queue size
    #[must_use]
    pub fn max_queue_size(mut self, size: usize) -> Self {
        self.config.max_queue_size = size;
        self
    }

    /// Set the latency warning threshold
    #[must_use]
    pub fn latency_warning_threshold_us(mut self, threshold: u64) -> Self {
        self.config.latency_warning_threshold_us = threshold;
        self
    }

    /// Set the latency critical threshold
    #[must_use]
    pub fn latency_critical_threshold_us(mut self, threshold: u64) -> Self {
        self.config.latency_critical_threshold_us = threshold;
        self
    }

    /// Enable or disable SIMD optimizations
    #[must_use]
    pub fn enable_simd(mut self, enable: bool) -> Self {
        self.config.enable_simd = enable;
        self
    }

    /// Set the batch size
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set the scheduler tick interval
    #[must_use]
    pub fn scheduler_tick_interval(mut self, interval: Duration) -> Self {
        self.config.scheduler_tick_interval = interval;
        self
    }

    /// Set the worker thread stack size
    #[must_use]
    pub fn worker_stack_size(mut self, size: usize) -> Self {
        self.config.worker_stack_size = size;
        self
    }

    /// Build the configuration
    ///
    /// Validates the configuration and returns it if valid.
    pub fn build(self) -> CoreResult<CoreConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for CoreConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Add num_cpus as a dependency placeholder
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

    #[test]
    fn test_default_config() {
        let config = CoreConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.worker_threads > 0);
        assert!(config.memory_pool_size >= 1024 * 1024);
    }

    #[test]
    fn test_config_builder() -> CoreResult<()> {
        let config = CoreConfig::builder()
            .worker_threads(8)
            .enable_cpu_affinity(true)
            .memory_pool_size(32 * 1024 * 1024)
            .build()?;

        assert_eq!(config.worker_threads, 8);
        assert!(config.enable_cpu_affinity);
        assert_eq!(config.memory_pool_size, 32 * 1024 * 1024);
        Ok(())
    }

    #[test]
    fn test_config_validation() {
        let mut config = CoreConfig::default();

        // Test invalid worker threads
        config.worker_threads = 0;
        assert!(config.validate().is_err());

        config.worker_threads = 1;
        assert!(config.validate().is_ok());

        // Test invalid memory pool size
        config.memory_pool_size = 1024;
        assert!(config.validate().is_err());

        config.memory_pool_size = 1024 * 1024;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_optimal_config() {
        let config = CoreConfig::optimal();
        assert!(config.validate().is_ok());
        assert!(config.worker_threads <= 16);
        assert_eq!(config.memory_pool_size, 128 * 1024 * 1024);
    }

    #[test]
    fn test_minimal_config() {
        let config = CoreConfig::minimal();
        assert!(config.validate().is_ok());
        assert_eq!(config.worker_threads, 1);
        assert!(!config.enable_cpu_affinity);
    }

    #[test]
    fn test_config_validation_comprehensive() {
        let mut config = CoreConfig::default();

        // Test worker threads > 64 (line 72)
        config.worker_threads = 65;
        assert!(config.validate().is_err());

        // Test memory pool size < 1MB (line 76)
        config.worker_threads = 4;
        config.memory_pool_size = 1024 * 1024 - 1;
        assert!(config.validate().is_err());

        // Test max queue size < 1000 (line 80)
        config.memory_pool_size = 1024 * 1024;
        config.max_queue_size = 999;
        assert!(config.validate().is_err());

        // Test latency thresholds (lines 84-85)
        config.max_queue_size = 1000;
        config.latency_critical_threshold_us = 100;
        config.latency_warning_threshold_us = 200;
        assert!(config.validate().is_err());

        // Test batch size = 0 (line 90)
        config.latency_critical_threshold_us = 200;
        config.latency_warning_threshold_us = 100;
        config.batch_size = 0;
        assert!(config.validate().is_err());

        // Test worker stack size < 512KB (lines 94-95)
        config.batch_size = 1;
        config.worker_stack_size = 512 * 1024 - 1;
        assert!(config.validate().is_err());

        // Test valid configuration
        config.worker_stack_size = 512 * 1024;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder_all_methods() -> CoreResult<()> {
        // Test all builder methods to cover lines 176-220
        let config = CoreConfigBuilder::new()
            .worker_threads(4)
            .enable_cpu_affinity(false)
            .memory_pool_size(16 * 1024 * 1024)
            .max_queue_size(50_000)
            .latency_warning_threshold_us(300)
            .latency_critical_threshold_us(600)
            .enable_simd(true)
            .batch_size(50)
            .scheduler_tick_interval(Duration::from_micros(200))
            .worker_stack_size(1024 * 1024)
            .build()?;

        assert_eq!(config.worker_threads, 4);
        assert!(!config.enable_cpu_affinity);
        assert_eq!(config.memory_pool_size, 16 * 1024 * 1024);
        assert_eq!(config.max_queue_size, 50_000);
        assert_eq!(config.latency_warning_threshold_us, 300);
        assert_eq!(config.latency_critical_threshold_us, 600);
        assert!(config.enable_simd);
        assert_eq!(config.batch_size, 50);
        assert_eq!(config.scheduler_tick_interval, Duration::from_micros(200));
        assert_eq!(config.worker_stack_size, 1024 * 1024);
        Ok(())
    }

    #[test]
    fn test_config_builder_default() -> CoreResult<()> {
        // Test Default implementation for CoreConfigBuilder (lines 233-234)
        let builder = CoreConfigBuilder::default();
        let config = builder.build()?;
        assert!(config.validate().is_ok());
        Ok(())
    }

    #[test]
    fn test_config_builder_invalid() {
        // Test builder with invalid configuration
        let result = CoreConfigBuilder::new().worker_threads(0).build();
        assert!(result.is_err());
    }
}
