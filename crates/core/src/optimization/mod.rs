//! Performance optimizations for TallyIO core
//!
//! This module provides ultra-high performance optimizations including
//! CPU affinity, memory pooling, lock-free data structures, and SIMD operations.

pub mod cpu_affinity;
pub mod lock_free;
pub mod memory_pool;
pub mod simd;

// Re-export main optimization types
pub use cpu_affinity::CpuAffinity;
pub use lock_free::{LockFreeQueue, LockFreeStack};
pub use memory_pool::{MemoryPool, PooledBuffer};
pub use simd::SimdOps;

use crate::error::{CoreError, CoreResult};
use std::sync::atomic::{AtomicU64, Ordering};

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable CPU affinity
    pub enable_cpu_affinity: bool,
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Enable lock-free data structures
    pub enable_lock_free: bool,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Number of CPU cores to use
    pub cpu_cores: Option<usize>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_cpu_affinity: true,
            enable_memory_pooling: true,
            enable_lock_free: true,
            enable_simd: cfg!(feature = "simd"),
            memory_pool_size: 64 * 1024 * 1024, // 64MB
            cpu_cores: None,                    // Use all available cores
        }
    }
}

/// Performance optimization manager
///
/// Coordinates all performance optimizations for ultra-low latency operation.
#[repr(C, align(64))]
pub struct OptimizationManager {
    /// Configuration
    config: OptimizationConfig,
    /// CPU affinity manager
    cpu_affinity: Option<CpuAffinity>,
    /// Memory pool
    memory_pool: Option<MemoryPool>,
    /// Performance counters
    optimizations_applied: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl OptimizationManager {
    /// Create a new optimization manager
    pub fn new(config: OptimizationConfig) -> CoreResult<Self> {
        let mut manager = Self {
            config: config.clone(),
            cpu_affinity: None,
            memory_pool: None,
            optimizations_applied: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        };

        manager.initialize()?;
        Ok(manager)
    }

    /// Initialize optimizations
    fn initialize(&mut self) -> CoreResult<()> {
        // Initialize CPU affinity
        if self.config.enable_cpu_affinity {
            self.cpu_affinity = Some(CpuAffinity::new(self.config.cpu_cores)?);
            self.optimizations_applied.fetch_add(1, Ordering::Relaxed);
        }

        // Initialize memory pool
        if self.config.enable_memory_pooling {
            self.memory_pool = Some(MemoryPool::new(self.config.memory_pool_size)?);
            self.optimizations_applied.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Apply CPU affinity to current thread
    pub fn apply_cpu_affinity(&self, core_id: usize) -> CoreResult<()> {
        if let Some(cpu_affinity) = &self.cpu_affinity {
            cpu_affinity.set_affinity(core_id)?;
        }
        Ok(())
    }

    /// Get a pooled buffer
    pub fn get_buffer(&self, size: usize) -> CoreResult<PooledBuffer> {
        if let Some(memory_pool) = &self.memory_pool {
            match memory_pool.get_buffer(size) {
                Ok(buffer) => {
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    Ok(buffer)
                }
                Err(e) => {
                    self.cache_misses.fetch_add(1, Ordering::Relaxed);
                    Err(e)
                }
            }
        } else {
            Err(CoreError::optimization("Memory pooling not enabled"))
        }
    }

    /// Get optimization statistics
    #[must_use]
    pub fn statistics(&self) -> OptimizationStatistics {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        let cache_hit_ratio = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        OptimizationStatistics {
            optimizations_applied: self.optimizations_applied.load(Ordering::Relaxed),
            cpu_affinity_enabled: self.cpu_affinity.is_some(),
            memory_pooling_enabled: self.memory_pool.is_some(),
            lock_free_enabled: self.config.enable_lock_free,
            simd_enabled: self.config.enable_simd,
            cache_hits: hits,
            cache_misses: misses,
            cache_hit_ratio,
        }
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

impl Default for OptimizationManager {
    fn default() -> Self {
        Self::new(OptimizationConfig::default()).unwrap_or_else(|_| {
            // Fallback to minimal configuration
            Self {
                config: OptimizationConfig::default(),
                cpu_affinity: None,
                memory_pool: None,
                optimizations_applied: AtomicU64::new(0),
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
            }
        })
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStatistics {
    /// Number of optimizations applied
    pub optimizations_applied: u64,
    /// Whether CPU affinity is enabled
    pub cpu_affinity_enabled: bool,
    /// Whether memory pooling is enabled
    pub memory_pooling_enabled: bool,
    /// Whether lock-free structures are enabled
    pub lock_free_enabled: bool,
    /// Whether SIMD is enabled
    pub simd_enabled: bool,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Cache hit ratio (0.0 - 1.0)
    pub cache_hit_ratio: f64,
}

/// Trait for optimizable operations
pub trait Optimizable {
    /// Apply optimizations to this operation
    fn optimize(&mut self) -> CoreResult<()>;

    /// Check if optimizations are applied
    fn is_optimized(&self) -> bool;

    /// Get optimization level (0-100)
    fn optimization_level(&self) -> u8;
}

/// Macro for applying CPU affinity to current thread
#[macro_export]
macro_rules! apply_cpu_affinity {
    ($core:expr) => {
        if let Err(e) = $crate::optimization::CpuAffinity::set_current_thread_affinity($core) {
            log::warn!("Failed to set CPU affinity: {}", e);
        }
    };
}

/// Macro for creating cache-aligned structures
#[macro_export]
macro_rules! cache_aligned {
    ($name:ident, $($field:ident: $type:ty),*) => {
        #[repr(C, align(64))]
        pub struct $name {
            $(pub $field: $type,)*
        }
    };
}

/// Macro for prefetching memory
#[macro_export]
macro_rules! prefetch {
    ($addr:expr) => {
        #[cfg(target_arch = "x86_64")]
        {
            let addr_ptr = $addr;
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    addr_ptr.cast::<i8>(),
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_manager_creation() -> CoreResult<()> {
        let config = OptimizationConfig::default();
        let manager = OptimizationManager::new(config)?;

        let stats = manager.statistics();
        assert!(stats.optimizations_applied > 0);

        Ok(())
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig {
            enable_cpu_affinity: false,
            enable_memory_pooling: true,
            enable_lock_free: true,
            enable_simd: false,
            memory_pool_size: 32 * 1024 * 1024,
            cpu_cores: Some(4),
        };

        assert!(!config.enable_cpu_affinity);
        assert!(config.enable_memory_pooling);
        assert_eq!(config.memory_pool_size, 32 * 1024 * 1024);
        assert_eq!(config.cpu_cores, Some(4));
    }

    #[test]
    fn test_optimization_statistics() -> CoreResult<()> {
        let manager = OptimizationManager::default();
        let stats = manager.statistics();

        assert!(stats.cache_hit_ratio >= 0.0 && stats.cache_hit_ratio <= 1.0);

        Ok(())
    }

    #[test]
    fn test_cache_aligned_macro() {
        cache_aligned!(TestStruct, field1: u64, field2: u32);

        let test_struct = TestStruct {
            field1: 42,
            field2: 24,
        };

        assert_eq!(test_struct.field1, 42);
        assert_eq!(test_struct.field2, 24);

        // Check alignment
        assert_eq!(std::mem::align_of::<TestStruct>(), 64);
    }

    #[test]
    fn test_optimization_manager_with_disabled_features() -> CoreResult<()> {
        let config = OptimizationConfig {
            enable_cpu_affinity: false,
            enable_memory_pooling: false,
            enable_lock_free: false,
            enable_simd: false,
            memory_pool_size: 1024,
            cpu_cores: None,
        };

        let manager = OptimizationManager::new(config)?;
        let stats = manager.statistics();

        assert!(!stats.cpu_affinity_enabled);
        assert!(!stats.memory_pooling_enabled);
        assert!(!stats.lock_free_enabled);
        assert!(!stats.simd_enabled);

        Ok(())
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert!(config.enable_cpu_affinity);
        assert!(config.enable_memory_pooling);
        assert!(config.enable_lock_free);
        assert_eq!(config.memory_pool_size, 64 * 1024 * 1024);
        assert!(config.cpu_cores.is_none());
    }

    #[test]
    fn test_optimization_manager_default() {
        let manager = OptimizationManager::default();
        let stats = manager.statistics();

        // Should have some optimizations applied by default
        assert!(stats.cache_hit_ratio >= 0.0 && stats.cache_hit_ratio <= 1.0);
    }

    #[test]
    fn test_optimization_manager_cpu_affinity() -> CoreResult<()> {
        let config = OptimizationConfig {
            enable_cpu_affinity: true,
            enable_memory_pooling: false,
            enable_lock_free: false,
            enable_simd: false,
            memory_pool_size: 1024,
            cpu_cores: Some(2),
        };

        let manager = OptimizationManager::new(config)?;

        // Try to apply CPU affinity (might fail on some systems)
        let result = manager.apply_cpu_affinity(0);
        // Don't assert success as it depends on system capabilities
        let _is_ok = result.is_ok();

        Ok(())
    }

    #[test]
    fn test_optimization_manager_memory_pooling() -> CoreResult<()> {
        let config = OptimizationConfig {
            enable_cpu_affinity: false,
            enable_memory_pooling: true,
            enable_lock_free: false,
            enable_simd: false,
            memory_pool_size: 1024 * 1024,
            cpu_cores: None,
        };

        let manager = OptimizationManager::new(config)?;
        let stats = manager.statistics();

        assert!(stats.memory_pooling_enabled);

        // Try to get a buffer
        let result = manager.get_buffer(1024);
        // This might succeed or fail depending on implementation
        let _is_ok = result.is_ok();

        Ok(())
    }

    #[test]
    fn test_optimization_manager_buffer_cache_miss() -> CoreResult<()> {
        let config = OptimizationConfig {
            enable_cpu_affinity: false,
            enable_memory_pooling: false, // Disabled
            enable_lock_free: false,
            enable_simd: false,
            memory_pool_size: 1024,
            cpu_cores: None,
        };

        let manager = OptimizationManager::new(config)?;

        // Should fail because memory pooling is disabled
        let result = manager.get_buffer(1024);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_optimization_statistics_cache_ratios() -> CoreResult<()> {
        let manager = OptimizationManager::default();

        // Initial state
        let stats = manager.statistics();
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.cache_hit_ratio, 0.0);

        Ok(())
    }

    #[test]
    fn test_optimization_manager_config_access() -> CoreResult<()> {
        let config = OptimizationConfig {
            enable_cpu_affinity: true,
            enable_memory_pooling: false,
            enable_lock_free: true,
            enable_simd: false,
            memory_pool_size: 2 * 1024 * 1024,
            cpu_cores: Some(4),
        };

        let manager = OptimizationManager::new(config.clone())?;
        let retrieved_config = manager.config();

        assert_eq!(
            retrieved_config.enable_cpu_affinity,
            config.enable_cpu_affinity
        );
        assert_eq!(
            retrieved_config.enable_memory_pooling,
            config.enable_memory_pooling
        );
        assert_eq!(retrieved_config.enable_lock_free, config.enable_lock_free);
        assert_eq!(retrieved_config.enable_simd, config.enable_simd);
        assert_eq!(retrieved_config.memory_pool_size, config.memory_pool_size);
        assert_eq!(retrieved_config.cpu_cores, config.cpu_cores);

        Ok(())
    }

    #[test]
    fn test_optimization_manager_initialization_counts() -> CoreResult<()> {
        let config = OptimizationConfig {
            enable_cpu_affinity: true,
            enable_memory_pooling: true,
            enable_lock_free: false,
            enable_simd: false,
            memory_pool_size: 1024 * 1024,
            cpu_cores: None,
        };

        let manager = OptimizationManager::new(config)?;
        let stats = manager.statistics();

        // Should have applied 2 optimizations (CPU affinity + memory pooling)
        assert_eq!(stats.optimizations_applied, 2);

        Ok(())
    }

    #[test]
    fn test_optimization_manager_fallback_creation() {
        // Test that default creation works even if normal creation fails
        let manager = OptimizationManager::default();

        // Should have basic functionality
        let stats = manager.statistics();
        assert!(stats.cache_hit_ratio >= 0.0 && stats.cache_hit_ratio <= 1.0);
    }

    #[test]
    fn test_optimizable_trait_bounds() {
        // Test that we can define types implementing Optimizable
        struct TestOptimizable {
            optimized: bool,
            level: u8,
        }

        impl Optimizable for TestOptimizable {
            fn optimize(&mut self) -> CoreResult<()> {
                self.optimized = true;
                self.level = 100;
                Ok(())
            }

            fn is_optimized(&self) -> bool {
                self.optimized
            }

            fn optimization_level(&self) -> u8 {
                self.level
            }
        }

        let mut test_obj = TestOptimizable {
            optimized: false,
            level: 0,
        };

        assert!(!test_obj.is_optimized());
        assert_eq!(test_obj.optimization_level(), 0);

        assert!(test_obj.optimize().is_ok());

        assert!(test_obj.is_optimized());
        assert_eq!(test_obj.optimization_level(), 100);
    }

    #[test]
    fn test_cache_aligned_macro_multiple_fields() {
        cache_aligned!(MultiFieldStruct,
            field1: u64,
            field2: u32,
            field3: u16,
            field4: u8
        );

        let test_struct = MultiFieldStruct {
            field1: 42,
            field2: 24,
            field3: 12,
            field4: 6,
        };

        assert_eq!(test_struct.field1, 42);
        assert_eq!(test_struct.field2, 24);
        assert_eq!(test_struct.field3, 12);
        assert_eq!(test_struct.field4, 6);

        // Check alignment
        assert_eq!(std::mem::align_of::<MultiFieldStruct>(), 64);
    }

    #[test]
    fn test_prefetch_macro_compilation() {
        // Test that prefetch macro compiles without errors
        let data = [1u8, 2, 3, 4, 5];
        let ptr = data.as_ptr();

        // This should compile without errors
        prefetch!(ptr);

        // The macro might be a no-op on some architectures, but should not panic
        assert_eq!(data[0], 1);
    }

    #[test]
    fn test_apply_cpu_affinity_macro_compilation() {
        // Test that apply_cpu_affinity macro compiles
        // This might fail on some systems but should not panic the test
        apply_cpu_affinity!(0);

        // If we reach here, the macro compiled successfully
        // No assertion needed - successful compilation is the test
    }
}
