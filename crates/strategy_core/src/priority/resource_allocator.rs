//! # CPU/Memory Resource Allocation
//!
//! Advanced resource allocation for optimal strategy execution performance.

use crate::{StrategyResult, StrategyPriority, types::*};

/// Resource allocation info
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ResourceAllocation {
    /// CPU cores allocated
    pub cpu_cores: u32,
    /// Memory allocated in MB
    pub memory_mb: u64,
    /// NUMA node preference
    pub numa_node: Option<u32>,
}

/// Resource allocator
#[derive(Debug)]
#[non_exhaustive]
pub struct ResourceAllocator {
    /// Total available CPU cores
    total_cpu_cores: u32,
    /// Total available memory in MB
    total_memory_mb: u64,
}

impl ResourceAllocator {
    /// Create new resource allocator
    #[inline]
    #[must_use]
    pub const fn new(total_cpu_cores: u32, total_memory_mb: u64) -> Self {
        Self {
            total_cpu_cores,
            total_memory_mb,
        }
    }
    
    /// Allocate resources for strategy
    #[inline]
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn allocate_resources(&self, priority: StrategyPriority, _expected_profit: ProfitAmount) -> StrategyResult<ResourceAllocation> {
        let (cpu_ratio, memory_ratio) = match priority {
            StrategyPriority::Critical => (0.5_f64, 0.4_f64), // 50% CPU, 40% memory
            StrategyPriority::High => (0.3_f64, 0.3_f64),     // 30% CPU, 30% memory
            StrategyPriority::Medium => (0.15_f64, 0.2_f64),  // 15% CPU, 20% memory
            StrategyPriority::Low => (0.05_f64, 0.1_f64),     // 5% CPU, 10% memory
            StrategyPriority::Background => (0.02_f64, 0.05_f64), // 2% CPU, 5% memory
        };
        
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::float_arithmetic, reason = "Resource allocation requires float arithmetic and safe casting")]

        
        let cpu_cores = ((f64::from(self.total_cpu_cores)) * cpu_ratio).max(1.0) as u32;
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::float_arithmetic, reason = "Resource allocation requires float arithmetic and safe casting")]

        #[expect(clippy::cast_precision_loss, reason = "Precision loss acceptable for resource allocation ratios")]


        let memory_mb = ((self.total_memory_mb as f64) * memory_ratio).max(100.0) as u64;
        
        Ok(ResourceAllocation {
            cpu_cores,
            memory_mb,
            numa_node: None, // Will be determined by system
        })
    }
}

impl Default for ResourceAllocator {
    #[inline]
    fn default() -> Self {
        // Default to 48 cores and 220GB (typical for AMD EPYC 9454P setup)
        Self::new(48, 225_280) // 220 * 1024 MB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_allocator_creation() {
        let allocator = ResourceAllocator::new(48, 225_280);
        assert_eq!(allocator.total_cpu_cores, 48);
        assert_eq!(allocator.total_memory_mb, 225_280);
    }

    #[test]
    fn resource_allocation_critical_priority() {
        let allocator = ResourceAllocator::default();
        let allocation = allocator.allocate_resources(StrategyPriority::Critical, 10000);
        
        assert!(allocation.is_ok());
        if let Ok(alloc) = allocation {
            assert_eq!(alloc.cpu_cores, 24); // 50% of 48
            assert_eq!(alloc.memory_mb, 90_112); // 40% of 225_280
        }
    }

    #[test]
    fn resource_allocation_background_priority() {
        let allocator = ResourceAllocator::default();
        let allocation = allocator.allocate_resources(StrategyPriority::Background, 100);
        
        assert!(allocation.is_ok());
        if let Ok(alloc) = allocation {
            assert_eq!(alloc.cpu_cores, 1); // Minimum 1 core
            assert_eq!(alloc.memory_mb, 11_264); // 5% of 225_280
        }
    }
}
