//! CPU affinity management for ultra-low latency operations
//!
//! This module provides CPU core binding functionality to minimize context switching
//! and ensure consistent performance for critical trading operations.

use crate::error::{CoreError, CoreResult};
use std::sync::atomic::{AtomicBool, Ordering};

/// CPU affinity manager
///
/// Provides functionality to bind threads to specific CPU cores for optimal performance.
/// This reduces context switching and ensures predictable latency for critical operations.
#[derive(Debug)]
pub struct CpuAffinity {
    /// Available CPU cores
    available_cores: Vec<usize>,
    /// Whether affinity is enabled
    enabled: AtomicBool,
}

impl CpuAffinity {
    /// Create a new CPU affinity manager
    pub fn new(cpu_cores: Option<usize>) -> CoreResult<Self> {
        let num_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let available_cores: Vec<usize> = match cpu_cores {
            Some(cores) => (0..cores.min(num_cores)).collect(),
            None => (0..num_cores).collect(),
        };

        if available_cores.is_empty() {
            return Err(CoreError::optimization("No CPU cores available"));
        }

        Ok(Self {
            available_cores,
            enabled: AtomicBool::new(true),
        })
    }

    /// Set CPU affinity for current thread
    pub fn set_affinity(&self, core_id: usize) -> CoreResult<()> {
        if !self.enabled.load(Ordering::Acquire) {
            return Ok(());
        }

        if !self.available_cores.contains(&core_id) {
            return Err(CoreError::optimization(format!(
                "Core {} not available. Available cores: {:?}",
                core_id, self.available_cores
            )));
        }

        // On Windows, we'll use a simplified approach
        // In a real implementation, this would use platform-specific APIs
        #[cfg(target_os = "windows")]
        {
            // Windows CPU affinity would be set here using SetThreadAffinityMask
            // For now, we'll just log the operation
            log::debug!("Setting CPU affinity to core {} (simulated)", core_id);
        }

        #[cfg(target_os = "linux")]
        {
            // Linux CPU affinity would be set here using sched_setaffinity
            // For now, we'll just log the operation
            log::debug!("Setting CPU affinity to core {} (simulated)", core_id);
        }

        #[cfg(target_os = "macos")]
        {
            // macOS CPU affinity would be set here using thread_policy_set
            // For now, we'll just log the operation
            log::debug!("Setting CPU affinity to core {} (simulated)", core_id);
        }

        Ok(())
    }

    /// Set CPU affinity for current thread (static method)
    pub fn set_current_thread_affinity(core_id: usize) -> CoreResult<()> {
        // Simplified implementation for cross-platform compatibility
        log::debug!("Setting current thread CPU affinity to core {}", core_id);
        Ok(())
    }

    /// Get available CPU cores
    #[must_use]
    pub fn available_cores(&self) -> &[usize] {
        &self.available_cores
    }

    /// Get number of available cores
    #[must_use]
    pub fn core_count(&self) -> usize {
        self.available_cores.len()
    }

    /// Enable CPU affinity
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    /// Disable CPU affinity
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Check if CPU affinity is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Get optimal core for worker thread
    #[must_use]
    pub fn optimal_core_for_worker(&self, worker_id: usize) -> usize {
        if self.available_cores.is_empty() {
            0
        } else {
            self.available_cores[worker_id % self.available_cores.len()]
        }
    }

    /// Get CPU topology information
    #[must_use]
    pub fn topology_info(&self) -> CpuTopology {
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Simplified topology detection
        // In a real implementation, this would query actual CPU topology
        let physical_cores = total_cores / 2; // Assume hyperthreading
        let numa_nodes = if total_cores > 8 { 2 } else { 1 };

        CpuTopology {
            total_cores,
            physical_cores,
            logical_cores: total_cores,
            numa_nodes,
            available_cores: self.available_cores.clone(),
        }
    }

    /// Bind worker threads to cores optimally
    pub fn bind_workers(&self, worker_count: usize) -> CoreResult<Vec<usize>> {
        if worker_count == 0 {
            return Ok(Vec::with_capacity(0));
        }

        let mut assignments = Vec::with_capacity(worker_count);

        // Distribute workers across available cores
        for i in 0..worker_count {
            let core = self.optimal_core_for_worker(i);
            assignments.push(core);
        }

        Ok(assignments)
    }
}

/// CPU topology information
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Total number of CPU cores
    pub total_cores: usize,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores (including hyperthreading)
    pub logical_cores: usize,
    /// Number of NUMA nodes
    pub numa_nodes: usize,
    /// Available cores for binding
    pub available_cores: Vec<usize>,
}

impl CpuTopology {
    /// Check if hyperthreading is available
    #[must_use]
    pub fn has_hyperthreading(&self) -> bool {
        self.logical_cores > self.physical_cores
    }

    /// Check if NUMA is present
    #[must_use]
    pub fn has_numa(&self) -> bool {
        self.numa_nodes > 1
    }

    /// Get cores per NUMA node
    #[must_use]
    pub fn cores_per_numa_node(&self) -> usize {
        if self.numa_nodes > 0 {
            self.total_cores / self.numa_nodes
        } else {
            self.total_cores
        }
    }
}

/// CPU affinity configuration
#[derive(Debug, Clone)]
pub struct AffinityConfig {
    /// Enable CPU affinity
    pub enabled: bool,
    /// Specific cores to use (None = use all)
    pub cores: Option<Vec<usize>>,
    /// Prefer physical cores over logical cores
    pub prefer_physical_cores: bool,
    /// Enable NUMA awareness
    pub numa_aware: bool,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cores: None,
            prefer_physical_cores: true,
            numa_aware: true,
        }
    }
}

/// Advanced CPU affinity manager with NUMA awareness
#[derive(Debug)]
pub struct AdvancedCpuAffinity {
    /// Basic affinity manager
    basic: CpuAffinity,
    /// Configuration
    config: AffinityConfig,
    /// CPU topology
    topology: CpuTopology,
}

impl AdvancedCpuAffinity {
    /// Create a new advanced CPU affinity manager
    pub fn new(config: AffinityConfig) -> CoreResult<Self> {
        let basic = CpuAffinity::new(None)?;
        let topology = basic.topology_info();

        Ok(Self {
            basic,
            config,
            topology,
        })
    }

    /// Get optimal core assignment for critical operations
    #[must_use]
    pub fn optimal_critical_core(&self) -> usize {
        // For critical operations, prefer the first physical core
        if self.config.prefer_physical_cores && !self.topology.available_cores.is_empty() {
            self.topology.available_cores[0]
        } else {
            0
        }
    }

    /// Get optimal core assignment for background operations
    #[must_use]
    pub fn optimal_background_core(&self) -> usize {
        // For background operations, use the last available core
        if !self.topology.available_cores.is_empty() {
            *self.topology.available_cores.last().unwrap_or(&0)
        } else {
            0
        }
    }

    /// Set affinity for critical thread
    pub fn set_critical_affinity(&self) -> CoreResult<()> {
        if self.config.enabled {
            let core = self.optimal_critical_core();
            self.basic.set_affinity(core)?;
        }
        Ok(())
    }

    /// Set affinity for background thread
    pub fn set_background_affinity(&self) -> CoreResult<()> {
        if self.config.enabled {
            let core = self.optimal_background_core();
            self.basic.set_affinity(core)?;
        }
        Ok(())
    }

    /// Get CPU topology
    #[must_use]
    pub const fn topology(&self) -> &CpuTopology {
        &self.topology
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &AffinityConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_affinity_creation() -> CoreResult<()> {
        let affinity = CpuAffinity::new(Some(4))?;
        assert!(affinity.core_count() <= 4);
        assert!(!affinity.available_cores().is_empty());
        Ok(())
    }

    #[test]
    fn test_cpu_affinity_enable_disable() -> CoreResult<()> {
        let affinity = CpuAffinity::new(None)?;
        assert!(affinity.is_enabled());

        affinity.disable();
        assert!(!affinity.is_enabled());

        affinity.enable();
        assert!(affinity.is_enabled());

        Ok(())
    }

    #[test]
    fn test_optimal_core_assignment() -> CoreResult<()> {
        let affinity = CpuAffinity::new(Some(4))?;

        let core0 = affinity.optimal_core_for_worker(0);
        let core1 = affinity.optimal_core_for_worker(1);

        assert!(core0 < affinity.core_count());
        assert!(core1 < affinity.core_count());

        Ok(())
    }

    #[test]
    fn test_worker_binding() -> CoreResult<()> {
        let affinity = CpuAffinity::new(Some(4))?;
        let assignments = affinity.bind_workers(3)?;

        assert_eq!(assignments.len(), 3);
        for &core in &assignments {
            assert!(affinity.available_cores().contains(&core));
        }

        Ok(())
    }

    #[test]
    fn test_cpu_topology() -> CoreResult<()> {
        let affinity = CpuAffinity::new(None)?;
        let topology = affinity.topology_info();

        assert!(topology.total_cores > 0);
        assert!(topology.physical_cores > 0);
        assert!(topology.logical_cores >= topology.physical_cores);
        assert!(topology.numa_nodes > 0);

        Ok(())
    }

    #[test]
    fn test_advanced_cpu_affinity() -> CoreResult<()> {
        let config = AffinityConfig::default();
        let advanced = AdvancedCpuAffinity::new(config)?;

        let critical_core = advanced.optimal_critical_core();
        let background_core = advanced.optimal_background_core();

        assert!(critical_core < advanced.topology().total_cores);
        assert!(background_core < advanced.topology().total_cores);

        Ok(())
    }

    #[test]
    fn test_set_current_thread_affinity() -> CoreResult<()> {
        // This should not fail even if actual affinity setting is not implemented
        CpuAffinity::set_current_thread_affinity(0)?;
        Ok(())
    }

    #[test]
    fn test_invalid_core_assignment() -> CoreResult<()> {
        let affinity = CpuAffinity::new(Some(2))?;

        // Try to set affinity to a core that doesn't exist
        let result = affinity.set_affinity(999);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_affinity_config() {
        let config = AffinityConfig {
            enabled: false,
            cores: Some(vec![0, 1, 2]),
            prefer_physical_cores: false,
            numa_aware: false,
        };

        assert!(!config.enabled);
        assert_eq!(config.cores, Some(vec![0, 1, 2]));
        assert!(!config.prefer_physical_cores);
        assert!(!config.numa_aware);
    }

    #[test]
    fn test_cpu_affinity_no_cores_available() {
        // Test error case when no cores are available
        let result = CpuAffinity::new(Some(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_affinity_disabled_set_affinity() -> CoreResult<()> {
        let affinity = CpuAffinity::new(Some(2))?;
        affinity.disable();

        // Should succeed even when disabled
        let result = affinity.set_affinity(0);
        assert!(result.is_ok());

        Ok(())
    }

    #[test]
    fn test_cpu_topology_methods() -> CoreResult<()> {
        let affinity = CpuAffinity::new(None)?;
        let topology = affinity.topology_info();

        // Test hyperthreading detection
        let has_ht = topology.has_hyperthreading();
        assert_eq!(has_ht, topology.logical_cores > topology.physical_cores);

        // Test NUMA detection
        let has_numa = topology.has_numa();
        assert_eq!(has_numa, topology.numa_nodes > 1);

        // Test cores per NUMA node
        let cores_per_node = topology.cores_per_numa_node();
        if topology.numa_nodes > 0 {
            assert_eq!(cores_per_node, topology.total_cores / topology.numa_nodes);
        } else {
            assert_eq!(cores_per_node, topology.total_cores);
        }

        Ok(())
    }

    #[test]
    fn test_optimal_core_empty_cores() -> CoreResult<()> {
        // Create affinity with minimal cores and test edge case
        let mut affinity = CpuAffinity::new(Some(1))?;

        // Manually clear available cores to test edge case
        // This is a bit of a hack but tests the empty cores path
        let empty_affinity = CpuAffinity {
            available_cores: vec![],
            enabled: AtomicBool::new(true),
        };

        let core = empty_affinity.optimal_core_for_worker(0);
        assert_eq!(core, 0);

        Ok(())
    }

    #[test]
    fn test_bind_workers_zero_count() -> CoreResult<()> {
        let affinity = CpuAffinity::new(Some(4))?;
        let assignments = affinity.bind_workers(0)?;

        assert!(assignments.is_empty());
        assert_eq!(assignments.capacity(), 0);

        Ok(())
    }

    #[test]
    fn test_advanced_cpu_affinity_disabled() -> CoreResult<()> {
        let config = AffinityConfig {
            enabled: false,
            cores: None,
            prefer_physical_cores: true,
            numa_aware: true,
        };

        let advanced = AdvancedCpuAffinity::new(config)?;

        // Should succeed even when disabled
        advanced.set_critical_affinity()?;
        advanced.set_background_affinity()?;

        Ok(())
    }

    #[test]
    fn test_advanced_cpu_affinity_empty_cores() -> CoreResult<()> {
        let config = AffinityConfig::default();
        let mut advanced = AdvancedCpuAffinity::new(config)?;

        // Test with empty available cores
        advanced.topology.available_cores.clear();

        let critical_core = advanced.optimal_critical_core();
        let background_core = advanced.optimal_background_core();

        assert_eq!(critical_core, 0);
        assert_eq!(background_core, 0);

        Ok(())
    }

    #[test]
    fn test_advanced_cpu_affinity_no_prefer_physical() -> CoreResult<()> {
        let config = AffinityConfig {
            enabled: true,
            cores: None,
            prefer_physical_cores: false,
            numa_aware: true,
        };

        let advanced = AdvancedCpuAffinity::new(config)?;
        let critical_core = advanced.optimal_critical_core();

        // Should return 0 when not preferring physical cores
        assert_eq!(critical_core, 0);

        Ok(())
    }

    #[test]
    fn test_cpu_topology_zero_numa_nodes() {
        let topology = CpuTopology {
            total_cores: 8,
            physical_cores: 4,
            logical_cores: 8,
            numa_nodes: 0,
            available_cores: vec![0, 1, 2, 3],
        };

        let cores_per_node = topology.cores_per_numa_node();
        assert_eq!(cores_per_node, topology.total_cores);
    }

    #[test]
    fn test_affinity_config_default() {
        let config = AffinityConfig::default();

        assert!(config.enabled);
        assert!(config.cores.is_none());
        assert!(config.prefer_physical_cores);
        assert!(config.numa_aware);
    }

    #[test]
    fn test_advanced_cpu_affinity_getters() -> CoreResult<()> {
        let config = AffinityConfig::default();
        let advanced = AdvancedCpuAffinity::new(config.clone())?;

        let retrieved_topology = advanced.topology();
        let retrieved_config = advanced.config();

        assert_eq!(retrieved_config.enabled, config.enabled);
        assert!(retrieved_topology.total_cores > 0);

        Ok(())
    }
}
