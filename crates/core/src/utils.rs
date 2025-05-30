//! Utility functions for `TallyIO` Core

use crate::{CoreError, CriticalError};

/// CPU affinity utilities for performance optimization
pub mod affinity {
    /// Set CPU affinity to specific core for ultra-low latency
    ///
    /// Binds the current thread to a specific CPU core to reduce context switching
    /// and improve cache locality. Only implemented on Linux systems.
    ///
    /// # Arguments
    /// * `core_id` - CPU core ID to bind to (0-based)
    ///
    /// # Errors
    /// * `CriticalError::Invalid` - If system call fails
    ///
    /// # Returns
    /// `Ok(())` if affinity was set successfully
    #[allow(clippy::unnecessary_wraps)] // Platform-specific implementation may need Result
    #[allow(clippy::missing_const_for_fn)] // Cannot be const due to system calls
    pub fn set_core_affinity(core_id: usize) -> Result<(), crate::CoreError> {
        #[cfg(target_os = "linux")]
        {
            use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
            use std::mem;

            unsafe {
                let mut set: cpu_set_t = mem::zeroed();
                CPU_ZERO(&mut set);
                CPU_SET(core_id, &mut set);

                if sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &set) != 0 {
                    return Err(crate::CoreError::Critical(crate::CriticalError::Invalid(
                        100,
                    )));
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms - no-op
            let _ = core_id;
        }

        Ok(())
    }
}

/// Memory utilities for performance optimization
pub mod memory {
    use std::alloc::{alloc_zeroed, dealloc, Layout};

    /// Allocate cache-aligned memory
    ///
    /// Allocates zero-initialized memory with specified alignment for optimal cache performance.
    /// The alignment will be at least as large as the type's natural alignment.
    ///
    /// # Arguments
    /// * `count` - Number of elements to allocate
    /// * `align` - Minimum alignment requirement (will be max with type alignment)
    ///
    /// # Errors
    /// * `CriticalError::OutOfMemory` - If layout creation fails or allocation fails
    ///
    /// # Returns
    /// Pointer to allocated memory
    ///
    /// # Safety
    /// The returned pointer must be deallocated with `dealloc_aligned` using the same parameters.
    pub fn alloc_aligned<T>(count: usize, align: usize) -> Result<*mut T, crate::CoreError> {
        let layout = Layout::from_size_align(
            std::mem::size_of::<T>() * count,
            align.max(std::mem::align_of::<T>()),
        )
        .map_err(|_| crate::CoreError::Critical(crate::CriticalError::OutOfMemory(200)))?;

        unsafe {
            let ptr = alloc_zeroed(layout).cast::<T>();
            if ptr.is_null() {
                Err(crate::CoreError::Critical(
                    crate::CriticalError::OutOfMemory(201),
                ))
            } else {
                Ok(ptr)
            }
        }
    }

    /// Deallocate aligned memory
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was allocated with `alloc_aligned` with the same `count` and `align`
    /// - `ptr` is not used after this call
    /// - `count` and `align` match the original allocation
    /// - The pointer is not null
    #[allow(clippy::disallowed_methods)]
    pub unsafe fn dealloc_aligned<T>(ptr: *mut T, count: usize, align: usize) {
        let layout = Layout::from_size_align_unchecked(
            std::mem::size_of::<T>() * count,
            align.max(std::mem::align_of::<T>()),
        );
        dealloc(ptr.cast::<u8>(), layout);
    }
}

/// Time utilities for latency measurement
pub mod time {
    use super::{CoreError, CriticalError};
    use std::time::{Duration, Instant};

    /// High-precision timer for latency measurement
    ///
    /// Provides microsecond-precision timing for performance monitoring
    /// and timeout detection in critical paths.
    pub struct LatencyTimer {
        start: Instant,
        max_duration: Duration,
    }

    impl LatencyTimer {
        /// Create new timer with maximum allowed duration
        ///
        /// Starts timing immediately upon creation.
        ///
        /// # Arguments
        /// * `max_duration` - Maximum allowed duration before timeout
        ///
        /// # Returns
        /// New timer instance
        #[must_use]
        pub fn new(max_duration: Duration) -> Self {
            Self {
                start: Instant::now(),
                max_duration,
            }
        }

        /// Check if timer has exceeded maximum duration
        ///
        /// Compares elapsed time against the configured maximum duration.
        ///
        /// # Errors
        /// * `CriticalError::LatencyViolation` - If maximum duration exceeded
        ///
        /// # Returns
        /// `Ok(())` if within time limit
        pub fn check_timeout(&self) -> Result<(), CoreError> {
            if self.start.elapsed() > self.max_duration {
                let exceeded_micros =
                    u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX);
                Err(CoreError::Critical(CriticalError::LatencyViolation(
                    exceeded_micros,
                )))
            } else {
                Ok(())
            }
        }

        /// Get elapsed time in nanoseconds
        ///
        /// Returns elapsed time since timer creation with nanosecond precision.
        /// Saturates at `u64::MAX` if duration is too large.
        ///
        /// # Returns
        /// Elapsed time in nanoseconds
        #[must_use]
        pub fn elapsed_ns(&self) -> u64 {
            u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }

        /// Get elapsed time in microseconds
        ///
        /// Returns elapsed time since timer creation with microsecond precision.
        /// Saturates at `u64::MAX` if duration is too large.
        ///
        /// # Returns
        /// Elapsed time in microseconds
        #[must_use]
        pub fn elapsed_us(&self) -> u64 {
            u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX)
        }
    }
}

/// Hash utilities for fast lookups
pub mod hash {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Fast hash function for hot paths
    ///
    /// Computes a hash using the default hasher for maximum performance.
    /// Suitable for hash tables and quick comparisons.
    ///
    /// # Arguments
    /// * `value` - Value to hash
    ///
    /// # Returns
    /// 64-bit hash value
    #[must_use]
    pub fn fast_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash bytes directly for maximum performance
    ///
    /// Directly hashes a byte slice without additional overhead.
    /// Optimized for raw data hashing in hot paths.
    ///
    /// # Arguments
    /// * `bytes` - Byte slice to hash
    ///
    /// # Returns
    /// 64-bit hash value
    #[must_use]
    pub fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write(bytes);
        hasher.finish()
    }
}

/// Validation utilities
pub mod validation {
    use super::{CoreError, CriticalError};

    /// Validate Ethereum address
    ///
    /// Performs basic validation to ensure the address is not all zeros.
    /// More comprehensive validation should be done at higher levels.
    ///
    /// # Arguments
    /// * `addr` - 20-byte Ethereum address
    ///
    /// # Returns
    /// `true` if address appears valid, `false` otherwise
    #[must_use]
    pub fn is_valid_address(addr: [u8; 20]) -> bool {
        // Basic validation - not all zeros
        !addr.iter().all(|&b| b == 0)
    }

    /// Validate transaction data
    ///
    /// Checks transaction data size against reasonable limits.
    /// Prevents extremely large transactions that could cause memory issues.
    ///
    /// # Arguments
    /// * `data` - Transaction data to validate
    ///
    /// # Errors
    /// * `CriticalError::Invalid` - If data exceeds size limits
    ///
    /// # Returns
    /// `Ok(())` if data is valid
    pub const fn validate_tx_data(data: &[u8]) -> Result<(), CoreError> {
        if data.len() > 1_000_000 {
            // 1MB limit
            return Err(CoreError::Critical(CriticalError::Invalid(400)));
        }
        Ok(())
    }

    /// Validate gas parameters
    ///
    /// Ensures gas price and limit are within reasonable bounds.
    /// Prevents zero gas price and excessive gas limits.
    ///
    /// # Arguments
    /// * `gas_price` - Gas price in wei
    /// * `gas_limit` - Maximum gas to consume
    ///
    /// # Errors
    /// * `CriticalError::Invalid` - If parameters are invalid
    ///
    /// # Returns
    /// `Ok(())` if parameters are valid
    pub const fn validate_gas(gas_price: u64, gas_limit: u64) -> Result<(), CoreError> {
        if gas_price == 0 {
            return Err(CoreError::Critical(CriticalError::Invalid(401)));
        }
        if gas_limit == 0 || gas_limit > 30_000_000 {
            return Err(CoreError::Critical(CriticalError::Invalid(402)));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::missing_errors_doc)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_latency_timer() {
        let timer = time::LatencyTimer::new(Duration::from_millis(1));
        assert!(timer.check_timeout().is_ok());
        assert!(timer.elapsed_ns() > 0);
    }

    #[test]
    fn test_hash_utilities() {
        let data = b"test data";
        let hash1 = hash::hash_bytes(data);
        let hash2 = hash::hash_bytes(data);
        assert_eq!(hash1, hash2);

        // Test that different data produces different hashes
        let different_data = b"different data";
        let hash3 = hash::hash_bytes(different_data);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_validation() {
        let addr = [1u8; 20];
        assert!(validation::is_valid_address(addr));

        let zero_addr = [0u8; 20];
        assert!(!validation::is_valid_address(zero_addr));

        assert!(validation::validate_gas(1000, 21_000).is_ok());
        assert!(validation::validate_gas(0, 21_000).is_err());
        assert!(validation::validate_gas(1000, 0).is_err());
        assert!(validation::validate_gas(1000, 50_000_000).is_err());

        // Test transaction data validation
        let small_data = vec![0u8; 1000];
        assert!(validation::validate_tx_data(&small_data).is_ok());

        let large_data = vec![0u8; 2_000_000];
        assert!(validation::validate_tx_data(&large_data).is_err());
    }

    #[test]
    fn test_latency_timer_timeout() {
        let timer = time::LatencyTimer::new(Duration::from_nanos(1));
        std::thread::sleep(Duration::from_millis(1));
        assert!(timer.check_timeout().is_err());
    }

    #[test]
    fn test_latency_timer_elapsed() {
        let timer = time::LatencyTimer::new(Duration::from_millis(100));
        std::thread::sleep(Duration::from_millis(1));

        let elapsed_nanoseconds = timer.elapsed_ns();
        let elapsed_microseconds = timer.elapsed_us();

        // Allow some tolerance for timing precision
        assert!(elapsed_nanoseconds > 0);
        assert!(elapsed_microseconds > 0);
    }

    #[test]
    fn test_fast_hash() {
        let value1 = "test string";
        let value2 = "test string";
        let value3 = "different string";

        let hash1 = hash::fast_hash(&value1);
        let hash2 = hash::fast_hash(&value2);
        let hash3 = hash::fast_hash(&value3);

        assert_eq!(hash1, hash2); // Same values should have same hash
        assert_ne!(hash1, hash3); // Different values should have different hash
    }

    #[test]
    fn test_memory_alignment() -> Result<(), CoreError> {
        // Test that our alignment utilities work
        let ptr = memory::alloc_aligned::<u64>(8, 64)?;
        assert!(!ptr.is_null());

        // Check alignment
        let addr = ptr as usize;
        assert_eq!(addr % 64, 0); // Should be 64-byte aligned

        // Clean up
        unsafe {
            memory::dealloc_aligned(ptr, 8, 64);
        }
        Ok(())
    }

    #[test]
    fn test_validation_edge_cases() {
        // Test gas limit at boundary
        assert!(validation::validate_gas(1, 30_000_000).is_ok()); // Max allowed
        assert!(validation::validate_gas(1, 30_000_001).is_err()); // Over limit

        // Test data at boundary
        let boundary_data = vec![0u8; 1_000_000]; // Exactly at limit
        assert!(validation::validate_tx_data(&boundary_data).is_ok());

        let over_boundary = vec![0u8; 1_000_001]; // Over limit
        assert!(validation::validate_tx_data(&over_boundary).is_err());
    }

    #[test]
    fn test_core_affinity() {
        // Test core affinity setting (should work on all platforms)
        let result = affinity::set_core_affinity(0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_allocation_failure() {
        // Test allocation with invalid parameters that would cause overflow
        let result = memory::alloc_aligned::<u64>(usize::MAX / 8, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_allocation_success() -> Result<(), CoreError> {
        // Test successful allocation to cover lines 79-82
        let ptr = memory::alloc_aligned::<u32>(4, 32)?;
        assert!(!ptr.is_null());

        // Verify the pointer is properly aligned
        let addr = ptr as usize;
        assert_eq!(addr % 32, 0);

        // Clean up
        unsafe {
            memory::dealloc_aligned(ptr, 4, 32);
        }
        Ok(())
    }

    #[test]
    fn test_memory_allocation_layout_error() {
        // Test allocation with invalid layout to cover line 77
        let result = memory::alloc_aligned::<u8>(usize::MAX, usize::MAX);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(CoreError::Critical(CriticalError::OutOfMemory(200)))
        ));
    }

    #[test]
    fn test_memory_allocation_null_ptr() {
        // Test allocation that would return null pointer
        // This is hard to test reliably, but we can test with extreme values
        let result = memory::alloc_aligned::<u64>(usize::MAX / 16, 64);
        // This should either succeed or fail with OutOfMemory
        match result {
            Ok(ptr) => {
                // If it succeeds, clean up
                unsafe {
                    memory::dealloc_aligned(ptr, usize::MAX / 16, 64);
                }
            }
            Err(CoreError::Critical(CriticalError::OutOfMemory(_))) => {
                // Expected failure
            }
            Err(_) => {
                // Other errors are also acceptable for this edge case test
            }
        }
    }

    #[test]
    fn test_affinity_linux_path() {
        // Test the Linux-specific path (will be no-op on other platforms)
        let result = affinity::set_core_affinity(0);
        assert!(result.is_ok());

        // Test with a higher core ID
        let result = affinity::set_core_affinity(1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_timer_elapsed_overflow() {
        // Test timer with very long duration to test overflow handling
        let timer = time::LatencyTimer::new(Duration::from_secs(3600));

        // Add a small delay to ensure measurable time
        std::thread::sleep(Duration::from_millis(1));

        // The elapsed time should be small, so no overflow
        let elapsed_ns = timer.elapsed_ns();
        let elapsed_us = timer.elapsed_us();

        assert!(elapsed_ns > 0);
        // Test that the conversion functions work correctly
        // Note: Due to precision differences, we just verify they're reasonable
        assert!(elapsed_ns < u64::MAX);
        assert!(elapsed_us < u64::MAX);
    }

    #[test]
    fn test_validation_comprehensive() {
        // Test all validation functions comprehensively

        // Test address validation edge cases
        let mut partial_zero = [0u8; 20];
        partial_zero[19] = 1; // Only last byte is non-zero
        assert!(validation::is_valid_address(partial_zero));

        // Test gas validation with minimum values
        assert!(validation::validate_gas(1, 1).is_ok());

        // Test transaction data validation with empty data
        assert!(validation::validate_tx_data(&[]).is_ok());

        // Test with maximum allowed data size
        let max_data = vec![0u8; 1_000_000];
        assert!(validation::validate_tx_data(&max_data).is_ok());
    }
}
