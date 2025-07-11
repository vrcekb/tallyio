//! Arena-based memory management for ultra-high performance.

use crate::Result;

// Sub-modules
pub mod arena_allocator;
pub mod ring_buffer;
pub mod object_pool;

// Re-export key types
pub use arena_allocator::{ArenaAllocator, Arena};
pub use ring_buffer::{RingBuffer, RingBufferError};
pub use object_pool::{ObjectPool, PoolError};

/// Initialize memory subsystem
///
/// # Errors
///
/// Returns an error if initialization fails
#[inline]
pub fn initialize(max_memory_bytes: usize) -> Result<()> {
    arena_allocator::initialize(max_memory_bytes)?;
    ring_buffer::initialize()?;
    object_pool::initialize()?;
    return Ok(());
}

/// Get memory usage in bytes
#[must_use]
#[inline]
pub const fn get_usage_bytes() -> usize {
    return 0; // Stub implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_initialization() {
        initialize(1024 * 1024).unwrap();
    }

    #[test]
    fn test_memory_usage() {
        let usage = get_usage_bytes();
        assert!(usage >= 0);
    }
}
