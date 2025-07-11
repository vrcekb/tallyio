//! Arena-based allocation for ultra-fast memory management.

use crate::Result;

/// Arena allocator
#[derive(Debug)]
#[non_exhaustive]
pub struct ArenaAllocator {
    /// Total capacity
    capacity: usize,
}

/// Memory arena
#[derive(Debug)]
#[non_exhaustive]
pub struct Arena {
    /// Arena size
    size: usize,
}

impl ArenaAllocator {
    /// Create new arena allocator
    #[must_use]
    #[inline]
    pub const fn new(capacity: usize) -> Self {
        return Self { capacity };
    }

    /// Get allocator capacity
    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        return self.capacity;
    }
}

impl Arena {
    /// Create new arena
    #[must_use]
    #[inline]
    pub const fn new(size: usize) -> Self {
        return Self { size };
    }

    /// Get arena size
    #[must_use]
    #[inline]
    pub const fn size(&self) -> usize {
        return self.size;
    }
}

/// Initialize arena allocator
///
/// # Errors
///
/// Currently returns `Ok(())` but may return errors in future implementations
#[inline]
pub const fn initialize(_max_memory_bytes: usize) -> Result<()> {
    return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocator() {
        let allocator = ArenaAllocator::new(1024);
        assert_eq!(allocator.capacity, 1024);
    }
}
