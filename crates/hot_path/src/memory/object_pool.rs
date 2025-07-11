//! Pre-allocated object pools for zero-allocation performance.

use crate::Result;

/// Object pool error
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PoolError {
    /// Pool is exhausted
    Exhausted,
}

/// Pre-allocated object pool
#[derive(Debug)]
#[non_exhaustive]
pub struct ObjectPool<T> {
    /// Pool capacity
    capacity: usize,
    /// Phantom data
    _phantom: core::marker::PhantomData<T>,
}

impl<T> ObjectPool<T> {
    /// Create new object pool
    #[must_use]
    #[inline]
    pub const fn new(capacity: usize) -> Self {
        return Self { capacity, _phantom: core::marker::PhantomData };
    }

    /// Get pool capacity
    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        return self.capacity;
    }
}

/// Initialize object pool subsystem
///
/// # Errors
///
/// Currently returns `Ok(())` but may return errors in future implementations
#[inline]
pub const fn initialize() -> Result<()> {
    return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_pool() {
        let pool: ObjectPool<u32> = ObjectPool::new(1024);
        assert_eq!(pool.capacity, 1024);
    }
}
