//! Vectorized searching operations using SIMD.

use crate::Result;

/// Search result
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SearchResult {
    /// Found position
    pub position: usize,
    /// Match confidence
    pub confidence: u8,
}

/// SIMD search operations
#[derive(Debug)]
#[non_exhaustive]
pub struct SearchOperations {
    /// Search algorithm
    algorithm: u8,
}

impl SearchOperations {
    /// Create new search operations
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        return Self { algorithm: 1 };
    }

    /// Get search algorithm
    #[must_use]
    #[inline]
    pub const fn algorithm(&self) -> u8 {
        return self.algorithm;
    }
}

impl Default for SearchOperations {
    #[inline]
    fn default() -> Self {
        return Self::new();
    }
}

/// Initialize search operations
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
    fn test_search_operations() {
        let ops = SearchOperations::new();
        assert_eq!(ops.algorithm, 1);
    }
}
