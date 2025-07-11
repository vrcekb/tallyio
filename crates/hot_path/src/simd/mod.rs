//! SIMD optimizations for vectorized operations.

use crate::Result;

// Sub-modules
pub mod price_calc;
pub mod hash_ops;
pub mod search_ops;

// Re-export key types
pub use price_calc::{PriceCalculator, VectorizedPrice};
pub use hash_ops::{HashOperations, SimdHash};
pub use search_ops::{SearchOperations, SearchResult};

/// SIMD capabilities detection
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SimdCapabilities {
    /// SSE4.2 support
    pub sse42: bool,
    /// AVX2 support
    pub avx2: bool,
    /// AVX-512 support
    pub avx512: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities
    #[must_use]
    #[inline]
    pub fn detect() -> Self {
        return Self {
            sse42: cfg!(target_feature = "sse4.2"),
            avx2: cfg!(target_feature = "avx2"),
            avx512: cfg!(target_feature = "avx512f"),
        };
    }

    /// Check if any SIMD features are available
    #[must_use]
    #[inline]
    pub const fn has_simd(&self) -> bool {
        return self.sse42 || self.avx2 || self.avx512;
    }

    /// Get optimal vector width
    #[must_use]
    #[inline]
    pub const fn optimal_vector_width(&self) -> usize {
        if self.avx512 { return 64; }
        if self.avx2 { return 32; }
        if self.sse42 { return 16; }
        return 8;
    }
}

/// Initialize SIMD subsystem
///
/// # Errors
///
/// Returns an error if initialization fails
#[inline]
pub fn initialize() -> Result<()> {
    price_calc::initialize()?;
    hash_ops::initialize()?;
    search_ops::initialize()?;
    return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capabilities_detection() {
        let caps = SimdCapabilities::detect();
        // Test that detection works (may return false in test environment)
        let _has_simd = caps.has_simd();
        // Just verify the structure is working
        assert!(caps.optimal_vector_width() >= 8);
    }

    #[test]
    fn test_simd_initialization() {
        initialize().unwrap();
    }
}
