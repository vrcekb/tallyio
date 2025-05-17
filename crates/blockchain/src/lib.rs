//! Blockchain integracije
//!
//! Ta modul vsebuje integracijo z različnimi blockchain omrežji,
//! vključno z Ethereum, Solana, Polygon, Arbitrum, in Optimism.
//! Funkcionalnost je optimizirana za ultra-nizko latenco, ključno za MEV operacije.

pub mod chain;
pub mod error;
pub mod types;

// Testi so premaknjeni v centralno testno strukturo:
// - E:\alpha\Tallyio\tests\unit\blockchain_tests.rs
// - E:\alpha\Tallyio\tests\integration\mev_arbitrage_scenarios.rs
// - E:\alpha\Tallyio\tests\benchmarks\blockchain_benchmarks.rs
// - E:\alpha\Tallyio\tests\fuzz\fuzz_targets\blockchain_events.rs
