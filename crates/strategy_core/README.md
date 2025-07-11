# Strategy Core

Ultra-performance strategy execution engine for TallyIO MEV and liquidation operations.

## Overview

This crate provides the core strategy execution infrastructure for TallyIO, including liquidation engines, arbitrage systems, zero-risk strategies, time-bandit operations, and coordination systems.

## Performance Targets

- Liquidation detection: <1μs
- Arbitrage opportunity scanning: <500ns  
- Strategy coordination: <1ms
- Profit calculation: <100ns
- Cross-chain operations: <50ns

## Architecture

The crate is organized into priority-based modules:

1. **Liquidation** (Priority 1) - Real-time liquidation engine
2. **Arbitrage** (Priority 2) - Multi-DEX arbitrage systems
3. **Zero Risk** (Priority 3) - Gas optimization and backrunning
4. **Time Bandit** (Priority 4) - Sequencer monitoring and L2 arbitrage
5. **Priority** - ML-based opportunity scoring and resource allocation
6. **Coordination** - Multi-strategy coordination and conflict resolution

## Safety and Performance

All code follows TallyIO's ultra-strict safety standards:
- No `unwrap()`, `expect()`, or `panic!()` in production code
- Result-based error handling with `thiserror`
- Decimal arithmetic using `rust_decimal` for financial calculations
- Lock-free data structures where possible
- NUMA-aware memory allocation
- SIMD-optimized hot paths

## Usage

```rust
use strategy_core::{StrategyConfig, init_strategy_core};

let config = StrategyConfig::default();
init_strategy_core(config)?;
```

## License

MIT
