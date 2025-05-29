---
trigger: always_on
priority: critical
context: tallyio_financial_mev_platform
---

# TallyIO AI COMPILER
## Ultra-performančna finančna MEV platforma
**Zahteve**: <1ms latenca, brez panik, production-ready

## 🚨 ABSOLUTNE PREPOVEDI
```rust
// ❌ NIKOLI
.unwrap() .expect() panic!() .unwrap_or_default() todo!() unimplemented!()
const fn complex_logic() {} // Samo za preproste funkcije!
std::sync::Mutex<T>         // Uporabi atomics
Vec::new()                  // Uporabi Vec::with_capacity()
```

## ✅ OBVEZNI VZORCI

### Error Handling
```rust
// Kritične napake - Copy, fast
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CriticalError { Invalid(u16), OutOfMemory(u16) }

// Nekritične - thiserror
#[derive(thiserror::Error, Debug)]
pub enum ModError {
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("Critical: {0:?}")]
    Critical(#[from] CriticalError),
}

// Vsaka funkcija
pub fn op(&self, data: &Data) -> Result<Output, ModError> {
    let validated = self.validate(data)?;
    self.process(validated)
}
```

### Performance
```rust
// Kritične poti
#[inline(always)]
pub fn critical(&self, x: u64) -> Result<u64, CriticalError> {
    if x == 0 { return Err(CriticalError::Invalid(001)); }
    Ok(x.saturating_mul(2))
}

// Memory + Cache alignment
let mut buffer = Vec::with_capacity(size);
#[repr(C, align(64))]
struct HotData { counter: AtomicU64, flags: AtomicU8 }

// Lock-free
use dashmap::DashMap;
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicU64, Ordering};
```

## 🏗️ MODULE TEMPLATE
```rust
//! Ultra-performančen [MODULE] z <1ms latenco

// Imports - pravilni vrstni red
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam::queue::SegQueue;
use thiserror::Error;
use crate::error::Error as CoreError;

// Error types FIRST
#[derive(Error, Debug)]
pub enum ModError {
    #[error("Critical: {0:?}")]
    Critical(#[from] CriticalError),
    #[error("Invalid: {message}")]
    Invalid { message: String },
}

// Constants
const MAX_CAPACITY: usize = 1024;

// Main struct - hot data first
pub struct Module {
    counter: AtomicU64,    // Hot
    active: AtomicBool,    // Hot
    config: Config,        // Cold
    buffer: SegQueue<Item>, // Cold
}

impl Module {
    pub fn new(config: Config) -> Result<Self, ModError> {
        if config.capacity == 0 {
            return Err(ModError::Invalid { 
                message: "Capacity zero".to_string() 
            });
        }
        Ok(Self {
            counter: AtomicU64::new(0),
            active: AtomicBool::new(true),
            config,
            buffer: SegQueue::new(),
        })
    }
    
    #[inline(always)]
    pub fn process(&self, item: &Item) -> Result<Output, ModError> {
        if !self.active.load(Ordering::Relaxed) {
            return Err(ModError::Invalid { 
                message: "Not active".to_string() 
            });
        }
        let result = self.internal_process(item)?;
        self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    }
    
    fn internal_process(&self, item: &Item) -> Result<Output, ModError> {
        Ok(Output::default())
    }
}
```

## 🧪 TESTIRANJE - HIBRIDNI PRISTOP

### Unit testi - V MODULIH (src/lib.rs)
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_success() -> Result<(), ModError> {
        let m = Module::new(Config::test())?;
        let result = m.process(&Item::default())?;
        assert!(result.is_valid());
        Ok(())
    }
    
    #[test]
    fn test_latency() -> Result<(), ModError> {
        let m = Module::new(Config::test())?;
        let start = std::time::Instant::now();
        m.process(&Item::default())?;
        assert!(start.elapsed() < std::time::Duration::from_millis(1));
        Ok(())
    }
}

// Property tests - tudi v modulih
use proptest::prelude::*;
proptest! {
    #[test]
    fn invariant(x in 1u64..1000) {
        let result = process_value(x).unwrap();
        prop_assert!(result >= x);
    }
}
```

### Integration testi - V MODULIH (tests/ subdirectory)
```rust
// crates/core/tests/integration_test.rs
use tallyio_core::*;

#[test]
fn test_core_integration() -> Result<(), CoreError> {
    let engine = Engine::new(Config::test())?;
    let processor = Processor::new()?;
    
    // Test real interaction
    let result = engine.process_with(&processor, &test_data())?;
    assert!(result.is_complete());
    Ok(())
}
```

### Centralni testi - SAMO za cross-module
```
E:\alpha\Tallyio\tests\
├── e2e\                    # End-to-end testi
├── performance\            # Cross-module performance  
├── security\               # Security & fuzz testi
└── regression\             # Regression testi
```
```

## ⚡ OPTIMIZACIJE
### Memory Layout
```rust
// Hot data first, cache-aligned
#[repr(C, align(64))]
pub struct Optimized {
    counter: AtomicU64,      // 8 bytes
    flags: AtomicU32,        // 4 bytes  
    _pad: [u8; 52],          // Padding to 64
    config: Box<Config>,     // Cold data
}

// Fixed arrays za O(1)
const MAX: usize = 64;
let items: [Option<Item>; MAX] = [None; MAX];
```

### Lock-free Patterns
```rust
pub struct LockFree {
    queue: SegQueue<Request>,
    count: AtomicU64,
}

impl LockFree {
    pub fn try_process(&self) -> Option<Response> {
        if let Some(req) = self.queue.pop() {
            self.count.fetch_add(1, Ordering::AcqRel);
            let resp = self.handle(req);
            self.count.fetch_sub(1, Ordering::AcqRel);
            Some(resp)
        } else { None }
    }
}
```

## 🔒 VARNOST
### Input Validation
```rust
pub fn validate(data: &RawData) -> Result<ValidatedData, ValidationError> {
    if data.len() > MAX_SIZE { return Err(ValidationError::TooLarge); }
    if data.is_empty() { return Err(ValidationError::Empty); }
    
    for byte in data.iter() {
        if !byte.is_ascii() { return Err(ValidationError::InvalidEncoding); }
    }
    
    Ok(ValidatedData { inner: parse_data(data)? })
}
```

### Secure Storage
```rust
use zeroize::Zeroize;

#[derive(Zeroize)]
pub struct SecretKey([u8; 32]);

impl Drop for SecretKey {
    fn drop(&mut self) { self.zeroize(); }
}
```

## 🖥️ SISTEMSKA OPTIMIZACIJA
### CPU Affinity
```rust
use core_affinity;

pub fn configure_affinity() -> Result<(), SystemError> {
    let mev_cores = vec![0, 1, 2, 3];     // P-cores
    let blockchain_cores = vec![4, 5, 6, 7]; // Ločena jedra
    
    std::thread::spawn(move || {
        core_affinity::set_for_current(core_affinity::CoreId { id: mev_cores[0] });
        run_mev_scanner();
    });
    Ok(())
}
```

### Memory Allocator
```toml
# Cargo.toml - OBVEZNO
[dependencies]
jemallocator = "0.5"

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
opt-level = 3
```

```rust
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
```

### Resource Monitoring
```rust
pub struct ResourceMonitor {
    system: System,
    emergency_threshold: f32, // 95% = emergency
}

impl ResourceMonitor {
    #[inline(always)]
    pub fn check(&mut self) -> Result<ResourceStatus, SystemError> {
        self.system.refresh_all();
        let cpu = self.system.global_processor_info().cpu_usage();
        
        if cpu > self.emergency_threshold {
            emergency!("CPU {}% > {}%", cpu, self.emergency_threshold);
            return Ok(ResourceStatus::Emergency);
        }
        Ok(ResourceStatus::Normal)
    }
}
```

### Real-time Scheduling
```rust
use libc::{sched_setscheduler, sched_param, SCHED_FIFO};

pub fn set_realtime() -> Result<(), SystemError> {
    unsafe {
        let param = sched_param { sched_priority: 99 };
        if sched_setscheduler(0, SCHED_FIFO, &param) != 0 {
            return Err(SystemError::RealtimeSetupFailed);
        }
    }
    Ok(())
}
```

## 🚀 WORKFLOW
### Postopna implementacija
```bash
# 1. En modul naenkrat
cargo new --lib my_module

# 2. Implementiraj template
# 3. Clippy OBVEZNO
cargo clippy --all-targets --all-features -- -D warnings

# 4. Testi
cargo test

# 5. Šele potem naslednji modul
```

### Import vrstni red
```rust
// 1. Std library
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// 2. External crates
use crossbeam::queue::SegQueue;
use thiserror::Error;

// 3. Crate imports
use crate::error::Error;
use crate::types::Transaction;

// 4. Parent/sibling
use super::utils;
```

### Pre-submit checklist
```bash
# OBVEZNO pred submit
grep -r "unwrap\|expect\|panic!" src/ # = 0 results
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
cargo bench --all
cargo audit
```

## 🎯 MEV SPECIFIKA
```rust
// Ultra-low latency mempool
pub struct MempoolProcessor {
    queue: SegQueue<Transaction>,
    scanner: OpportunityScanner,
}

impl MempoolProcessor {
    #[inline(always)]
    pub fn scan(&self, tx: &Transaction) -> Result<Option<Opportunity>, Error> {
        let start = Instant::now();
        
        if !tx.is_defi_related() { return Ok(None); }
        
        let opportunity = self.scanner.analyze(tx)?;
        debug_assert!(start.elapsed() < Duration::from_micros(100));
        
        Ok(opportunity)
    }
}
```

## 🏆 SUCCESS METRICS
- ✅ Zero unwrap/expect/panic
- ✅ <1ms latency za kritične funkcije
- ✅ 100% test coverage kritičnih poti
- ✅ Zero clippy warnings
- ✅ Pravilni imports
- ✅ Dokumentirane javne funkcije
- ✅ Cache-optimized memory layout
- ✅ Lock-free kjer možno

**🚨 TallyIO = Financial system. Vsaka linija kode production-ready, ultra-performant, absolutely safe.**