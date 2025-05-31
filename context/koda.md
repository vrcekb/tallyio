---
trigger: always_on
---

AI Agent Navodila - TallyIO Koda
KONTEKST
TallyIO = ultra-performančna finančna MEV platforma. Zahteve: <1ms latenca, brez panik, absolutna varnost.
ABSOLUTNE PREPOVEDI
rust// ❌ NIKOLI
.unwrap() .expect() panic!() .unwrap_or_default() todo!() unimplemented!()
OBVEZNO
Error Handling
rust// ✅ Vedno Result
fn op() -> Result<Value, Error> { risky_op()? }

// ✅ Kritične napake - Copy
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CriticalError { Invalid(u16), OutOfMemory(u16) }

// ✅ Nekritične - thiserror
#[derive(thiserror::Error, Debug)]
pub enum StdError {
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}
Performance
rust// ✅ Inline kritične poti
#[inline(always)]
pub fn critical(&self, x: u64) -> Result<u64, CriticalError> {
    if x == 0 { return Err(CriticalError::Invalid(001)); }
    Ok(x * 2)
}

// ✅ Predrezerviraj memory
Vec::with_capacity(size)

// ✅ Cache alignment
#[repr(C, align(64))]
struct Aligned { data: [u8; 64] }

// ✅ Lock-free
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicU64, Ordering};
Concurrency
rust// ✅ DashMap, atomics, timeout async
use dashmap::DashMap;
self.count.fetch_add(1, Ordering::Relaxed);
tokio::time::timeout(duration, future).await
STRUKTURA
Module Template
rust//! Opis modula

use std::collections::HashMap;
use crate::error::Error;

// Error types first
#[derive(thiserror::Error, Debug)]
pub enum ModError { #[error("Failed")] Failed }

// Main types
pub struct Main { data: HashMap<String, u64> }

impl Main {
    pub fn new() -> Result<Self, ModError> { Ok(Self { data: HashMap::new() }) }
    pub fn process(&self, x: u64) -> Result<u64, ModError> { Ok(x) }
    fn helper(&self) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() -> Result<(), ModError> {
        let m = Main::new()?;
        assert_eq!(m.process(42)?, 42);
        Ok(())
    }
}
Functions
rust// ✅ Jasni tipi, borrow preference, explicit bounds
pub fn process(&self, tx: &Transaction) -> Result<Receipt, TxError> {}
pub fn generic<T: Clone + Send>(&self, x: T) -> Result<T, Error> {}
Documentation
rust/// Kratka povzetek
/// 
/// Podroben opis.
/// 
/// # Arguments
/// * `x` - Parameter opis
/// 
/// # Returns
/// Opis return vrednosti
/// 
/// # Errors  
/// * `Error::Type` - Kdaj se zgodi
/// 
/// # Examples
/// ```
/// let result = func(param)?;
/// ```
pub fn func(&self, x: Type) -> Result<RetType, Error> {}
TESTING
Osnovni vzorec
rust#[cfg(test)]
mod tests {
    use super::*;
    
    fn test_data() -> Data { Data::default() }
    
    #[test]
    fn test_success() -> Result<(), Error> {
        let p = Processor::new()?;
        let r = p.process(&test_data())?;
        assert!(r.is_valid());
        Ok(())
    }
    
    #[test] 
    fn test_error() -> Result<(), Error> {
        let p = Processor::new()?;
        let r = p.process(&invalid_data());
        assert!(matches!(r, Err(Error::Invalid(_))));
        Ok(())
    }
    
    #[test]
    fn test_latency() -> Result<(), Error> {
        let p = Processor::new()?;
        let start = std::time::Instant::now();
        p.critical_op()?;
        assert!(start.elapsed() < std::time::Duration::from_millis(1));
        Ok(())
    }
}
Property tests
rustuse proptest::prelude::*;
proptest! {
    #[test]
    fn invariant(x in 1u64..1000) {
        let r = process(x).unwrap();
        prop_assert!(r <= x);
    }
}
OPTIMIZACIJE
Memory
rust// ✅ Hot data first, compact types
#[repr(C)]
struct Opt { hot: u64, flag: bool, cold: String }

#[repr(C, packed)]  
struct Compact { id: u32, flags: u8 }
Algorithms
rust// ✅ O(1) arrays za kritične poti
const MAX: usize = 16;
let buffer: [Item; MAX] = [Default::default(); MAX];

// ✅ Batch processing
fn batch(&self, items: &[Item]) -> Vec<Result<Out, Error>> {
    items.iter().map(|i| self.process(i)).collect()
}
PREVERJANJA
Pred submit:

grep -r "unwrap\|expect\|panic!" src/ = 0 results
Kritične funkcije <1ms
Minimizirane heap alokacije
Vse funkcije return Result
Error poti testirane
Javne funkcije dokumentirane
Clippy strict pass

PRIMER
rust//! Ultra-performančni mempool

use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MempoolError {
    #[error("Full")]
    Full,
}

pub struct Mempool {
    queue: SegQueue<Transaction>,
    size: AtomicU64,
    capacity: u64,
}

impl Mempool {
    pub fn new(capacity: u64) -> Result<Self, MempoolError> {
        Ok(Self {
            queue: SegQueue::new(),
            size: AtomicU64::new(0), 
            capacity,
        })
    }
    
    #[inline(always)]
    pub fn push(&self, tx: Transaction) -> Result<(), MempoolError> {
        if self.size.load(Ordering::Relaxed) >= self.capacity {
            return Err(MempoolError::Full);
        }
        self.queue.push(tx);
        self.size.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    #[inline(always)]
    pub fn pop(&self) -> Option<Transaction> {
        self.queue.pop().map(|tx| {
            self.size.fetch_sub(1, Ordering::Relaxed);
            tx
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mempool() -> Result<(), MempoolError> {
        let m = Mempool::new(10)?;
        m.push(Transaction::default())?;
        assert!(m.pop().is_some());
        Ok(())
    }
}
CILJ: Ultra-performančna, varna, panic-free koda za finančni sistem, production ready. Ne uporabljaj mok-ov, razen če je nujno za delovanje aplikacije v produkciji.

Vsi univerzalni testi se shranijo v centralni testni direktorij E:\alpha\Tallyio\tests!!