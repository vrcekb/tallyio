//! Enotažni in integracijski testi za blockchain modul

use crate::chain::{Chain, EthereumChain};
use crate::error::BlockchainError;
use crate::types::{Transaction, Block};
use tokio::runtime::Runtime;

fn test_runtime() -> Runtime {
    Runtime::new().unwrap()
}

#[test]
fn test_ethereumchain_current_block_latency() {
    let rt = test_runtime();
    let chain = EthereumChain;
    let start = std::time::Instant::now();
    let block = rt.block_on(chain.current_block()).unwrap();
    let elapsed = start.elapsed();
    // TODO: Optimiziraj latenco pod 100µs v produkciji
    // Za teste dovolimo do 1ms
    assert!(elapsed.as_micros() < 1000, "Previsoka latenca: {elapsed:?}");
    assert_eq!(block.number, 123_456);
}

#[test]
fn test_ethereumchain_send_transaction() {
    let rt = test_runtime();
    let chain = EthereumChain;
    let tx = Transaction { from: [0u8; 20], to: [1u8; 20], value: 42, data: vec![] };
    let hash = rt.block_on(chain.send_transaction(tx)).unwrap();
    assert_eq!(hash, [1u8; 32]);
}

#[test]
fn test_ethereumchain_get_balance() {
    let rt = test_runtime();
    let chain = EthereumChain;
    let address = [0u8; 20];
    let balance = rt.block_on(chain.get_balance(address)).unwrap();
    assert_eq!(balance, 1000);
}

#[test]
fn test_error_handling() {
    // TODO: Implementiraj error handling teste ko bo prava implementacija
    // - Test za network error
    // - Test za invalid data
    // - Test za timeout
    // - Test za graceful degradation
}

/// Performance testi
#[cfg(test)]
mod perf_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_block_latency_distribution() {
        let rt = test_runtime();
        let chain = EthereumChain;
        let mut latencies = Vec::new();

        // Vzorči latenco 100x
        for _ in 0..100 {
            let start = std::time::Instant::now();
            let _ = rt.block_on(chain.current_block()).unwrap();
            latencies.push(start.elapsed());
        }

        // Izračunaj p99 latenco
        latencies.sort();
        let p99 = latencies[98]; // 99th percentile
        
        // TODO: Optimiziraj p99 latenco pod 500µs v produkciji
        // Za teste dovolimo do 2ms
        assert!(p99 < Duration::from_micros(2000), "P99 latenca previsoka: {p99:?}");
    }
}


