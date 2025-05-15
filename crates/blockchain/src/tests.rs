//! Enotažni in integracijski testi za blockchain modul

use crate::chain::{Chain, EthereumChain};
use crate::types::Transaction;
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
    // Ultra-nizka latenca: <0.1ms
    assert!(elapsed.as_micros() < 100, "Previsoka latenca: {elapsed:?}");
    assert_eq!(block.number, 123456);
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
    let balance = rt.block_on(chain.get_balance([0u8; 20])).unwrap();
    assert_eq!(balance, 1000);
}
