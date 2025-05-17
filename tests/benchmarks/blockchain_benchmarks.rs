//! Performance testi za blockchain modul
//! 
//! Ti testi merijo latenco blockchain operacij in preverjajo,
//! da dosegajo potrebne zahteve za MEV izvajanje.

use blockchain::chain::{Chain, EthereumChain};
use std::time::Duration;
use tokio::runtime::Runtime;

fn test_runtime() -> Runtime {
    Runtime::new().unwrap()
}

/// Meri distribucijo latenc pri pridobivanju blokov
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
    
    // Izračunaj še ostale metrike
    let p50 = latencies[49]; // Median
    let p95 = latencies[94]; // 95th percentile
    let min = latencies[0];
    let max = latencies[99];
    
    println!("Block latency metrics:");
    println!("P50 (median): {:?}", p50);
    println!("P95: {:?}", p95);
    println!("P99: {:?}", p99);
    println!("Min: {:?}", min);
    println!("Max: {:?}", max);
}

/// Meri latenco pošiljanja transakcij
#[test]
fn test_transaction_latency() {
    let rt = test_runtime();
    let chain = EthereumChain;
    let mut latencies = Vec::new();
    
    // Vzorči latenco 100x
    for i in 0..100 {
        let tx = blockchain::types::Transaction {
            from: [i as u8; 20],
            to: [(i+1) as u8; 20],
            value: i,
            data: vec![],
        };
        
        let start = std::time::Instant::now();
        let _ = rt.block_on(chain.send_transaction(tx)).unwrap();
        latencies.push(start.elapsed());
    }
    
    // Izračunaj p99 latenco
    latencies.sort();
    let p99 = latencies[98]; // 99th percentile
    
    // Za MEV operacije je kritična hitra oddaja transakcij
    assert!(p99 < Duration::from_micros(1500), "P99 tx latenca previsoka: {p99:?}");
    
    // Izračunaj še ostale metrike
    let p50 = latencies[49]; // Median
    let p95 = latencies[94]; // 95th percentile
    
    println!("Transaction latency metrics:");
    println!("P50 (median): {:?}", p50);
    println!("P95: {:?}", p95);
    println!("P99: {:?}", p99);
}

/// Benchmark, ki simulira MEV scenarij s hitrim spremljanjem mempool-a in oddajo transakcij
#[test]
fn benchmark_mev_scenario() {
    let rt = test_runtime();
    let chain = EthereumChain;
    
    let start = std::time::Instant::now();
    
    // Simuliraj MEV scenarij:
    // 1. Pridobi trenutni blok
    // 2. Preveri saldo
    // 3. Pošlji transakcijo
    rt.block_on(async {
        let block = chain.current_block().await.unwrap();
        let balance = chain.get_balance([0u8; 20]).await.unwrap();
        
        // Če obstaja priložnost, pošlji transakcijo
        if balance > 0 && block.number > 0 {
            let tx = blockchain::types::Transaction {
                from: [0u8; 20],
                to: [1u8; 20],
                value: 1,
                data: vec![],
            };
            chain.send_transaction(tx).await.unwrap();
        }
    });
    
    let elapsed = start.elapsed();
    
    // Za MEV je kritična celotna izvedba scenarija pod 2ms
    assert!(elapsed < Duration::from_micros(2000), 
            "MEV scenarij prepočasen: {:?}", elapsed);
    println!("MEV end-to-end latenca: {:?}", elapsed);
}
