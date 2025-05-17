//! Integracijski test za MEV komponente in secure_storage
//!
//! Ta test preverja, kako se secure_storage modul obnaša v kontekstu
//! MEV operacij, kjer je kritična nizka latenca in zanesljivo delovanje.

mod common;

use tokio::test;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;
use rand::{Rng, thread_rng};

use secure_storage::{SecureStorage, StorageError};
use tallyio_tests::utils::mev_testing::{
    MevOpportunity, MevOpportunityType, MevTransactionGenerator, BlockchainSimulator
};
use tallyio_tests::utils::performance_testing::{LatencyMeasurement, PerformanceReport};
use tallyio_tests::utils::test_framework::{TestContext, TestEnvironment, assert_latency_under};
use tallyio_tests::utils::coverage_reporting::{TestCoverageCollector, track_line};
use tallyio_tests::utils::test_config::{TestConfig, LogLevel, PerformanceTestLevel};

/// Izvedi integracijski test med MEV komponento in secure_storage
#[test]
async fn test_mev_secure_storage_integration() {
    // Nastavi kolektor pokritosti
    let _coverage = TestCoverageCollector::new("mev_storage_integration")
        .generate_html(true, Some("mev_storage_integration_coverage.html"));
    
    // Ustvari začasno mapo za teste
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let storage_dir = temp_dir.path().to_path_buf();
    
    // Ustvari testno okolje
    let config = TestConfig::new("mev_storage_integration")
        .with_log_level(LogLevel::Debug)
        .with_performance_level(PerformanceTestLevel::Standard)
        .with_timeout(60)
        .with_extra_param("opportunity_count", "100");
    
    let env = TestEnvironment::new(config);
    let ctx = TestContext::new("MEV Storage Integration Test", &env);
    
    ctx.log_info("Initializing secure storage for MEV operations");
    
    // Ustvari secure storage instanco
    let storage = Arc::new(SecureStorage::new(&storage_dir).await
        .expect("Failed to create secure storage"));
    
    // Ustvari generator MEV priložnosti
    let tx_generator = MevTransactionGenerator::new();
    let blockchain_sim = BlockchainSimulator::new();
    
    // Pripravi latency report
    let mut latency_report = PerformanceReport::new("MEV Storage Operations");
    
    // Test 1: Shranjevanje in nalaganje MEV ključev
    ctx.log_info("Testing MEV key storage and retrieval");
    let key_latencies = test_mev_key_operations(&storage, 20).await;
    latency_report.add_measurement("Store MEV Key", key_latencies.store_latency);
    latency_report.add_measurement("Load MEV Key", key_latencies.load_latency);
    
    // Preveri latence
    assert_latency_under!(key_latencies.store_latency.p99(), Duration::from_millis(10));
    assert_latency_under!(key_latencies.load_latency.p99(), Duration::from_millis(5));
    
    // Test 2: Shranjevanje podatkov o MEV priložnostih
    ctx.log_info("Testing MEV opportunity data storage");
    let mut opportunities = Vec::new();
    
    // Generiraj priložnosti in izmeri čas shranjevanja
    let opportunity_store = LatencyMeasurement::new();
    
    for i in 0..100 {
        let opportunity_type = match i % 6 {
            0 => MevOpportunityType::DexArbitrage,
            1 => MevOpportunityType::Sandwich,
            2 => MevOpportunityType::Liquidation,
            3 => MevOpportunityType::Frontrunning,
            4 => MevOpportunityType::Backrunning,
            _ => MevOpportunityType::LongTail,
        };
        
        let opportunity = MevOpportunity::new(
            &format!("opp-{}", i),
            opportunity_type,
            thread_rng().gen_range(1_000_000..1_000_000_000),
        )
        .with_priority(thread_rng().gen_range(1..4))
        .with_data("test_key", "test_value");
        
        opportunities.push(opportunity);
    }
    
    // Shrani podatke o priložnostih
    for opportunity in &opportunities {
        track_line!("mev_integration", "store_opportunity_start");
        
        let start = Instant::now();
        let serialized = serde_json::to_vec(&opportunity)
            .expect("Failed to serialize opportunity");
        
        storage.store(&format!("mev_opportunity_{}", opportunity.id), &serialized).await
            .expect("Failed to store opportunity");
        
        opportunity_store.record(start.elapsed());
        track_line!("mev_integration", "store_opportunity_end");
    }
    
    latency_report.add_measurement("Store MEV Opportunity", opportunity_store);
    assert_latency_under!(opportunity_store.p99(), Duration::from_millis(10));
    
    // Test 3: Simuliraj MEV operacije v proizvodnem okolju
    ctx.log_info("Simulating production MEV environment");
    
    // Ustvari kanale za komunikacijo
    let (tx, mut rx) = tokio::sync::mpsc::channel(100);
    let storage_clone = Arc::clone(&storage);
    
    // Vzporedno spremljaj bloke
    let handle = tokio::spawn(async move {
        let mut block_count = 0;
        let mut opportunity_count = 0;
        
        // Spremljaj bloke in išči priložnosti
        blockchain_sim.simulate_blocks(20, 100, 50, move |block| {
            block_count += 1;
            
            // Za demonstracijo: najdi priložnosti v bloku
            for (i, tx) in block.transactions.iter().enumerate() {
                if i % 10 == 0 {  // Simuliraj, da je 10% transakcij potencialno MEV priložnost
                    opportunity_count += 1;
                    
                    let value = tx.value.min(1_000_000);  // Omejimo vrednost za test
                    
                    let opportunity = MevOpportunity::new(
                        &format!("block-{}-tx-{}", block.block_number, i),
                        MevOpportunityType::DexArbitrage,
                        value as u128,
                    )
                    .with_priority(2)
                    .with_data("block_hash", &block.block_hash)
                    .with_data("tx_hash", &tx.tx_hash);
                    
                    // Pošlji priložnost preko kanala
                    let tx = tx.clone();
                    tokio::spawn(async move {
                        let _ = tx.send(opportunity).await;
                    });
                }
            }
        });
        
        (block_count, opportunity_count)
    });
    
    // Vzporedno procesiraj priložnosti
    let processing_handle = tokio::spawn(async move {
        let mut processed_count = 0;
        let processing_latency = LatencyMeasurement::new();
        
        while let Some(opportunity) = rx.recv().await {
            track_line!("mev_integration", "process_opportunity_start");
            let start = Instant::now();
            
            // Procesiraj priložnost
            let serialized = serde_json::to_vec(&opportunity)
                .expect("Failed to serialize opportunity");
            
            // Shrani priložnost v secure storage
            let store_key = format!("mev_opportunity_{}", opportunity.id);
            storage_clone.store(&store_key, &serialized).await
                .expect("Failed to store opportunity");
            
            // Simuliraj MEV analizo
            tokio::time::sleep(Duration::from_micros(thread_rng().gen_range(100..1000))).await;
            
            // Naloži podatke iz secure storage
            let loaded_data = storage_clone.load(&store_key).await
                .expect("Failed to load opportunity");
            
            let _opportunity: MevOpportunity = serde_json::from_slice(&loaded_data)
                .expect("Failed to deserialize opportunity");
            
            processing_latency.record(start.elapsed());
            processed_count += 1;
            track_line!("mev_integration", "process_opportunity_end");
        }
        
        (processed_count, processing_latency)
    });
    
    // Počakaj na zaključek simulacije
    tokio::time::sleep(Duration::from_secs(3)).await;
    
    // Zaključi
    drop(tx);  // Zapri kanal, da se processing_handle lahko zaključi
    
    let (block_stats, opportunity_stats) = handle.await.expect("Failed to get block stats");
    let (processed_count, processing_latency) = processing_handle.await.expect("Failed to get processing stats");
    
    ctx.log_info(&format!(
        "Processed {} opportunities from {} blocks",
        processed_count, block_stats
    ));
    
    latency_report.add_measurement("Process MEV Opportunity", processing_latency);
    
    // Preveri, da je latenca MEV procesiranja pod 5ms
    assert_latency_under!(processing_latency.p99(), Duration::from_millis(5), 
        "MEV processing latency too high for production use");
    
    // Izpiši poročilo
    ctx.log_info("MEV Storage Integration Test Completed");
    ctx.log_info(&format!("Latency Report:\n{}", latency_report.generate_report()));
}

/// Struktura za latence operacij s ključi
struct KeyOperationLatencies {
    /// Latence shranjevanja
    pub store_latency: LatencyMeasurement,
    /// Latence nalaganja
    pub load_latency: LatencyMeasurement,
}

/// Testiraj shranjevanje in nalaganje MEV ključev
async fn test_mev_key_operations(storage: &SecureStorage, iterations: usize) -> KeyOperationLatencies {
    let mut store_latency = LatencyMeasurement::new();
    let mut load_latency = LatencyMeasurement::new();
    
    for i in 0..iterations {
        let key = format!("mev_key_{}", i);
        let value = generate_random_key();
        
        // Shrani ključ
        track_line!("mev_integration", "store_key_start");
        let start = Instant::now();
        storage.store(&key, &value).await.expect("Failed to store key");
        store_latency.record(start.elapsed());
        track_line!("mev_integration", "store_key_end");
        
        // Naloži ključ
        track_line!("mev_integration", "load_key_start");
        let start = Instant::now();
        let loaded = storage.load(&key).await.expect("Failed to load key");
        load_latency.record(start.elapsed());
        track_line!("mev_integration", "load_key_end");
        
        // Preveri, da sta enaka
        assert_eq!(value, loaded, "Loaded key doesn't match stored key");
    }
    
    KeyOperationLatencies {
        store_latency,
        load_latency,
    }
}

/// Generiraj naključen ključ za testiranje
fn generate_random_key() -> Vec<u8> {
    let mut rng = thread_rng();
    let mut key = vec![0u8; 32];
    rng.fill(&mut key[..]);
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_integration() {
        // Preprost test za preverjanje, da osnovne komponente delujejo
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage_dir = temp_dir.path().to_path_buf();
        
        let storage = SecureStorage::new(&storage_dir).await
            .expect("Failed to create secure storage");
        
        let key = "test_key";
        let value = b"test_value".to_vec();
        
        // Shrani in naloži
        storage.store(key, &value).await.expect("Failed to store");
        let loaded = storage.load(key).await.expect("Failed to load");
        
        assert_eq!(value, loaded);
    }
}
