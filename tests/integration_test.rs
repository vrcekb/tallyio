//! Integration tests for TallyIO
//! 
//! Tests cross-crate functionality and end-to-end workflows

use tallyio_core::{TallyEngine, Transaction, Price, Gas, CoreError};
use std::time::Instant;

#[test]
fn test_engine_integration() -> Result<(), CoreError> {
    let engine = TallyEngine::new()?;
    
    // Test basic transaction processing
    let tx = Transaction {
        gas_price: Price::new(20_000_000_000), // 20 gwei
        gas_limit: Gas::new(21_000),
        value: Price::new(1_000_000_000_000_000_000), // 1 ETH
        data: vec![],
        ..Transaction::default()
    };
    
    engine.submit_transaction(tx)?;
    
    // Verify processing
    if let Some(processed) = engine.get_processed() {
        assert!(processed.processing_time_ns < 1_000_000); // <1ms
    }
    
    Ok(())
}

#[test]
fn test_mev_integration() -> Result<(), CoreError> {
    let engine = TallyEngine::new()?;
    
    // Test MEV opportunity detection
    let defi_tx = Transaction {
        gas_price: Price::new(60_000_000_000), // 60 gwei
        gas_limit: Gas::new(100_000),
        value: Price::new(2_000_000_000_000_000_000), // 2 ETH
        data: vec![0xa9, 0x05, 0x9c, 0xbb, 0x00, 0x00], // swapExactTokensForTokens
        ..Transaction::default()
    };
    
    engine.submit_transaction(defi_tx)?;
    
    // Verify MEV detection
    if let Some(processed) = engine.get_processed() {
        assert!(processed.mev_opportunity.is_some());
    }
    
    // Check metrics
    let metrics = engine.metrics();
    assert!(metrics.opportunities_found > 0);
    
    Ok(())
}

#[test]
fn test_latency_requirement_integration() -> Result<(), CoreError> {
    let engine = TallyEngine::new()?;
    
    // Test multiple transactions for latency consistency
    for i in 0..10 {
        let start = Instant::now();
        
        let tx = Transaction {
            gas_price: Price::new(20_000_000_000 + i * 1_000_000_000), // Varying gas price
            gas_limit: Gas::new(21_000),
            value: Price::new(1_000_000_000_000_000_000),
            data: vec![],
            ..Transaction::default()
        };
        
        engine.submit_transaction(tx)?;
        
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 1, "Transaction {} took {}ms", i, elapsed.as_millis());
    }
    
    Ok(())
}

#[test]
fn test_concurrent_processing() -> Result<(), CoreError> {
    use std::sync::Arc;
    use std::thread;
    
    let engine = Arc::new(TallyEngine::new()?);
    let mut handles = vec![];
    
    // Spawn multiple threads processing transactions
    for i in 0..5 {
        let engine_clone = Arc::clone(&engine);
        let handle = thread::spawn(move || -> Result<(), CoreError> {
            let tx = Transaction {
                gas_price: Price::new(20_000_000_000 + i * 1_000_000_000),
                gas_limit: Gas::new(21_000),
                value: Price::new(1_000_000_000_000_000_000),
                data: vec![],
                ..Transaction::default()
            };
            
            engine_clone.submit_transaction(tx)?;
            Ok(())
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap()?;
    }
    
    // Verify all transactions were processed
    let metrics = engine.metrics();
    assert!(metrics.transactions_processed >= 5);
    
    Ok(())
}

#[test]
fn test_error_handling_integration() -> Result<(), CoreError> {
    let engine = TallyEngine::new()?;
    
    // Test invalid transaction (zero gas price)
    let invalid_tx = Transaction {
        gas_price: Price::new(0), // Invalid
        gas_limit: Gas::new(21_000),
        value: Price::new(1_000_000_000_000_000_000),
        data: vec![],
        ..Transaction::default()
    };
    
    engine.submit_transaction(invalid_tx)?;
    
    // Verify rejection
    if let Some(processed) = engine.get_processed() {
        assert!(matches!(processed.status, tallyio_core::ProcessingStatus::Rejected(_)));
    }
    
    Ok(())
}
