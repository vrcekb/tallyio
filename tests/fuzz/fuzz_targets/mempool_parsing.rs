//! Fuzz test za robustno parsiranje mempool podatkov
//! 
//! MEV platforme so posebej občutljive na nepričakovane vhodne podatke iz mempool-a, 
//! saj so to podatki, ki prihajajo iz zunanjega, nepreverjenjega vira.
//! Ta fuzz test zagotavlja robustnost parsiranja in obdelave mempool transakcij.

#![no_main]
use libfuzzer_sys::fuzz_target;
use blockchain::types::{Transaction, RawTransaction};
use blockchain::mempool::{MempoolManager, TransactionParser, ParseResult};
use core::error::Error;
use std::collections::HashMap;
use arbitrary::Arbitrary;

#[derive(Debug, Arbitrary)]
struct FuzzTransaction {
    raw_data: Vec<u8>,
    is_valid: bool,
    gas_price: Option<u64>,
    value: Option<u64>,
    replay_protection: bool,
    input_data_fragments: Vec<Vec<u8>>,
}

/// Custom MempoolManager za testiranje
struct TestMempoolManager {
    rejected_count: usize,
    accepted_count: usize,
    exception_count: usize,
}

impl TestMempoolManager {
    fn new() -> Self {
        Self {
            rejected_count: 0,
            accepted_count: 0,
            exception_count: 0,
        }
    }
}

impl MempoolManager for TestMempoolManager {
    fn process_transaction(&mut self, tx: Result<Transaction, Error>) -> Result<(), Error> {
        match tx {
            Ok(_) => {
                self.accepted_count += 1;
                Ok(())
            }
            Err(_) => {
                self.rejected_count += 1;
                Err(Error::ValidationError("Invalid transaction".to_string()))
            }
        }
    }
    
    fn handle_parser_exception(&mut self, e: Error) {
        self.exception_count += 1;
    }
}

/// Parser za testne transakcije
struct TestTransactionParser;

impl TransactionParser for TestTransactionParser {
    fn parse_transaction(&self, raw_data: &[u8]) -> ParseResult<Transaction> {
        // Enostavna logika za parsiranje - v realnem sistemu bi bila bolj kompleksna
        if raw_data.len() < 10 {
            return Err(Error::ParsingError("Transaction too short".to_string()));
        }
        
        // Simuliramo napačno formatiran RLP/protobuf/itd.
        if raw_data[0] == 0xFF && raw_data[1] == 0xFF {
            return Err(Error::ParsingError("Invalid format".to_string()));
        }
        
        // Simuliramo napake pri deserializaciji
        if raw_data[0] == 0xDE && raw_data[1] == 0xAD {
            return Err(Error::DeserializationError("Failed to deserialize".to_string()));
        }
        
        // Simuliramo napake pri validaciji
        if raw_data.len() > 100 && raw_data[99] == 0xFF {
            return Err(Error::ValidationError("Failed validation".to_string()));
        }
        
        // Če pridemo do sem, je transakcija "veljavna"
        let chain_id = (raw_data[0] as u64) % 5 + 1; // Omejimo na 1-5
        
        Ok(Transaction {
            chain_id,
            from: [raw_data[1]; 20],
            to: [raw_data[2]; 20],
            value: (u64::from_be_bytes([0, 0, 0, 0, raw_data[3], raw_data[4], raw_data[5], raw_data[6]])) * 1_000_000,
            gas_price: (raw_data[7] as u64) * 10,
            gas_limit: 21000 + (raw_data[8] as u64) * 1000,
            input: raw_data[9..].to_vec(),
            nonce: (raw_data[7] as u64),
            ..Default::default()
        })
    }
}

/// Fuzz target funkcija, ki testira robustnost mempool parsiranja
fuzz_target!(|data: Vec<FuzzTransaction>| {
    if data.is_empty() {
        return;
    }
    
    let mut manager = TestMempoolManager::new();
    let parser = TestTransactionParser;
    
    for fuzz_tx in data {
        // Ustvari RawTransaction iz fuzz podatkov
        let mut raw_tx = RawTransaction {
            data: fuzz_tx.raw_data.clone(),
            source: "fuzz_test".to_string(),
            timestamp: std::time::SystemTime::now(),
        };
        
        // Dodaj nekaj posebnih primerov za testiranje robustnosti
        if fuzz_tx.is_valid {
            // Vsebinsko veljavna transakcija
            let tx = parser.parse_transaction(&raw_tx.data);
            let _ = manager.process_transaction(tx);
        } else {
            // Namerno neveljaven format
            let invalid_data = if fuzz_tx.raw_data.len() > 2 {
                let mut modified = fuzz_tx.raw_data.clone();
                modified[0] = 0xFF;
                modified[1] = 0xFF;
                modified
            } else {
                vec![0xFF, 0xFF]
            };
            
            raw_tx.data = invalid_data;
            let tx = parser.parse_transaction(&raw_tx.data);
            let _ = manager.process_transaction(tx);
        }
        
        // Testiraj obdelavo izjem
        if let Some(gas) = fuzz_tx.gas_price {
            if gas > 1000 {
                // Simuliramo izjemo pri procesiranju
                manager.handle_parser_exception(Error::SystemError("Test exception".to_string()));
            }
        }
        
        // Testiraj fragmentirane podatke (kot bi prihajali po mreži)
        for fragment in &fuzz_tx.input_data_fragments {
            if !fragment.is_empty() {
                raw_tx.data = fragment.clone();
                let tx = parser.parse_transaction(&raw_tx.data);
                let _ = manager.process_transaction(tx);
            }
        }
    }
    
    // Če smo prišli do sem, je test uspešen - ni panic-ov ali crashev
});
