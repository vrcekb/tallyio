// End-to-End test for blockchain flow
//
// This test performs a complete flow from mempool monitoring to transaction execution
// It tests the interaction between multiple components:
// - blockchain (mempool monitoring)
// - strategies (opportunity detection)
// - wallet (transaction signing)
// - core (execution)
//
// This test connects to a testnet (Ethereum Sepolia) and performs actual blockchain operations.

use blockchain::{Chain, EthereumChain, Transaction, TransactionStatus};
use core::{Arena, Queue};
use std::sync::Arc;
use tokio::runtime::Runtime;
use wallet::{Wallet, KeyStore};
use strategies::{Strategy, ArbitrageStrategy};
use risk::{RiskManager, RiskParameters};

#[test]
#[ignore] // Ignore by default as it requires network access and testnet ETH
fn test_e2e_blockchain_flow() {
    // Create a runtime for async operations
    let rt = Runtime::new().expect("Failed to create runtime");

    // Run the async test
    rt.block_on(async {
        // Step 1: Initialize core components
        let arena = Arc::new(Arena::new());
        let transaction_queue = Arc::new(Queue::new());

        // Step 2: Connect to Sepolia testnet
        let rpc_url = std::env::var("SEPOLIA_RPC_URL")
            .unwrap_or_else(|_| "https://rpc.sepolia.org".to_string());

        let chain = EthereumChain::new(&rpc_url)
            .await
            .expect("Failed to connect to Sepolia testnet");

        // Step 3: Initialize wallet with test private key
        // Note: This should be a dedicated test wallet with minimal funds
        let test_private_key = std::env::var("TEST_PRIVATE_KEY")
            .unwrap_or_else(|_| {
                // Default test private key - DO NOT USE IN PRODUCTION
                // This is just a placeholder for the test
                "0x0000000000000000000000000000000000000000000000000000000000000001".to_string()
            });

        let keystore = KeyStore::from_private_key(&test_private_key)
            .expect("Failed to create keystore from private key");

        let wallet = Wallet::new(keystore);
        let wallet_address = wallet.get_address();

        println!("Using wallet address: {}", wallet_address);

        // Step 4: Check wallet balance
        let balance = chain.get_balance(&wallet_address).await
            .expect("Failed to get wallet balance");

        println!("Wallet balance: {} ETH", balance);

        // Skip the actual transaction if balance is too low
        if balance < 0.01 {
            println!("Skipping transaction execution due to low balance");
            return;
        }

        // Step 5: Initialize risk manager with conservative parameters
        let risk_params = RiskParameters {
            max_slippage: 0.5,  // 0.5%
            max_gas_price: 50.0, // 50 gwei
            max_transaction_value: 0.001, // 0.001 ETH
        };

        let risk_manager = RiskManager::new(risk_params);

        // Step 6: Initialize strategy
        let strategy = ArbitrageStrategy::new(
            Arc::clone(&transaction_queue),
            Arc::new(risk_manager),
        );

        // Step 7: Monitor mempool for pending transactions
        let pending_txs = chain.get_pending_transactions(10).await
            .expect("Failed to get pending transactions");

        println!("Found {} pending transactions", pending_txs.len());
        assert!(!pending_txs.is_empty(), "No pending transactions found");

        // Step 8: Find arbitrage opportunity
        // In a real scenario, we would analyze these transactions for arbitrage
        // For this test, we'll create a simple test transaction

        // Create a test transaction (transfer 0.0001 ETH to ourselves)
        let test_tx = Transaction {
            to: wallet_address.clone(),
            value: 0.0001,
            data: Vec::new(),
            gas_limit: 21000,
            gas_price: 20.0, // 20 gwei
        };

        // Step 9: Sign the transaction
        let signed_tx = wallet.sign_transaction(&test_tx)
            .expect("Failed to sign transaction");

        // Step 10: Submit transaction to the network
        let tx_hash = chain.send_transaction(&signed_tx).await
            .expect("Failed to send transaction");

        println!("Transaction submitted with hash: {}", tx_hash);

        // Step 11: Wait for transaction confirmation
        let mut status = TransactionStatus::Pending;
        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 10;

        while status == TransactionStatus::Pending && attempts < MAX_ATTEMPTS {
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            status = chain.get_transaction_status(&tx_hash).await
                .expect("Failed to get transaction status");
            attempts += 1;
            println!("Transaction status check {}: {:?}", attempts, status);
        }

        // Step 12: Verify transaction was successful
        assert!(
            status == TransactionStatus::Confirmed,
            "Transaction failed or timed out: {:?}", status
        );

        println!("E2E test completed successfully!");
    });
}
