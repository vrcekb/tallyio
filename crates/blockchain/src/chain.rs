//! Abstrakcije in implementacije za posamezne verige

use crate::error::BlockchainError;
use crate::types::{Address, Balance, Block, Transaction, TxHash};
use async_trait::async_trait;

#[async_trait]
pub trait Chain: Send + Sync {
    /// Pridobi trenutni blok
    async fn current_block(&self) -> Result<Block, BlockchainError>;
    /// Pošlji transakcijo
    async fn send_transaction(&self, tx: Transaction) -> Result<TxHash, BlockchainError>;
    /// Pridobi stanje računa
    async fn get_balance(&self, address: Address) -> Result<Balance, BlockchainError>;
}

// Primer implementacije za EthereumChain (mock)
pub struct EthereumChain;

#[async_trait]
impl Chain for EthereumChain {
    async fn current_block(&self) -> Result<Block, BlockchainError> {
        Ok(Block { number: 123_456, hash: [0u8; 32] })
    }
    async fn send_transaction(&self, _tx: Transaction) -> Result<TxHash, BlockchainError> {
        Ok([1u8; 32])
    }
    async fn get_balance(&self, _address: Address) -> Result<Balance, BlockchainError> {
        Ok(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ethereum_chain_current_block() {
        let chain = EthereumChain;
        let block = chain.current_block().await.unwrap();
        assert_eq!(block.number, 123_456);
        assert_eq!(block.hash, [0u8; 32]);
    }

    #[tokio::test]
    async fn test_ethereum_chain_send_transaction() {
        let chain = EthereumChain;
        let tx = Transaction { from: [1u8; 20], to: [2u8; 20], value: 100, data: vec![0, 1, 2, 3] };
        let tx_hash = chain.send_transaction(tx).await.unwrap();
        assert_eq!(tx_hash, [1u8; 32]);
    }

    #[tokio::test]
    async fn test_ethereum_chain_get_balance() {
        let chain = EthereumChain;
        let address: Address = [3u8; 20];
        let balance = chain.get_balance(address).await.unwrap();
        assert_eq!(balance, 1000);
    }
}
