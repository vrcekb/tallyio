//! Abstrakcije in implementacije za posamezne verige

use crate::error::BlockchainError;
use crate::types::*;
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
        Ok(Block { number: 123456, hash: [0u8; 32] })
    }
    async fn send_transaction(&self, _tx: Transaction) -> Result<TxHash, BlockchainError> {
        Ok([1u8; 32])
    }
    async fn get_balance(&self, _address: Address) -> Result<Balance, BlockchainError> {
        Ok(1000)
    }
}
