//! Tests za chain.rs modul

#[cfg(test)]
mod tests {
    use super::super::chain::{Chain, EthereumChain};
    use super::super::types::{Address, Transaction};
    use tokio::test;

    #[test]
    async fn test_ethereum_chain_current_block() {
        let chain = EthereumChain;
        let block = chain.current_block().await.unwrap();
        assert_eq!(block.number, 123_456);
        assert_eq!(block.hash, [0u8; 32]);
    }

    #[test]
    async fn test_ethereum_chain_send_transaction() {
        let chain = EthereumChain;
        let tx = Transaction {
            from: [1u8; 20],
            to: [2u8; 20],
            value: 100,
            data: vec![0, 1, 2, 3],
        };
        let tx_hash = chain.send_transaction(tx).await.unwrap();
        assert_eq!(tx_hash, [1u8; 32]);
    }

    #[test]
    async fn test_ethereum_chain_get_balance() {
        let chain = EthereumChain;
        let address: Address = [3u8; 20];
        let balance = chain.get_balance(address).await.unwrap();
        assert_eq!(balance, 1000);
    }
}
