//! Skupni tipi za blockchain modul

pub type Address = [u8; 20];
pub type TxHash = [u8; 32];
pub type Balance = u128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub number: u64,
    pub hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transaction {
    pub from: Address,
    pub to: Address,
    pub value: Balance,
    pub data: Vec<u8>,
}
