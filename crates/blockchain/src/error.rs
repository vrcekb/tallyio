//! Definicije napak za blockchain modul

use thiserror::Error;

#[derive(Error, Debug)]
pub enum BlockchainError {
    #[error("Network error: {0}")]
    Network(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Timeout")]
    Timeout,
    #[error("Unknown error")]
    Unknown,
}
