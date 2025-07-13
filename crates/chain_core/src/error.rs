//! Error handling for chain core operations
//!
//! This module provides comprehensive error handling with detailed context
//! for debugging and monitoring in production environments.

use rust_decimal::Decimal;
use std::fmt;
use thiserror::Error;

/// Result type alias for chain core operations
pub type Result<T> = std::result::Result<T, ChainCoreError>;

/// Main error type for chain core operations
#[derive(Error, Debug)]
pub enum ChainCoreError {
    /// Network-related errors
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),
    
    /// RPC-related errors
    #[error("RPC error: {0}")]
    Rpc(#[from] RpcError),
    
    /// Transaction-related errors
    #[error("Transaction error: {0}")]
    Transaction(#[from] TransactionError),
    
    /// Gas estimation errors
    #[error("Gas estimation failed: {reason}")]
    GasEstimation { reason: String },
    
    /// Liquidity errors
    #[error("Insufficient liquidity: required {required}, available {available}")]
    InsufficientLiquidity { required: Decimal, available: Decimal },
    
    /// Slippage errors
    #[error("Slippage too high: expected {expected}%, actual {actual}%")]
    SlippageTooHigh { expected: Decimal, actual: Decimal },
    
    /// Deadline errors
    #[error("Deadline exceeded: deadline {deadline}, current {current}")]
    DeadlineExceeded { deadline: u64, current: u64 },
    
    /// Chain support errors
    #[error("Chain not supported: {chain_id}")]
    UnsupportedChain { chain_id: u64 },
    
    /// DEX support errors
    #[error("DEX not supported: {dex_name} on chain {chain_id}")]
    UnsupportedDex { dex_name: String, chain_id: u64 },
    
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    /// Validation errors
    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },
    
    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] SerializationError),
    
    /// Database errors
    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),
    
    /// Cache errors
    #[error("Cache error: {0}")]
    Cache(#[from] CacheError),
    
    /// Authentication errors
    #[error("Authentication error: {message}")]
    Authentication { message: String },
    
    /// Authorization errors
    #[error("Authorization error: {message}")]
    Authorization { message: String },
    
    /// Rate limiting errors
    #[error("Rate limit exceeded: {limit} requests per {window_seconds}s")]
    RateLimit { limit: u32, window_seconds: u32 },
    
    /// Resource exhaustion errors
    #[error("Resource exhausted: {resource} - {message}")]
    ResourceExhausted { resource: String, message: String },
    
    /// Timeout errors
    #[error("Operation timeout: {operation} took longer than {timeout_ms}ms")]
    Timeout { operation: String, timeout_ms: u64 },
    
    /// Internal errors (should be rare in production)
    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Network-related errors
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    
    #[error("Connection refused to {endpoint}")]
    ConnectionRefused { endpoint: String },
    
    #[error("DNS resolution failed for {hostname}")]
    DnsResolution { hostname: String },
    
    #[error("TLS handshake failed: {reason}")]
    TlsHandshake { reason: String },
    
    #[error("HTTP error {status}: {message}")]
    Http { status: u16, message: String },
    
    #[error("WebSocket error: {message}")]
    WebSocket { message: String },
    
    #[error("Connection lost to {endpoint}")]
    ConnectionLost { endpoint: String },
    
    #[error("Invalid URL: {url}")]
    InvalidUrl { url: String },
    
    #[error("Protocol error: {message}")]
    Protocol { message: String },
}

/// RPC-related errors
#[derive(Error, Debug)]
pub enum RpcError {
    #[error("RPC call failed: {method} - {message} (code: {code})")]
    CallFailed { method: String, message: String, code: i32 },
    
    #[error("Invalid response format for method {method}")]
    InvalidResponse { method: String },
    
    #[error("Rate limit exceeded: {requests_per_second} req/s")]
    RateLimit { requests_per_second: u32 },
    
    #[error("Node unavailable: {node_url}")]
    NodeUnavailable { node_url: String },
    
    #[error("Subscription failed: {subscription_type} - {reason}")]
    SubscriptionFailed { subscription_type: String, reason: String },
    
    #[error("Method not supported: {method}")]
    MethodNotSupported { method: String },
    
    #[error("Invalid parameters for {method}: {message}")]
    InvalidParameters { method: String, message: String },
    
    #[error("Node sync error: behind by {blocks_behind} blocks")]
    NodeSyncError { blocks_behind: u64 },
}

/// Transaction-related errors
#[derive(Error, Debug)]
pub enum TransactionError {
    #[error("Transaction reverted: {reason}")]
    Reverted { reason: String },
    
    #[error("Insufficient gas: provided {provided}, required {required}")]
    InsufficientGas { provided: u64, required: u64 },
    
    #[error("Gas price too low: provided {provided} wei, minimum {minimum} wei")]
    GasPriceTooLow { provided: u64, minimum: u64 },
    
    #[error("Nonce too low: provided {provided}, expected {expected}")]
    NonceTooLow { provided: u64, expected: u64 },
    
    #[error("Nonce too high: provided {provided}, expected {expected}")]
    NonceTooHigh { provided: u64, expected: u64 },
    
    #[error("Transaction not found: {hash}")]
    NotFound { hash: String },
    
    #[error("Invalid signature: {reason}")]
    InvalidSignature { reason: String },
    
    #[error("Transaction timeout: {hash} not mined within {timeout_seconds}s")]
    Timeout { hash: String, timeout_seconds: u64 },
    
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance { required: Decimal, available: Decimal },
    
    #[error("Transaction underpriced: {message}")]
    Underpriced { message: String },
    
    #[error("Transaction replaced: {original_hash} replaced by {new_hash}")]
    Replaced { original_hash: String, new_hash: String },
    
    #[error("Transaction failed in simulation: {reason}")]
    SimulationFailed { reason: String },
}

/// Serialization errors
#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("JSON serialization error: {message}")]
    Json { message: String },
    
    #[error("Binary serialization error: {message}")]
    Binary { message: String },
    
    #[error("RLP encoding error: {message}")]
    Rlp { message: String },
    
    #[error("ABI encoding error: {message}")]
    Abi { message: String },
    
    #[error("Invalid data format: expected {expected}, got {actual}")]
    InvalidFormat { expected: String, actual: String },
}

/// Database errors
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection error: {message}")]
    Connection { message: String },
    
    #[error("Query error: {query} - {message}")]
    Query { query: String, message: String },
    
    #[error("Transaction error: {message}")]
    Transaction { message: String },
    
    #[error("Migration error: {version} - {message}")]
    Migration { version: String, message: String },
    
    #[error("Constraint violation: {constraint} - {message}")]
    ConstraintViolation { constraint: String, message: String },
    
    #[error("Data not found: {table}.{id}")]
    NotFound { table: String, id: String },
}

/// Cache errors
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Cache miss: {key}")]
    Miss { key: String },
    
    #[error("Cache connection error: {message}")]
    Connection { message: String },
    
    #[error("Cache serialization error: {key} - {message}")]
    Serialization { key: String, message: String },
    
    #[error("Cache eviction: {key} evicted due to {reason}")]
    Eviction { key: String, reason: String },
    
    #[error("Cache full: cannot store {key}")]
    Full { key: String },
}

impl ChainCoreError {
    /// Create a configuration error
    #[must_use]
    pub fn config<T: Into<String>>(message: T) -> Self {
        Self::Configuration { message: message.into() }
    }
    
    /// Create a validation error
    #[must_use]
    pub fn validation<T: Into<String>, U: Into<String>>(field: T, message: U) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }
    
    /// Create an internal error
    #[must_use]
    pub fn internal<T: Into<String>>(message: T) -> Self {
        Self::Internal { message: message.into() }
    }
    
    /// Create a timeout error
    #[must_use]
    pub fn timeout<T: Into<String>>(operation: T, timeout_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_ms,
        }
    }
    
    /// Check if error is retryable
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self,
            Self::Network(NetworkError::Timeout { .. } | NetworkError::ConnectionRefused { .. } | NetworkError::ConnectionLost { .. }) |
            Self::Rpc(RpcError::NodeUnavailable { .. } | RpcError::RateLimit { .. }) |
            Self::Transaction(TransactionError::Timeout { .. } | TransactionError::Underpriced { .. }) |
            Self::Database(DatabaseError::Connection { .. }) |
            Self::Cache(CacheError::Connection { .. }) |
            Self::Timeout { .. }
        )
    }
    
    /// Check if error is critical (requires immediate attention)
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        matches!(self,
            Self::Internal { .. } |
            Self::Database(DatabaseError::Migration { .. } | DatabaseError::ConstraintViolation { .. }) |
            Self::Authentication { .. } |
            Self::Authorization { .. }
        )
    }
    
    /// Get error category for monitoring
    #[must_use]
    pub const fn category(&self) -> &'static str {
        match self {
            Self::Network(_) => "network",
            Self::Rpc(_) => "rpc",
            Self::Transaction(_) => "transaction",
            Self::GasEstimation { .. } => "gas",
            Self::InsufficientLiquidity { .. } => "liquidity",
            Self::SlippageTooHigh { .. } => "slippage",
            Self::DeadlineExceeded { .. } => "deadline",
            Self::UnsupportedChain { .. } => "chain",
            Self::UnsupportedDex { .. } => "dex",
            Self::Configuration { .. } => "config",
            Self::Validation { .. } => "validation",
            Self::Serialization(_) => "serialization",
            Self::Database(_) => "database",
            Self::Cache(_) => "cache",
            Self::Authentication { .. } => "auth",
            Self::Authorization { .. } => "authz",
            Self::RateLimit { .. } => "rate_limit",
            Self::ResourceExhausted { .. } => "resource",
            Self::Timeout { .. } => "timeout",
            Self::Internal { .. } => "internal",
        }
    }
}

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub chain_id: Option<u64>,
    pub transaction_hash: Option<String>,
    pub block_number: Option<u64>,
    pub timestamp: u64,
    pub additional_data: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    #[must_use]
    pub fn new<T: Into<String>>(operation: T) -> Self {
        Self {
            operation: operation.into(),
            chain_id: None,
            transaction_hash: None,
            block_number: None,
            timestamp: crate::utils::time::now_timestamp(),
            additional_data: std::collections::HashMap::new(),
        }
    }
    
    /// Add chain ID to context
    #[must_use]
    pub const fn with_chain_id(mut self, chain_id: u64) -> Self {
        self.chain_id = Some(chain_id);
        self
    }
    
    /// Add transaction hash to context
    #[must_use]
    pub fn with_transaction_hash<T: Into<String>>(mut self, hash: T) -> Self {
        self.transaction_hash = Some(hash.into());
        self
    }
    
    /// Add block number to context
    #[must_use]
    pub const fn with_block_number(mut self, block_number: u64) -> Self {
        self.block_number = Some(block_number);
        self
    }
    
    /// Add additional data to context
    #[must_use]
    pub fn with_data<T: Into<String>, U: Into<String>>(mut self, key: T, value: U) -> Self {
        self.additional_data.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Operation: {}", self.operation)?;
        
        if let Some(chain_id) = self.chain_id {
            write!(f, ", Chain: {chain_id}")?;
        }
        
        if let Some(ref hash) = self.transaction_hash {
            write!(f, ", Tx: {hash}")?;
        }
        
        if let Some(block) = self.block_number {
            write!(f, ", Block: {block}")?;
        }
        
        if !self.additional_data.is_empty() {
            write!(f, ", Data: {:?}", self.additional_data)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = ChainCoreError::config("Invalid RPC URL");
        assert!(matches!(error, ChainCoreError::Configuration { .. }));
        
        let error = ChainCoreError::validation("gas_limit", "Must be positive");
        assert!(matches!(error, ChainCoreError::Validation { .. }));
        
        let error = ChainCoreError::internal("Unexpected state");
        assert!(matches!(error, ChainCoreError::Internal { .. }));
    }
    
    #[test]
    fn test_error_properties() {
        let timeout_error = ChainCoreError::timeout("rpc_call", 5000);
        assert!(timeout_error.is_retryable());
        assert!(!timeout_error.is_critical());
        assert_eq!(timeout_error.category(), "timeout");
        
        let internal_error = ChainCoreError::internal("Critical failure");
        assert!(!internal_error.is_retryable());
        assert!(internal_error.is_critical());
        assert_eq!(internal_error.category(), "internal");
    }
    
    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("swap_tokens")
            .with_chain_id(1)
            .with_transaction_hash("0x123...")
            .with_block_number(18_000_000)
            .with_data("dex", "uniswap_v3");
        
        assert_eq!(context.operation, "swap_tokens");
        assert_eq!(context.chain_id, Some(1));
        assert_eq!(context.transaction_hash, Some("0x123...".to_string()));
        assert_eq!(context.block_number, Some(18_000_000));
        assert_eq!(context.additional_data.get("dex"), Some(&"uniswap_v3".to_string()));
        
        let display = format!("{context}");
        assert!(display.contains("Operation: swap_tokens"));
        assert!(display.contains("Chain: 1"));
        assert!(display.contains("Tx: 0x123..."));
        assert!(display.contains("Block: 18000000"));
    }
    
    #[test]
    fn test_network_error_conversion() {
        let network_error = NetworkError::Timeout { timeout_ms: 5000 };
        let chain_error: ChainCoreError = network_error.into();
        assert!(matches!(chain_error, ChainCoreError::Network(_)));
    }
    
    #[test]
    fn test_transaction_error_conversion() {
        let tx_error = TransactionError::InsufficientGas {
            provided: 21000,
            required: 25000,
        };
        let chain_error: ChainCoreError = tx_error.into();
        assert!(matches!(chain_error, ChainCoreError::Transaction(_)));
    }
}
