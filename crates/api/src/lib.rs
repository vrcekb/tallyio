//! `TallyIO` API - REST and WebSocket API

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("HTTP error: {0}")]
    Http(#[from] hyper::Error),

    #[error("WebSocket error: {0}")]
    WebSocket(Box<tungstenite::Error>),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl From<tungstenite::Error> for ApiError {
    fn from(err: tungstenite::Error) -> Self {
        Self::WebSocket(Box::new(err))
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

/// API manager for TallyIO
pub struct ApiManager {
    request_count: std::sync::atomic::AtomicU64,
}

impl ApiManager {
    /// Create new API manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> ApiResult<Self> {
        Ok(Self {
            request_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Handle an API request
    ///
    /// # Errors
    /// Returns error if request handling fails
    pub fn handle_request(&self, request: &str) -> ApiResult<String> {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("Processed: {}", request))
    }
}

impl Default for ApiManager {
    fn default() -> Self {
        // Use match instead of expect to comply with zero-panic policy
        #[allow(clippy::option_if_let_else)] // Result, not Option
        match Self::new() {
            Ok(manager) => manager,
            Err(_) => {
                // This should never happen in normal circumstances
                // If it does, it's a programming error
                std::process::abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_api_manager_creation() -> ApiResult<()> {
        let manager = ApiManager::new()?;
        assert_eq!(
            manager
                .request_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn test_api_manager_default() {
        let manager = ApiManager::default();
        assert_eq!(
            manager
                .request_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_handle_request() -> ApiResult<()> {
        let manager = ApiManager::new()?;
        let result = manager.handle_request("test_request")?;

        // Verify request was processed
        assert_eq!(result, "Processed: test_request");
        assert_eq!(
            manager
                .request_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        Ok(())
    }

    #[test]
    fn test_api_latency_requirement() -> ApiResult<()> {
        let manager = ApiManager::new()?;
        let start = Instant::now();

        manager.handle_request("latency_test")?;

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1,
            "API request took {}ms, must be <1ms",
            duration.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_requests() -> ApiResult<()> {
        let manager = ApiManager::new()?;

        for i in 0..10 {
            manager.handle_request(&format!("request_{}", i))?;
        }

        assert_eq!(
            manager
                .request_count
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
        Ok(())
    }

    #[test]
    fn test_concurrent_requests() -> ApiResult<()> {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(ApiManager::new()?);
        let mut handles = vec![];

        for i in 0..5 {
            let manager_clone = Arc::clone(&manager);
            let handle =
                thread::spawn(move || manager_clone.handle_request(&format!("concurrent_{}", i)));
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap()?;
        }

        assert_eq!(
            manager
                .request_count
                .load(std::sync::atomic::Ordering::Relaxed),
            5
        );
        Ok(())
    }
}
