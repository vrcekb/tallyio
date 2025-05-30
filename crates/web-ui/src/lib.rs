//! `TallyIO` Web UI - Web UI backend

use thiserror::Error;

#[derive(Error, Debug)]
pub enum WebUiError {
    #[error("Core error: {0}")]
    Core(#[from] tallyio_core::CoreError),

    #[error("API error: {0}")]
    Api(#[from] tallyio_api::ApiError),

    #[error("UI error: {0}")]
    Ui(String),
}

pub type WebUiResult<T> = Result<T, WebUiError>;

/// Web UI manager for `TallyIO`
pub struct WebUiManager {
    render_count: std::sync::atomic::AtomicU64,
}

impl WebUiManager {
    /// Create new web UI manager
    ///
    /// # Errors
    /// Currently never fails, but returns Result for future extensibility
    #[allow(clippy::unnecessary_wraps)] // API consistency
    pub const fn new() -> WebUiResult<Self> {
        Ok(Self {
            render_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Render a UI component
    ///
    /// # Errors
    /// Returns error if component rendering fails
    #[allow(clippy::unnecessary_wraps)] // API consistency with other crates
    pub fn render_component(&self, component: &str) -> WebUiResult<String> {
        self.render_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("Rendered component: {component}"))
    }
}

impl Default for WebUiManager {
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
    fn test_webui_manager_creation() -> WebUiResult<()> {
        let manager = WebUiManager::new()?;
        assert_eq!(
            manager
                .render_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn test_webui_manager_default() {
        let manager = WebUiManager::default();
        assert_eq!(
            manager
                .render_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_render_component() -> WebUiResult<()> {
        let manager = WebUiManager::new()?;
        let result = manager.render_component("test_component")?;

        // Verify component was rendered
        assert_eq!(result, "Rendered component: test_component");
        assert_eq!(
            manager
                .render_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        Ok(())
    }

    #[test]
    fn test_webui_latency_requirement() -> WebUiResult<()> {
        let manager = WebUiManager::new()?;
        let start = Instant::now();

        manager.render_component("latency_test")?;

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1,
            "WebUI rendering took {}ms, must be <1ms",
            duration.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_renders() -> WebUiResult<()> {
        let manager = WebUiManager::new()?;

        for i in 0_i32..10_i32 {
            manager.render_component(&format!("component_{i}"))?;
        }

        assert_eq!(
            manager
                .render_count
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
        Ok(())
    }

    #[test]
    fn test_concurrent_renders() -> WebUiResult<()> {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(WebUiManager::new()?);
        let mut handles = vec![];

        for i in 0_i32..5_i32 {
            let manager_clone = Arc::clone(&manager);
            let handle =
                thread::spawn(move || manager_clone.render_component(&format!("concurrent_{i}")));
            handles.push(handle);
        }

        for handle in handles {
            match handle.join() {
                Ok(result) => {
                    result?; // Process the result but ignore the return value
                }
                Err(_) => return Err(WebUiError::Ui("Thread join failed".to_string())),
            }
        }

        assert_eq!(
            manager
                .render_count
                .load(std::sync::atomic::Ordering::Relaxed),
            5
        );
        Ok(())
    }

    #[test]
    fn test_webui_error_display() {
        // Test WebUIError Display implementation (line 57)
        let core_error =
            tallyio_core::CoreError::Critical(tallyio_core::CriticalError::Invalid(505));
        let error = WebUiError::Core(core_error);
        let display_str = format!("{error}");
        assert!(display_str.contains("Core error"));
    }
}
