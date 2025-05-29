# Add comprehensive tests to all crates
$crates = @(
    @{name="contracts"; manager="ContractsManager"; method="deploy_contract"; result="Deployed contract:"; count="contract_count"},
    @{name="database"; manager="DatabaseManager"; method="execute_query"; result="Executed query:"; count="query_count"},
    @{name="liquidation"; manager="LiquidationManager"; method="process_liquidation"; result="Processed liquidation:"; count="liquidation_count"},
    @{name="metrics"; manager="MetricsManager"; method="record_metric"; result="Recorded metric:"; count="metric_count"},
    @{name="security"; manager="SecurityManager"; method="validate_request"; result="Validated request:"; count="validation_count"},
    @{name="web-ui"; manager="WebUiManager"; method="render_component"; result="Rendered component:"; count="render_count"}
)

foreach ($crate in $crates) {
    $file = "crates\$($crate.name)\src\lib.rs"
    Write-Host "Adding tests to $file..."
    
    $testModule = @"

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_$($crate.name)_manager_creation() -> $($crate.name.Substring(0,1).ToUpper() + $crate.name.Substring(1))Result<()> {
        let manager = $($crate.manager)::new()?;
        assert_eq!(manager.$($crate.count).load(std::sync::atomic::Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn test_$($crate.name)_manager_default() {
        let manager = $($crate.manager)::default();
        assert_eq!(manager.$($crate.count).load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn test_$($crate.method)() -> $($crate.name.Substring(0,1).ToUpper() + $crate.name.Substring(1))Result<()> {
        let manager = $($crate.manager)::new()?;
        let result = manager.$($crate.method)("test_data")?;
        
        // Verify operation was processed
        assert_eq!(result, "$($crate.result) test_data");
        assert_eq!(manager.$($crate.count).load(std::sync::atomic::Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn test_$($crate.name)_latency_requirement() -> $($crate.name.Substring(0,1).ToUpper() + $crate.name.Substring(1))Result<()> {
        let manager = $($crate.manager)::new()?;
        let start = Instant::now();
        
        manager.$($crate.method)("latency_test")?;
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 1, "$($crate.name) operation took {}ms, must be <1ms", duration.as_millis());
        Ok(())
    }

    #[test]
    fn test_multiple_operations() -> $($crate.name.Substring(0,1).ToUpper() + $crate.name.Substring(1))Result<()> {
        let manager = $($crate.manager)::new()?;
        
        for i in 0..10 {
            manager.$($crate.method)(&format!("operation_{}", i))?;
        }
        
        assert_eq!(manager.$($crate.count).load(std::sync::atomic::Ordering::Relaxed), 10);
        Ok(())
    }

    #[test]
    fn test_concurrent_operations() -> $($crate.name.Substring(0,1).ToUpper() + $crate.name.Substring(1))Result<()> {
        use std::sync::Arc;
        use std::thread;
        
        let manager = Arc::new($($crate.manager)::new()?);
        let mut handles = vec![];
        
        for i in 0..5 {
            let manager_clone = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                manager_clone.$($crate.method)(&format!("concurrent_{}", i))
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap()?;
        }
        
        assert_eq!(manager.$($crate.count).load(std::sync::atomic::Ordering::Relaxed), 5);
        Ok(())
    }
}
"@

    # Append test module to file
    Add-Content -Path $file -Value $testModule
}

Write-Host "Done adding tests to all crates!"
