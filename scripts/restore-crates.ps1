# Restore broken crates by reverting to working state and applying fixes properly
$crates = @(
    @{name="blockchain"; manager="BlockchainManager"; method="process_block"; result="Processed block:"; count="block_count"; error="BlockchainError"},
    @{name="contracts"; manager="ContractsManager"; method="deploy_contract"; result="Deployed contract:"; count="contract_count"; error="ContractsError"},
    @{name="liquidation"; manager="LiquidationManager"; method="process_liquidation"; result="Processed liquidation:"; count="liquidation_count"; error="LiquidationError"},
    @{name="metrics"; manager="MetricsManager"; method="record_metric"; result="Recorded metric:"; count="metric_count"; error="MetricsError"},
    @{name="security"; manager="SecurityManager"; method="validate_request"; result="Validated request:"; count="validation_count"; error="SecurityError"},
    @{name="web-ui"; manager="WebUiManager"; method="render_component"; result="Rendered component:"; count="render_count"; error="WebUiError"}
)

foreach ($crate in $crates) {
    $file = "crates\$($crate.name)\src\lib.rs"
    Write-Host "Restoring $file..." -ForegroundColor Yellow
    
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        
        # Fix the broken function definition line
        $content = $content -replace '#\[allow\(clippy::unnecessary_wraps\)\] // API consistency with other cratespub fn', "#[allow(clippy::unnecessary_wraps)] // API consistency with other crates`n    pub fn"
        
        # Fix inline format strings
        $content = $content -replace 'format!\("([^"]+): \{\}", ([^)]+)\)', 'format!("$1: {$2}")'
        
        # Fix thread join error handling
        $content = $content -replace '\$\{crate\}Error', "$($crate.error)"
        
        # Fix for loop type annotations
        $content = $content -replace 'for i in 0\.\.10 \{', 'for i in 0_i32..10_i32 {'
        $content = $content -replace 'for i in 0\.\.5 \{', 'for i in 0_i32..5_i32 {'
        
        # Fix format strings in loops
        $content = $content -replace 'format!\("(\w+)_\{\}", i\)', 'format!("$1_{i}")'
        $content = $content -replace 'format!\("(\w+)_(\w+)_\{\}", i\)', 'format!("$1_$2_{i}")'
        
        # Write back
        Set-Content $file $content -NoNewline
        Write-Host "Restored $file" -ForegroundColor Green
    } else {
        Write-Host "File $file not found" -ForegroundColor Red
    }
}

Write-Host "Done restoring crates!" -ForegroundColor Green
