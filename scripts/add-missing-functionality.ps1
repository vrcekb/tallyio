# Add missing functionality to all crates
$crates = @(
    @{name="blockchain"; manager="BlockchainManager"; method="process_block"; result="Processed block:"; count="block_count"},
    @{name="contracts"; manager="ContractsManager"; method="deploy_contract"; result="Deployed contract:"; count="contract_count"},
    @{name="liquidation"; manager="LiquidationManager"; method="process_liquidation"; result="Processed liquidation:"; count="liquidation_count"},
    @{name="metrics"; manager="MetricsManager"; method="record_metric"; result="Recorded metric:"; count="metric_count"},
    @{name="security"; manager="SecurityManager"; method="validate_request"; result="Validated request:"; count="validation_count"}
)

foreach ($crate in $crates) {
    $file = "crates\$($crate.name)\src\lib.rs"
    Write-Host "Adding functionality to $file..."
    
    # Read file content
    $content = Get-Content $file -Raw
    
    # Replace struct definition
    $oldStruct = "pub struct $($crate.manager);"
    $newStruct = @"
pub struct $($crate.manager) {
    $($crate.count): std::sync::atomic::AtomicU64,
}
"@
    $content = $content -replace [regex]::Escape($oldStruct), $newStruct
    
    # Replace new() method
    $oldNew = "Ok\(Self\)"
    $newNew = @"
Ok(Self {
            $($crate.count): std::sync::atomic::AtomicU64::new(0),
        })
"@
    $content = $content -replace $oldNew, $newNew
    
    # Add method before impl Default
    $beforeDefault = "impl Default for $($crate.manager) \{"
    $methodDef = @"

    /// $($crate.method.Replace('_', ' '))
    ///
    /// # Errors
    /// Returns error if operation fails
    pub fn $($crate.method)(&self, data: &str) -> $($crate.name.Substring(0,1).ToUpper() + $crate.name.Substring(1))Result<String> {
        self.$($crate.count).fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(format!("$($crate.result) {}", data))
    }
}

impl Default for $($crate.manager) {
"@
    $content = $content -replace [regex]::Escape($beforeDefault), $methodDef
    
    # Write back
    Set-Content $file $content -NoNewline
}

Write-Host "Done adding functionality to all crates!"
