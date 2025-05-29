# Fix all remaining errors in crates
$fixes = @(
    @{file="crates\liquidation\src\lib.rs"; search='#\[allow\(clippy::unnecessary_wraps\)\] // API consistency with other cratespub fn'; replace="#[allow(clippy::unnecessary_wraps)] // API consistency with other crates`n    pub fn"},
    @{file="crates\liquidation\src\lib.rs"; search='\$\{crate\}Error::Network'; replace='LiquidationError::Strategy'},
    
    @{file="crates\security\src\lib.rs"; search='#\[allow\(clippy::unnecessary_wraps\)\] // API consistency with other cratespub fn'; replace="#[allow(clippy::unnecessary_wraps)] // API consistency with other crates`n    pub fn"},
    @{file="crates\security\src\lib.rs"; search='\$\{crate\}Error::Validation'; replace='SecurityError::Core(tallyio_core::CoreError::Critical(tallyio_core::CriticalError::ResourceExhausted(500)))'},
    
    @{file="crates\metrics\src\lib.rs"; search='#\[allow\(clippy::unnecessary_wraps\)\] // API consistency with other cratespub fn'; replace="#[allow(clippy::unnecessary_wraps)] // API consistency with other crates`n    pub fn"},
    @{file="crates\metrics\src\lib.rs"; search='\$\{crate\}Error::Network'; replace='MetricsError::Core(tallyio_core::CoreError::Critical(tallyio_core::CriticalError::ResourceExhausted(500)))'},
    
    @{file="crates\blockchain\src\lib.rs"; search='#\[allow\(clippy::unnecessary_wraps\)\] // API consistency with other cratespub fn'; replace="#[allow(clippy::unnecessary_wraps)] // API consistency with other crates`n    pub fn"},
    
    @{file="crates\contracts\src\lib.rs"; search='#\[allow\(clippy::unnecessary_wraps\)\] // API consistency with other cratespub fn'; replace="#[allow(clippy::unnecessary_wraps)] // API consistency with other crates`n    pub fn"},
    
    @{file="crates\web-ui\src\lib.rs"; search='#\[allow\(clippy::unnecessary_wraps\)\] // API consistency with other cratespub fn'; replace="#[allow(clippy::unnecessary_wraps)] // API consistency with other crates`n    pub fn"},
    
    @{file="crates\database\src\lib.rs"; search='format!\("Executed query: \{\}", query\)'; replace='format!("Executed query: {query}")'},
    @{file="crates\database\src\lib.rs"; search='format!\("query_\{\}", i\)'; replace='format!("query_{i}")'}
)

foreach ($fix in $fixes) {
    if (Test-Path $fix.file) {
        Write-Host "Fixing $($fix.file)..." -ForegroundColor Yellow
        $content = Get-Content $fix.file -Raw
        $content = $content -replace $fix.search, $fix.replace
        Set-Content $fix.file $content -NoNewline
        Write-Host "Fixed $($fix.file)" -ForegroundColor Green
    }
}

Write-Host "Done fixing all errors!" -ForegroundColor Green
