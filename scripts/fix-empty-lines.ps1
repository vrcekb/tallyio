# Fix empty lines after attributes
$crates = @("blockchain", "contracts", "database", "liquidation", "metrics", "security", "web-ui")

foreach ($crate in $crates) {
    $file = "crates\$crate\src\lib.rs"
    Write-Host "Fixing empty lines in $file..."
    
    # Read file content
    $content = Get-Content $file -Raw
    
    # Remove empty lines after allow attributes
    $content = $content -replace "#\[allow\(clippy::unnecessary_wraps\)\] // API consistency\r?\n\r?\n", "#[allow(clippy::unnecessary_wraps)] // API consistency`n"
    $content = $content -replace "#\[allow\(clippy::option_if_let_else\)\] // Result, not Option\r?\n\r?\n", "#[allow(clippy::option_if_let_else)] // Result, not Option`n"
    
    # Write back
    Set-Content $file $content -NoNewline
}

Write-Host "Done!"
