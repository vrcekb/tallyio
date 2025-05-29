# Fix clippy allows for all crates
$crates = @("blockchain", "contracts", "database", "liquidation", "metrics", "security", "web-ui")

foreach ($crate in $crates) {
    $file = "crates\$crate\src\lib.rs"
    Write-Host "Fixing $file..."
    
    # Read file content
    $content = Get-Content $file -Raw
    
    # Add allow directive to new() function
    $content = $content -replace "(\s+)pub const fn new\(\) -> (\w+)Result<Self> \{", "`$1#[allow(clippy::unnecessary_wraps)] // API consistency`n`$1pub const fn new() -> `$2Result<Self> {"
    
    # Add allow directive to Default impl
    $content = $content -replace "(\s+)match Self::new\(\) \{", "`$1#[allow(clippy::option_if_let_else)] // Result, not Option`n`$1match Self::new() {"
    
    # Write back
    Set-Content $file $content -NoNewline
}

Write-Host "Done!"
