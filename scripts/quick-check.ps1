#!/usr/bin/env pwsh
# 🏃 TallyIO Quick Check
# Ultra-fast pre-commit checks (~30 seconds)

$ErrorActionPreference = "Stop"

Write-Host "🏃 TallyIO Quick Check" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date

# 🎨 1. Formatting
Write-Host "🎨 Checking formatting..." -ForegroundColor Yellow
try {
    cargo fmt --all -- --check | Out-Null
    Write-Host "✅ Formatting: OK" -ForegroundColor Green
} catch {
    Write-Host "❌ Formatting: FAILED" -ForegroundColor Red
    Write-Host "   Run: cargo fmt --all" -ForegroundColor Gray
    exit 1
}

# 📎 2. Clippy
Write-Host "📎 Running clippy..." -ForegroundColor Yellow
try {
    cargo clippy --all-targets --all-features -- -D warnings | Out-Null
    Write-Host "✅ Clippy: OK" -ForegroundColor Green
} catch {
    Write-Host "❌ Clippy: FAILED" -ForegroundColor Red
    exit 1
}

# 🚨 3. Zero Panic Check
Write-Host "🚨 Checking zero panic policy..." -ForegroundColor Yellow
$panicCount = 0
$patterns = @("unwrap", "expect", "panic!", "todo!", "unimplemented!")

foreach ($pattern in $patterns) {
    $matches = Select-String -Path "crates\*\src\*.rs" -Pattern $pattern -Recurse -ErrorAction SilentlyContinue
    if ($matches) {
        $panicCount += $matches.Count
    }
}

if ($panicCount -gt 0) {
    Write-Host "❌ Zero panic: FAILED ($panicCount violations)" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ Zero panic: OK" -ForegroundColor Green
}

# 🧪 4. Quick test
Write-Host "🧪 Running quick tests..." -ForegroundColor Yellow
try {
    cargo test --lib | Out-Null
    Write-Host "✅ Tests: OK" -ForegroundColor Green
} catch {
    Write-Host "❌ Tests: FAILED" -ForegroundColor Red
    exit 1
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "🎉 All quick checks passed!" -ForegroundColor Green
Write-Host "⏱️  Duration: $($duration.ToString('mm\:ss'))" -ForegroundColor Gray
Write-Host "🚀 Ready for commit!" -ForegroundColor Cyan
