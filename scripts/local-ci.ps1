#!/usr/bin/env pwsh
# 🚀 TallyIO Local CI/CD Pipeline
# Runs the same checks as GitHub Actions locally

param(
    [switch]$SkipCoverage,
    [switch]$SkipDocker,
    [switch]$SkipBenchmarks,
    [switch]$Fast,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
🚀 TallyIO Local CI/CD Pipeline

Usage: .\scripts\local-ci.ps1 [OPTIONS]

Options:
  -SkipCoverage    Skip code coverage analysis
  -SkipDocker      Skip Docker build test
  -SkipBenchmarks  Skip performance benchmarks
  -Fast            Run only essential checks (fmt, clippy, tests)
  -Help            Show this help message

Examples:
  .\scripts\local-ci.ps1                    # Full CI pipeline
  .\scripts\local-ci.ps1 -Fast              # Quick checks only
  .\scripts\local-ci.ps1 -SkipCoverage      # Skip coverage
"@
    exit 0
}

# 🎯 TallyIO Performance Requirements
$env:CARGO_TERM_COLOR = "always"
$env:RUST_BACKTRACE = "1"
$env:TALLYIO_MAX_LATENCY_MS = "1"
$env:TALLYIO_ZERO_PANIC = "true"

$ErrorActionPreference = "Stop"
$startTime = Get-Date

Write-Host "🚀 TallyIO Local CI/CD Pipeline Starting..." -ForegroundColor Cyan
Write-Host "📅 Started at: $startTime" -ForegroundColor Gray
Write-Host ""

# 📊 Track results
$results = @{
    "Formatting" = $false
    "Clippy" = $false
    "ZeroPanic" = $false
    "Build" = $false
    "UnitTests" = $false
    "IntegrationTests" = $false
    "PerformanceTests" = $false
    "Security" = $false
    "Coverage" = $false
    "Docker" = $false
    "Benchmarks" = $false
}

function Write-Step {
    param($Message)
    Write-Host "🔄 $Message" -ForegroundColor Yellow
}

function Write-Success {
    param($Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param($Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param($Message)
    Write-Host "⚠️  $Message" -ForegroundColor DarkYellow
}

function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# 🔍 Prerequisites check
Write-Step "Checking prerequisites..."

if (-not (Test-Command "cargo")) {
    Write-Error "Cargo not found. Please install Rust toolchain."
    exit 1
}

if (-not (Test-Command "rustfmt")) {
    Write-Warning "rustfmt not found. Installing..."
    rustup component add rustfmt
}

if (-not (Test-Command "cargo-clippy")) {
    Write-Warning "clippy not found. Installing..."
    rustup component add clippy
}

Write-Success "Prerequisites OK"
Write-Host ""

# 🎨 1. Code Formatting Check
Write-Step "Checking code formatting..."
try {
    cargo fmt --all -- --check
    $results["Formatting"] = $true
    Write-Success "Code formatting: PASSED"
} catch {
    Write-Error "Code formatting: FAILED"
    Write-Host "Run 'cargo fmt --all' to fix formatting issues." -ForegroundColor Gray
}
Write-Host ""

# 📎 2. Clippy Linting (Zero warnings policy)
Write-Step "Running Clippy (Zero warnings policy)..."
try {
    cargo clippy --all-targets --all-features -- -D warnings
    $results["Clippy"] = $true
    Write-Success "Clippy: PASSED"
} catch {
    Write-Error "Clippy: FAILED"
    Write-Host "Fix all clippy warnings before proceeding." -ForegroundColor Gray
}
Write-Host ""

# 🚨 3. TallyIO Zero Panic Check
Write-Step "TallyIO Zero Panic Policy Check..."
try {
    $panicPatterns = @("unwrap", "expect", "panic!", "todo!", "unimplemented!")
    $panicCount = 0
    $foundFiles = @()
    
    foreach ($pattern in $panicPatterns) {
        $matches = Select-String -Path "crates\*\src\*.rs" -Pattern $pattern -Recurse
        if ($matches) {
            $panicCount += $matches.Count
            $foundFiles += $matches
        }
    }
    
    if ($panicCount -gt 0) {
        Write-Error "Found $panicCount prohibited patterns:"
        foreach ($match in $foundFiles) {
            Write-Host "  $($match.Filename):$($match.LineNumber): $($match.Line.Trim())" -ForegroundColor Red
        }
        throw "Zero panic policy violated"
    }
    
    $results["ZeroPanic"] = $true
    Write-Success "Zero panic policy: PASSED"
} catch {
    Write-Error "Zero panic policy: FAILED"
}
Write-Host ""

if ($Fast) {
    Write-Host "🏃 Fast mode enabled - skipping remaining checks" -ForegroundColor Yellow
    Write-Host ""
    goto Summary
}

# 🔧 4. Build All Crates
Write-Step "Building all crates..."
try {
    cargo build --all --verbose
    $results["Build"] = $true
    Write-Success "Build: PASSED"
} catch {
    Write-Error "Build: FAILED"
}
Write-Host ""

# 🧪 5. Unit Tests
Write-Step "Running unit tests..."
try {
    cargo test --all --lib --verbose
    $results["UnitTests"] = $true
    Write-Success "Unit tests: PASSED"
} catch {
    Write-Error "Unit tests: FAILED"
}
Write-Host ""

# 🔗 6. Integration Tests
Write-Step "Running integration tests..."
try {
    cargo test --all --test '*' --verbose
    $results["IntegrationTests"] = $true
    Write-Success "Integration tests: PASSED"
} catch {
    Write-Error "Integration tests: FAILED"
}
Write-Host ""

# ⚡ 7. Performance & Latency Tests
Write-Step "Running TallyIO performance tests..."
try {
    Write-Host "🚀 Running latency requirement tests..." -ForegroundColor Gray
    cargo test --all --release test_latency_requirement -- --nocapture
    
    Write-Host "🚀 Running benchmark tests..." -ForegroundColor Gray
    cargo test --all --release benchmark -- --nocapture
    
    $results["PerformanceTests"] = $true
    Write-Success "Performance tests: PASSED"
} catch {
    Write-Error "Performance tests: FAILED"
}
Write-Host ""

# 🔒 8. Security Audit
Write-Step "Running security audit..."
try {
    if (-not (Test-Command "cargo-audit")) {
        Write-Warning "cargo-audit not found. Installing..."
        cargo install cargo-audit
    }
    
    cargo audit
    cargo audit --deny warnings
    $results["Security"] = $true
    Write-Success "Security audit: PASSED"
} catch {
    Write-Error "Security audit: FAILED"
}
Write-Host ""

# 📊 9. Code Coverage (Optional)
if (-not $SkipCoverage) {
    Write-Step "Generating code coverage report..."
    try {
        if (-not (Test-Command "cargo-llvm-cov")) {
            Write-Warning "cargo-llvm-cov not found. Installing..."
            cargo install cargo-llvm-cov
        }
        
        cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
        $results["Coverage"] = $true
        Write-Success "Code coverage: PASSED"
        Write-Host "📊 Coverage report saved to: lcov.info" -ForegroundColor Gray
    } catch {
        Write-Error "Code coverage: FAILED"
    }
    Write-Host ""
}

# 🐳 10. Docker Build (Optional)
if (-not $SkipDocker) {
    Write-Step "Testing Docker build..."
    try {
        if (-not (Test-Command "docker")) {
            Write-Warning "Docker not found. Skipping Docker build test."
        } else {
            docker build -t tallyio:local-test .
            $results["Docker"] = $true
            Write-Success "Docker build: PASSED"
        }
    } catch {
        Write-Error "Docker build: FAILED"
    }
    Write-Host ""
}

# 📈 11. Benchmarks (Optional)
if (-not $SkipBenchmarks) {
    Write-Step "Running performance benchmarks..."
    try {
        cargo bench --all
        $results["Benchmarks"] = $true
        Write-Success "Benchmarks: PASSED"
    } catch {
        Write-Error "Benchmarks: FAILED"
    }
    Write-Host ""
}

:Summary
# 📋 Summary Report
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "📋 TallyIO Local CI/CD Summary" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "⏱️  Duration: $($duration.ToString('mm\:ss'))" -ForegroundColor Gray
Write-Host ""

$passed = 0
$total = 0

foreach ($check in $results.GetEnumerator()) {
    $total++
    if ($check.Value) {
        $passed++
        Write-Host "✅ $($check.Key): PASSED" -ForegroundColor Green
    } else {
        Write-Host "❌ $($check.Key): FAILED" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📊 Results: $passed/$total checks passed" -ForegroundColor $(if ($passed -eq $total) { "Green" } else { "Red" })

if ($passed -eq $total) {
    Write-Host "🎉 All checks passed! Ready for GitHub push." -ForegroundColor Green
    exit 0
} else {
    Write-Host "💥 Some checks failed. Fix issues before pushing." -ForegroundColor Red
    exit 1
}
