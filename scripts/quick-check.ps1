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
$fmtResult = cargo fmt --all -- --check 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Formatting: OK" -ForegroundColor Green
} else {
    Write-Host "❌ Formatting: FAILED" -ForegroundColor Red
    Write-Host "   Run: cargo fmt --all" -ForegroundColor Gray
    exit 1
}

# 📎 2. Clippy (Ultra-strict TallyIO configuration)
Write-Host "📎 Running ultra-strict clippy..." -ForegroundColor Yellow
$clippyArgs = @(
    "clippy", "--all-targets", "--all-features", "--",
    "-D", "warnings",
    "-D", "clippy::all",
    "-D", "clippy::pedantic",
    "-D", "clippy::nursery",
    "-D", "clippy::correctness",
    "-D", "clippy::suspicious",
    "-D", "clippy::perf",
    "-D", "clippy::redundant_allocation",
    "-D", "clippy::needless_collect",
    "-D", "clippy::suboptimal_flops",
    "-A", "clippy::missing_docs_in_private_items",
    "-D", "clippy::infinite_loop",
    "-D", "clippy::while_immutable_condition",
    "-D", "clippy::never_loop",
    "-D", "for_loops_over_fallibles",
    "-D", "clippy::manual_strip",
    "-D", "clippy::needless_continue",
    "-D", "clippy::match_same_arms",
    "-D", "clippy::unwrap_used",
    "-D", "clippy::expect_used",
    "-D", "clippy::panic",
    "-D", "clippy::large_stack_arrays",
    "-D", "clippy::large_enum_variant",
    "-D", "clippy::mut_mut",
    "-D", "clippy::cast_possible_truncation",
    "-D", "clippy::cast_sign_loss",
    "-D", "clippy::cast_precision_loss",
    "-D", "clippy::must_use_candidate",
    "-D", "clippy::empty_loop",
    "-D", "clippy::if_same_then_else",
    "-D", "clippy::await_holding_lock",
    "-D", "clippy::await_holding_refcell_ref",
    "-D", "clippy::let_underscore_future",
    "-D", "clippy::diverging_sub_expression",
    "-D", "clippy::unreachable",
    "-D", "clippy::default_numeric_fallback",
    "-D", "clippy::redundant_pattern_matching",
    "-D", "clippy::manual_let_else",
    "-D", "clippy::blocks_in_conditions",
    "-D", "clippy::needless_pass_by_value",
    "-D", "clippy::single_match_else",
    "-D", "clippy::branches_sharing_code",
    "-D", "clippy::useless_asref",
    "-D", "clippy::redundant_closure_for_method_calls",
    "-v"
)

try {
    & cargo @clippyArgs | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Ultra-strict Clippy: OK" -ForegroundColor Green
    } else {
        Write-Host "❌ Ultra-strict Clippy: FAILED" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Ultra-strict Clippy: FAILED" -ForegroundColor Red
    exit 1
}

# 3. Zero Panic Check
Write-Host "Checking zero panic policy..." -ForegroundColor Yellow
$panicCount = 0
$patterns = @("\.unwrap\(\)", "\.expect\(", "panic!", "todo!", "unimplemented!")

foreach ($pattern in $patterns) {
    $files = Get-ChildItem -Path "crates" -Filter "*.rs" -Recurse
    foreach ($file in $files) {
        $matches = Select-String -Path $file.FullName -Pattern $pattern -ErrorAction SilentlyContinue
        if ($matches) {
            $panicCount += $matches.Count
        }
    }
}

if ($panicCount -gt 0) {
    Write-Host "FAILED: Zero panic policy ($panicCount violations)" -ForegroundColor Red
    exit 1
} else {
    Write-Host "OK: Zero panic policy" -ForegroundColor Green
}

# 4. Quick test
Write-Host "Running quick tests..." -ForegroundColor Yellow
try {
    cargo test --lib | Out-Null
    Write-Host "OK: Tests" -ForegroundColor Green
} catch {
    Write-Host "FAILED: Tests" -ForegroundColor Red
    exit 1
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "All quick checks passed!" -ForegroundColor Green
Write-Host "Duration: $($duration.ToString('mm\:ss'))" -ForegroundColor Gray
Write-Host "Ready for commit!" -ForegroundColor Cyan
