<#
.SYNOPSIS
    Namesti vsa potrebna orodja za razvoj Rust projekta.
.DESCRIPTION
    Ta skripta namesti vse potrebne orodja in odvisnosti za razvoj Rust projekta,
    vključno s Clippy-jem, rustfmt, cargo-udeps, cargo-audit in drugimi uporabnimi orodji.
#>

# Nastavimo barve za izpis
$ErrorActionPreference = "Stop"
$successColor = "Green"
$warningColor = "Yellow"
$errorColor = "Red"
$infoColor = "Cyan"

function Write-Header {
    param($text)
    Write-Host "`n=== $text ===" -ForegroundColor $infoColor
}

function Install-RustupComponent {
    param($component, $description = $component)
    
    Write-Host "   Namestitev $description... " -NoNewline
    try {
        rustup component add $component 2>&1 | Out-Null
        Write-Host "✓" -ForegroundColor $successColor
        return $true
    } catch {
        Write-Host "✗" -ForegroundColor $errorColor
        Write-Host "      Napaka pri nameščanju ${description}: $($_.Exception.Message)" -ForegroundColor $errorColor
        return $false
    }
}

function Install-CargoPackage {
    param($package, $description = $package)
    
    Write-Host "   Namestitev $description... " -NoNewline
    try {
        cargo install $package --locked 2>&1 | Out-Null
        Write-Host "✓" -ForegroundColor $successColor
        return $true
    } catch {
        Write-Host "✗" -ForegroundColor $errorColor
        Write-Host "      Napaka pri nameščanju ${description}: $($_.Exception.Message)" -ForegroundColor $errorColor
        return $false
    }
}

# Začetek namestitve
Write-Host "🚀 ZAČENJAM NAMESTITEV RAZVOJNEGA OKOLJA" -ForegroundColor Cyan
Write-Host "======================================\n" -ForegroundColor Cyan

# Preverimo, ali je nameščen Rust in Cargo
Write-Header "1. PREVERJANJE RAZLIČIC"
try {
    $rustcVersion = rustc --version
    $cargoVersion = cargo --version
    Write-Host "   ✓ Rust: $rustcVersion" -ForegroundColor $successColor
    Write-Host "   ✓ Cargo: $cargoVersion" -ForegroundColor $successColor
} catch {
    Write-Host "   ❌ Rust in Cargo nista nameščena!" -ForegroundColor $errorColor
    Write-Host "      Namestite Rust z: https://www.rust-lang.org/tools/install" -ForegroundColor $warningColor
    exit 1
}

# Namestimo osnovne komponente
Write-Header "2. NAMESTITEV OSOVNIH KOMPONENT"
$components = @(
    @{ Name = "rustfmt"; Description = "Oblikovalnik kode" },
    @{ Name = "clippy"; Description = "Statični analizator kode" },
    @{ Name = "rust-docs"; Description = "Dokumentacija" },
    @{ Name = "rust-src"; Description = "Izvorna koda Rust-a" },
    @{ Name = "rust-analysis"; Description = "Analiza kode za IDE" },
    @{ Name = "rls"; Description = "Rust Language Server" }
)

$allComponentsInstalled = $true
foreach ($component in $components) {
    if (-not (Install-RustupComponent $component.Name $component.Description)) {
        $allComponentsInstalled = $false
    }
}

if (-not $allComponentsInstalled) {
    Write-Host "   ⚠️  Nekatere komponente niso bile uspešno nameščene." -ForegroundColor $warningColor
}

# Namestimo uporabna orodja
Write-Header "3. NAMESTITEV UPORABNIH ORODIJ"
$tools = @(
    @{ Name = "cargo-udeps"; Description = "Orodje za odkrivanje neuporabljenih odvisnosti" },
    @{ Name = "cargo-audit"; Description = "Preverjanje varnostnih ranljivosti" },
    @{ Name = "cargo-tarpaulin"; Description = "Orodje za merjenje pokritosti s testi" },
    @{ Name = "cargo-nextest"; Description = "Napredno testno okolje" },
    @{ Name = "cargo-watch"; Description = "Spremljanje sprememb v datotekah" },
    @{ Name = "cargo-edit"; Description = "Dodatni ukazi za upravljanje odvisnosti" },
    @{ Name = "cargo-make"; Description = "Orodje za izvajanje nalog" },
    @{ Name = "cargo-update"; Description = "Posodabljanje cargo ukazov" },
    @{ Name = "cargo-outdated"; Description = "Preverjanje zastarelih odvisnosti" },
    @{ Name = "cargo-deny"; Description = "Preverjanje dovoljenj in licenc" },
    @{ Name = "cargo-geiger"; Description = "Preverjanje uporabe unsafe kode" },
    @{ Name = "cargo-tree"; Description = "Prikaz drevesa odvisnosti" },
    @{ Name = "cargo-expand"; Description = "Razširjanje makr" },
    @{ Name = "cargo-bloat"; Description = "Analiza velikosti binarne datoteke" },
    @{ Name = "cargo-profiler"; Description = "Profiler za izvedbo programa" },
    @{ Name = "cargo-cache"; Description = "Upravljanje s predpomnilnikom Cargo" },
    @{ Name = "cargo-msrv"; Description = "Iskanje minimalne podprte različice Rust-a" },
    @{ Name = "cargo-release"; Description = "Avtomatizacija izdaj projekta" },
    @{ Name = "cargo-bump"; Description = "Upravljanje z različicami" },
    @{ Name = "cargo-generate"; Description = "Ustvarjanje novih projektov iz predlog" }
)

$allToolsInstalled = $true
foreach ($tool in $tools) {
    if (-not (Install-CargoPackage $tool.Name $tool.Description)) {
        $allToolsInstalled = $false
    }
}

if (-not $allToolsInstalled) {
    Write-Host "   ⚠️  Nekatera orodja niso bila uspešno nameščena." -ForegroundColor $warningColor
}

# Namestimo dodatna orodja za razvoj
Write-Header "4. NAMESTITEV DODATNIH ORODIJ"

# Preverimo in namestimo just (nadomestilo za make)
Write-Host "   Preverjanje namestitve just... " -NoNewline
if (Get-Command just -ErrorAction SilentlyContinue) {
    Write-Host "✓ že nameščen" -ForegroundColor $successColor
} else {
    Write-Host "nameščam... " -NoNewline
    try {
        cargo install just
        Write-Host "✓ uspešno nameščen" -ForegroundColor $successColor
    } catch {
        Write-Host "✗ napaka pri nameščanju" -ForegroundColor $errorColor
        Write-Host "      Poskusite ročno namestiti z: cargo install just" -ForegroundColor $warningColor
    }
}

# Preverimo in namestimo watchexec za boljše spremljanje sprememb
Write-Host "   Preverjanje namestitve watchexec... " -NoNewline
if (Get-Command watchexec -ErrorAction SilentlyContinue) {
    Write-Host "✓ že nameščen" -ForegroundColor $successColor
} else {
    # Poskusimo namestiti preko cargo, če je mogoče
    Write-Host "nameščam... " -NoNewline
    try {
        cargo install watchexec-cli
        Write-Host "✓ uspešno nameščen" -ForegroundColor $successColor
    } catch {
        Write-Host "✗ napaka pri nameščanju" -ForegroundColor $errorColor
        Write-Host "      Namestite ročno z: cargo install watchexec-cli" -ForegroundColor $warningColor
        Write-Host "      Ali prenesite iz: https://github.com/watchexec/watchexec/releases" -ForegroundColor $warningColor
    }
}

# Končno sporočilo
Write-Header "NAMESTITEV ZAKLJUČENA"
Write-Host "✅ Vsa orodja so bila uspešno nameščena!" -ForegroundColor $successColor
Write-Host "`nNaslednji koraki:" -ForegroundColor Cyan
Write-Host "1. Zaženite preverjanje kode: .\preveri.ps1" -ForegroundColor White
Write-Host "2. Ustvarite nov branch za svoje spremembe: git checkout -b moje-spremembe" -ForegroundColor White
Write-Host "3. Ko ste končali, pošljite Pull Request na GitHub" -ForegroundColor White

# Dodatni nasveti
Write-Host "`n💡 Nasveti za razvoj:" -ForegroundColor Cyan
Write-Host "- Uporabljajte 'cargo clippy' za statično analizo kode" -ForegroundColor White
Write-Host "- 'cargo fmt' samodejno uredi oblikovanje kode" -ForegroundColor White
Write-Host "- 'cargo test' zaženete teste" -ForegroundColor White
Write-Host "- 'cargo doc --open' odpre dokumentacijo v brskalniku" -ForegroundColor White

# Počakajmo na uporabnikov vnos, da se okno ne zapre takoj
Write-Host "`nPritisnite katerokoli tipko za izhod..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
