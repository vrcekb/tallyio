# 🚀 TallyIO Local CI/CD Scripts

Lokalne skripte, ki izvajajo popolnoma iste preverjanja kot GitHub Actions CI/CD pipeline. To omogoča hitro preverjanje pred push-om in zmanjšuje število neuspešnih CI runs.

## 📋 Kaj se preverja

Skripte izvajajo **popolnoma iste korake** kot GitHub Actions:

### ✅ Obvezni koraki
1. **🎨 Code Formatting** - `cargo fmt --all -- --check`
2. **📎 Clippy Linting** - `cargo clippy --all-targets --all-features -- -D warnings`
3. **🚨 Zero Panic Policy** - Preišče `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`
4. **🔧 Build** - `cargo build --all --verbose`
5. **🧪 Unit Tests** - `cargo test --all --lib --verbose`
6. **🔗 Integration Tests** - `cargo test --all --test '*' --verbose`
7. **⚡ Performance Tests** - Latency in benchmark testi
8. **🔒 Security Audit** - `cargo audit --deny warnings`

### 🔧 Opcijski koraki
9. **📊 Code Coverage** - `cargo llvm-cov` (lahko preskočiš z `--skip-coverage`)
10. **🐳 Docker Build** - `docker build` (lahko preskočiš z `--skip-docker`)
11. **📈 Benchmarks** - `cargo bench` (lahko preskočiš z `--skip-benchmarks`)

## 🖥️ Uporaba

### Windows (PowerShell)
```powershell
# Polna CI pipeline
.\scripts\local-ci.ps1

# Hitri način (samo fmt, clippy, testi)
.\scripts\local-ci.ps1 -Fast

# Preskoči coverage
.\scripts\local-ci.ps1 -SkipCoverage

# Preskoči Docker
.\scripts\local-ci.ps1 -SkipDocker

# Pomoč
.\scripts\local-ci.ps1 -Help
```

### Linux/macOS (Bash)
```bash
# Polna CI pipeline
./scripts/local-ci.sh

# Hitri način (samo fmt, clippy, testi)
./scripts/local-ci.sh --fast

# Preskoči coverage
./scripts/local-ci.sh --skip-coverage

# Preskoči Docker
./scripts/local-ci.sh --skip-docker

# Pomoč
./scripts/local-ci.sh --help
```

## ⚡ Hitre možnosti

### 🏃 Fast Mode
```bash
# Izvede samo kritične preverjanja (~30s namesto ~5min)
./scripts/local-ci.sh --fast
.\scripts\local-ci.ps1 -Fast
```

Fast mode izvede:
- ✅ Code formatting
- ✅ Clippy linting  
- ✅ Zero panic check

### 🎯 Kombinacije
```bash
# Hitro + brez Docker-ja
./scripts/local-ci.sh --fast --skip-docker

# Vse razen coverage in benchmarks
./scripts/local-ci.sh --skip-coverage --skip-benchmarks
```

## 📊 Rezultati

Skripte prikazujejo:
- ✅/❌ Status vsakega koraka
- ⏱️ Čas izvajanja
- 📋 Povzetek rezultatov
- 🎉 Ali je vse pripravljeno za GitHub push

### Primer izpisa:
```
🚀 TallyIO Local CI/CD Pipeline Starting...
📅 Started at: 2024-01-15 10:30:00

🔄 Checking prerequisites...
✅ Prerequisites OK

🔄 Checking code formatting...
✅ Code formatting: PASSED

🔄 Running Clippy (Zero warnings policy)...
✅ Clippy: PASSED

🔄 TallyIO Zero Panic Policy Check...
✅ Zero panic policy: PASSED

📋 TallyIO Local CI/CD Summary
================================
⏱️  Duration: 02:45
✅ formatting: PASSED
✅ clippy: PASSED
✅ zero_panic: PASSED
✅ build: PASSED
✅ unit_tests: PASSED
✅ integration_tests: PASSED
✅ performance_tests: PASSED
✅ security: PASSED

📊 Results: 8/8 checks passed
🎉 All checks passed! Ready for GitHub push.
```

## 🔧 Predpogoji

### Obvezno
- **Rust toolchain** (`cargo`, `rustc`)
- **rustfmt** (`rustup component add rustfmt`)
- **clippy** (`rustup component add clippy`)

### Opcijsko (se avtomatsko namesti)
- **cargo-audit** (`cargo install cargo-audit`)
- **cargo-llvm-cov** (`cargo install cargo-llvm-cov`)
- **Docker** (za Docker build test)

## 🎯 TallyIO Specifične zahteve

Skripte preverjajo TallyIO specifične zahteve:
- **<1ms latency** za kritične funkcije
- **Zero panic policy** - prepovedani `unwrap`, `expect`, `panic!`
- **Production-ready** koda brez `todo!`, `unimplemented!`
- **Ultra-strict clippy** pravila

## 💡 Priporočila

### Pre-commit workflow
```bash
# Pred vsakim commit-om
./scripts/local-ci.sh --fast

# Pred push-om
./scripts/local-ci.sh
```

### IDE integracija
Dodaj v VS Code tasks.json:
```json
{
    "label": "TallyIO Local CI",
    "type": "shell",
    "command": "./scripts/local-ci.sh",
    "group": "test",
    "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "new"
    }
}
```

## 🚨 Troubleshooting

### Pogosti problemi:

1. **Permission denied** (Linux/macOS):
   ```bash
   chmod +x scripts/local-ci.sh
   ```

2. **PowerShell execution policy** (Windows):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Missing tools**:
   Skripte avtomatsko namestijo manjkajoče komponente.

4. **Docker not found**:
   Uporabi `--skip-docker` ali namesti Docker.

## 🔄 Sinhronizacija z GitHub Actions

Te skripte so **avtomatsko sinhronizirane** z `.github/workflows/ci.yml`. 
Ko se CI pipeline posodobi, se posodobijo tudi lokalne skripte.

**Garancija**: Če lokalne skripte uspešno preidejo, bo tudi GitHub Actions uspešen! 🎯
