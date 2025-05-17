# TallyIO - Razvojna navodila

## 0. Uvodne zahteve
- Preberi E:\alpha\Tallyio\Tallyio načrt.md in shrani v spomin
- Preberi E:\alpha\Tallyio\avodila.md in shrani v spomin

## 1. Performančne zahteve
- Vsaka kritična pot MORA imeti latenco pod 0.1ms
- Uporabi zero-allocation design na kritičnih poteh
- Implementiraj lock-free algoritme namesto mutex-ov
- Izogibaj se heap alokacijam v kritičnih sekcijah
- Uporabljaj Rust-ove arena allocatorje (npr. Bumpalo) za učinkovito upravljanje s spominom

## 2. Varnostne prakse
- NIKOLI ne uporabljaj `unwrap()` ali `expect()` razen če je absolutno nujno
- Vedno implementiraj proper error handling z uporabo `Result` in `Option`
- Vse zunanje vhodne podatke validiraj in sanitiziraj
- Občutljive podatke (API ključi, privatni ključi) shranjuj v `secure_storage` modul
- Uporabljaj kriptografske primitive iz preverjenih knjižnic (ring, openssl)

## 3. Kodne konvencije
- Po vsaki spremembi poženi E:\alpha\Tallyio\preveri.py
- Vse javne funkcije MORAJO imeti dokumentacijo
- Implementiraj teste za vsako novo funkcionalnost
- Sledi Rust best practices in idiomom

## 4. Arhitekturne smernice
- Vsak crate mora imeti točno določeno odgovornost
- Moduli morajo biti ohlapno povezani (loose coupling)
- Uporabljaj trait-e za abstrakcijo
- Implementiraj proper error types za vsak modul
- Sledi fail-fast principu
- Zagotovi visoko observability

## 5. Upravljanje z odvisnostmi
- Vse verzije odvisnosti definiraj v workspace root `Cargo.toml`
- Uporabljaj točno določene verzije (ne `*` ali `^`)
- Redno posodabljaj odvisnosti za varnostne popravke
- Miniminiziraj število odvisnosti v kritičnih poteh

## 6. Testiranje
- Piši unit teste za vse javne funkcije. Pokrij logiko v vsakem lib.rs in podmodulih. Vsaka funkcija mora imeti testne primere za pričakovane rezultate in robne pogoje.
- Implementiraj integracijske teste za kritične poti. Ustvari tests/ mapo z realnimi interakcijami med core, strategies, blockchain, wallet in network za validacijo sistemskega obnašanja.
- Dodaj performance teste za latency-sensitive kodo. Benchmarki (benches/) naj merijo core::executor, mempool, engine in strategije. Uporabi criterion z metriko <1ms.
- Uporabljaj property-based teste za preverjanje pravilnosti in robustnosti. S proptest pokrij algebraične lastnosti in robne pogoje v optimization, risk, oracle, strategy, transaction.
- Dodaj panic teste za nevarne funkcije. Uporabi #[should_panic] kjerkoli uporabljajo unwrap, expect, ali so druge točke napake.
- Piši doc teste. Vključi /// dokumentacijo z validnimi primeri uporabe, ki se samodejno testirajo.
- Uporabi fuzz teste za nepredvidljive inpute. Z cargo-fuzz in arbitrary generiraj neveljavne/robne inpute za transakcije, strategije in parserje.
- Dodaj stress teste za stabilnost. Simuliraj desetine tisoč zaporednih klicev kritičnih poti v ločenih niti. Testiraj memory leaks in CPU load.
- Izvedi security fuzzing za kritične točke. Fuzzaj RPC inpute, wallet key management, API endpoint-e in tx signing funkcionalnosti za robustnost in odpornost na napade.
- Vzpostavi regression test za vsako znano napako. Shrani minimalne primerke, ki povzročijo napake, kot stalni test primeri (npr. regression_test_issue_472.rs).
- Implementiraj end-to-end teste na testnetih. Validiraj celoten MEV flow od mempool zaznave do izvršene transakcije na realnem testnetu (npr. Goerli, Sepolia).
- Testiraj vse Result, Option, match in if let poti. Vsaka veja naj bo eksplicitno testirana, vključno z Err, None, fallback, timeout, cancellation scenariji.
- Uporabljaj test fixtures in mocks za izolacijo. Mockaj zunanje servise (npr. RPC, WebSocket, DEX) za deterministične in ponovljive teste. Izogibaj se testom, ki zahtevajo dostop do produkcijskega interneta.

## 7. Optimizacije
- Profiliraj kodo redno
- Optimiziraj hot paths
- Uporabljaj učinkovite podatkovne strukture (npr. DashMap za concurrent maps)
- Implementiraj caching kjer je smiselno
- Optimiziraj SQL poizvedbe

## 8. Monitoring in Logging
- Implementiraj proper metrics za vse kritične operacije
- Uporabljaj strukturirano logiranje
- Nastavi alerting za latency spikes
- Implementiraj distributed tracing
- Zagotovi audit logging za pomembne operacije

## 9. Error Handling
- Definiraj proper error types za vsak modul
- Implementiraj error recovery strategije
- Logiraj vse error-je z ustreznim kontekstom
- Zagotovi graceful degradation v primeru napak
- Implementiraj circuit breaker-je kjer je potrebno

## 10. Dokumentacija
- Dokumentiraj vse javne API-je
- Vzdržuj README.md za vsak crate
- Dokumentiraj arhitekturne odločitve
- Piši clear commit messages
- Vzdržuj changelog

## 11. Blockchain Specifike
- Implementiraj proper retry mehanizme za RPC klice
- Validiraj vse blockchain podatke
- Implementiraj proper gas management
- Zagotovi atomičnost transakcij
- Implementiraj proper mempool monitoring

## 12. Continuous Integration
- Vse spremembe morajo iti skozi CI pipeline
- Zagotovi 100% test coverage za kritične poti
- Implementiraj automated performance testing
- Dodaj security scanning
- Izvajaj regular dependency audits

## 13. Konfiguracija za avtomatsko preverjanje

### a) Rustfmt konfiguracija
Ustvari `rustfmt.toml` v root direktoriju:
```toml
edition = "2024"
max_width = 100
use_small_heuristics = "Max"
```
- Vklopi "Format on Save" v editorju
- Vedno formatiraj pred commitom

### b) Clippy konfiguracija
V vsakem `lib.rs` ali `main.rs` dodaj:
```rust
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]  // po potrebi
```

### c) Continuous Development Flow
Namesti cargo-watch:
```bash
cargo install cargo-watch
```

Med razvijanjem poganjaj:
```bash
cargo watch -x "fmt" -x "clippy" -x "test"
```

## 14. Razširjene testne zahteve

### a) Unit testi
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ime_funkcije() {
        // Arrange
        // Act
        // Assert
    }
}
```

### b) Integracijski testi
```rust
// tests/integration_test.rs
use tallyio_core::*;

#[test]
fn test_end_to_end_flow() {
    // Setup
    // Execute
    // Verify
}
```

### c) Property-based testing
```toml
[dev-dependencies]
proptest = "1.0"
```

```rust
proptest! {
    #[test]
    fn test_transaction_invariants(
        amount in 0..u64::MAX,
        gas_price in 1..1000u64
    ) {
        // Test properties that should always hold
    }
}
```

## 15. CI/CD Pipeline

### a) GitHub Actions konfiguracija
```yaml
name: Rust CI

on:
  push:
  pull_request:

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Rust
      uses: actions/setup-rust@v1
      with:
        rust-version: stable
    - name: Check formatting
      run: cargo fmt --all -- --check
    - name: Run Clippy
      run: cargo clippy --all-targets --all-features -- -D warnings
    - name: Run tests
      run: cargo test --all
```

### b) Code Coverage
```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```
- Zahtevaj minimalno 100% coverage.

## 16. Extra varnostne zahteve

### a) Fuzz Testing
- Implementiraj fuzz teste za parsing in network code
- Uporabljaj cargo-fuzz za kritične komponente

### b) Security Scanning
- Redno poganjaj `cargo audit`
- Implementiraj SAST (Static Application Security Testing)
- Skeniraj dependencies za znane ranljivosti

### c) Performance Testing
- Implementiraj benchmark teste
- Uporabljaj criterion.rs za performance teste
- Nastavi performance budgets
- Avtomatsko zaznavanje performance regresij

## 17. Development Workflow

### a) Pre-commit hooks
- Nastavi git hooks za:
  - Formatiranje kode
  - Clippy preverjanje
  - Unit teste
  - Cargo audit

### b) Code Review Process
- Zahtevaj code review za vse spremembe
- Uporabljaj PR template
- Preveri test coverage pred merge-om
- Zahtevaj uspešen CI pipeline

### c) Documentation
- Zahtevaj dokumentacijo za vse nove feature-e
- Avtomatsko preveri dokumentacijo z rustdoc
- Vzdržuj changelog
- Dokumentiraj breaking changes
