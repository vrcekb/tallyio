# TallyIO MEV Testno Ogrodje

Celovito testno ogrodje za TallyIO MEV in likvidacijsko platformo. Ta testna struktura je zasnovana za zagotavljanje sub-milisekundne latence, visoke zanesljivosti in robustnosti, kar je ključno za konkurenčno prednost v MEV okolju.

## Struktura in organizacija

Testno ogrodje je organizirano v logične module, ki pokrivajo različne vidike testiranja:

```
E:\alpha\Tallyio\tests\
├── unit\                    # Enotski testi posameznih komponent
│   ├── secure_storage_basic_tests.rs       # Osnovni testi secure_storage
│   ├── secure_storage_edge_cases.rs        # Robni primeri secure_storage
│   ├── blockchain_tests.rs                 # Enotski testi blockchain modula
│   └── core_metrics_tests.rs               # Testi za metrike performanc
├── integration\             # Integracijski testi med komponentami
│   ├── common.rs                           # Skupne komponente za teste
│   ├── mev_storage_integration_test.rs     # Integracija MEV in storage
│   └── core_components_integration.rs      # Interakcija med core komponentami
├── benchmarks\              # Performančni testi in benchmarki
│   ├── core_components_benchmark.rs        # Benchmark core komponent
│   ├── secure_storage_benchmark.rs         # Performančni testi shranjevanja
│   └── blockchain_benchmarks.rs            # Latenca blockchain operacij
├── property\                # Property-based testing in fuzz testi
│   └── secure_storage_property_tests.rs    # Property testi za storage
├── stress\                  # Stresni testi za obremenitev sistema
│   └── secure_storage_stress_tests.rs      # Stresni testi za storage
├── regression\              # Regresijski testi za odkrivanje regresij
│   └── secure_storage_regression.rs        # Regresijski testi za storage
├── fuzz\                    # Fuzz testi za validacijo robustnosti
│   ├── Cargo.toml                          # Konfiguracija fuzz testov
│   └── fuzz_targets\                       # Cilji za fuzz testiranje
│       ├── queue_operations.rs             # Fuzz testi za Queue operacije
│       ├── crypto_operations.rs            # Testiranje kriptografskih operacij
│       └── mev_data_validation.rs          # Validacija MEV podatkov
└── utils\                   # Pomožne funkcije za testiranje
    ├── coverage_reporting.rs               # Poročanje o pokritosti kode
    ├── mev_testing.rs                      # Specializirani MEV testi
    ├── performance_testing.rs              # Orodja za testiranje performanc
    ├── security_testing.rs                 # Varnostno testiranje
    ├── test_config.rs                      # Konfiguracija testnega okolja
    └── test_framework.rs                   # Osnovno testno ogrodje
```

## Tipi testov

### Enotski testi (Unit)

Testi posameznih komponent, ki preverjajo pravilnost implementacije izoliranih funkcionalnosti.

```bash
# Zagon vseh enotskih testov
cargo test --package tallyio-tests --test "*_tests"

# Zagon specifičnih enotskih testov
cargo test --package tallyio-tests --test "secure_storage_basic_tests"
```

### Integracijski testi

Preverjajo interakcijo med različnimi komponentami sistema in zagotavljajo celovitost delovanja.

```bash
# Zagon integracijskih testov
cargo test --package tallyio-tests --test "*_integration*" 
```

### Benchmarki

Merijo performančne karakteristike kritičnih MEV komponent, s poudarkom na latenci.

```bash
# Zagon vseh benchmark testov
cargo bench --package tallyio-tests

# Zagon specifičnega benchmarka
cargo bench --package tallyio-tests --bench core_components_benchmark
```

### Property testi

Generirajo naključne vhodne podatke za testiranje lastnosti sistema, kar je posebej pomembno za odkrivanje robnih primerov in nepričakovanih vhodov.

```bash
# Zagon property testov
cargo test --package tallyio-tests --features property-tests
```

### Stresni testi

Simulirajo visoko obremenitev in preverjajo robustnost sistema pod ekstremnimi pogoji.

```bash
# Zagon stresnih testov (dolgo izvajanje)
cargo test --package tallyio-tests --features stress-tests -- --ignored
```

### Fuzz testi

Sistematično generirajo nepričakovane vhodne podatke za odkrivanje ranljivosti.

```bash
# Nastavitev in zagon fuzz testov
cd tests/fuzz
cargo fuzz run queue_operations
```

## Zahteve za latenco

MEV aplikacije zahtevajo izjemno nizke latence za konkurenčno prednost. Naši testi preverjajo, da kritične poti izpolnjujejo naslednje zahteve:

- **Core operacije**: < 100μs (Arena alokacije, Queue operacije)
- **Secure Storage**: < 500μs za CRUD operacije
- **Blockchain interakcije**: < 1ms za branje stanja, < 2ms za oddajo transakcij
- **End-to-end MEV scenariji**: < 2ms

Latence so preverjene s specializiranimi benchmark testi:

```bash
# Zagon testov latence za kritične poti
cargo test --package tallyio-tests --features latency-tests
```

## Pokritost kode

Sistem vključuje napredno sledenje pokritosti kode, ki deluje tudi za asinhrone procese in kompleksne kontrolne tokove:

```bash
# Generiranje poročila o pokritosti
cargo test --package tallyio-tests --features coverage

# Pregled poročila
open target/coverage/index.html
```

Za specifične module uporabite:

```rust
// V testni datoteki
use tallyio_tests::utils::coverage_reporting::TestCoverageCollector;

#[test]
fn test_something() {
    let _collector = TestCoverageCollector::new("moj_test")
        .print_on_drop(true)
        .generate_html(true, Some("target/coverage/moj_test"));
    
    // Testna koda...
}
```

## CI/CD integracija

Testno ogrodje je integrirano v GitHub Actions CI/CD pipeline, ki avtomatsko izvaja teste ob pull requestih in commitih v razvejitev.

```yaml
# Zagon celotnega CI pipeline lokalno
act -j build-and-test
```

## Prispevanje novih testov

Za dodajanje novih testov sledite tem smernicam:

1. **Izbira prave kategorije**: Umestite teste v ustrezno kategorijo glede na njihov namen
2. **Imenovanje**: Sledite konvenciji `<modul>_<funkcionalnost>_<tip_testa>.rs`
3. **Dokumentacija**: Vsak test naj vsebuje kratek opis funkcionalnosti in kako je povezan z MEV zahtevami
4. **Latenca**: Za performančno kritične komponente dodajte eksplicitno preverjanje latence
5. **Pokritost**: Uporabite `track_line!` makro za sledenje pokritosti kode

## Specializirani MEV testi

MEV operacije imajo posebne zahteve za testiranje:

- **Mempool monitoring**: Testi, ki preverjajo hitrost in zanesljivost monitoringa mempool-a
- **Arbitrage detection**: Validacija algoritmov za detekcijo arbitražnih priložnosti
- **Sandwich attack simulation**: Testi za simulacijo in detekcijo sandwich napadov
- **Gas optimization**: Testi za validacijo optimizacij gas porabe

```bash
# Zagon specializiranih MEV testov
cargo test --package tallyio-tests --test "*mev*"
```
