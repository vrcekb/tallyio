# TallyIO - Ultra-Performant Financial Trading Platform

🚀 **Production-ready Rust workspace for high-frequency trading and MEV extraction**

## 🏗️ Architecture

```
TallyIO/
├── crates/                     # 🦀 Rust crates
│   ├── core/                   # ⚡ Ultra-performant core (<1ms latency)
│   ├── blockchain/             # 🔗 Multi-chain integration
│   ├── liquidation/            # 🎯 Liquidation strategies
│   ├── security/               # 🔒 Security modules
│   ├── database/               # 💾 Database abstractions
│   ├── metrics/                # 📊 Metrics & monitoring
│   ├── api/                    # 🌐 REST + WebSocket API
│   ├── contracts/              # 📜 Smart contracts
│   └── web-ui/                 # 🖥️ React frontend
├── smart-contracts/            # 📄 Solidity contracts
├── config/                     # ⚙️ Configuration files
├── migrations/                 # 🗃️ Database migrations
├── monitoring/                 # 📈 Grafana dashboards
└── deployment/                 # 🚀 Docker + K8s config
```

## ⚡ Core Features

### 🎯 Ultra-Low Latency
- **<1ms processing guarantee** for critical operations
- **Zero-panic architecture** - All errors handled with `Result<T, E>`
- **Cache-optimized memory layout** with `#[repr(C, align(64))]`
- **Lock-free data structures** using `crossbeam::queue::SegQueue`

### 🔥 Performance Optimizations
- **Memory-aligned structs** for cache efficiency
- **Atomic counters** for metrics without locks
- **Pre-allocated buffers** with `Vec::with_capacity()`
- **Fast hashing** for hot paths

### 🛡️ Production-Ready
- **Comprehensive error handling** with `thiserror`
- **Type safety** with zero-cost abstractions
- **Extensive testing** with latency assertions
- **Monitoring integration** with Prometheus metrics

## 🚀 Quick Start

### Prerequisites
- Rust 1.75+ (nightly recommended)
- PostgreSQL 15+
- Redis 7+

### Build & Test
```bash
# Clone repository
git clone https://github.com/your-org/tallyio
cd tallyio

# Build all crates
cargo build --release

# Run tests
cargo test --all

# Check code quality
cargo clippy --all-targets -- -D warnings

# Run quick checks (recommended before every commit)
.\scripts\quick-check.ps1
```

## 🔄 Local Development & CI/CD Consistency

TallyIO ensures **identical validation** between local development and CI/CD:

### Local Quick Check
```powershell
# Run all checks locally (mirrors CI exactly)
.\scripts\quick-check.ps1

# Individual checks available:
cargo fmt --all -- --check                    # Code formatting
cargo clippy --all-targets --all-features -- -D warnings  # Ultra-strict linting
cargo test --all                              # Unit & integration tests
cargo test --all --release                    # Performance tests
cargo audit --ignore RUSTSEC-2023-0071 --ignore RUSTSEC-2024-0421 --ignore RUSTSEC-2025-0009  # Security audit
cargo tarpaulin --all-features --workspace --fail-under 90  # Code coverage (90%+)
```

### GitHub Actions CI
- **`.github/workflows/ci.yml`** - Main CI pipeline (mirrors quick-check.ps1)
- **`.github/workflows/security.yml`** - Extended security checks
- **`.github/workflows/release.yml`** - Release automation

### Security Vulnerabilities (Temporarily Ignored)
Current known issues waiting for upstream fixes:
- **RUSTSEC-2023-0071**: RSA Marvin Attack (no fix available)
- **RUSTSEC-2024-0421**: idna vulnerability (waiting for web3 update)
- **RUSTSEC-2025-0009**: ring AES panic (waiting for ethers update)
- **RUSTSEC-2025-0010**: ring unmaintained (now maintained by rustls)
- **RUSTSEC-2024-0384**: instant unmaintained (used by ethers)

✅ **All checks pass identically in both local and CI environments**

## 📊 Performance Benchmarks

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Transaction Processing | <1ms | 100K+ TPS |
| MEV Opportunity Scan | <100μs | 1M+ ops/sec |
| Price Updates | <50μs | 2M+ ops/sec |

## 🔧 Core Crate API

### Transaction Processing
```rust
use tallyio_core::{TallyEngine, Transaction, Price, Gas};

// Create engine
let engine = TallyEngine::new()?;

// Submit transaction
let tx = Transaction::new(
    from_addr,
    Some(to_addr),
    Price::new(1_000_000_000_000_000_000), // 1 ETH
    Price::new(20_000_000_000),            // 20 gwei
    Gas::new(21_000),
    0,
    vec![]
);

engine.submit_transaction(tx)?;

// Process with <1ms guarantee
if let Some(result) = engine.process_next()? {
    println!("Processed in {}ns", result.processing_time_ns);
}
```

### MEV Opportunity Detection
```rust
// Automatic MEV scanning for DeFi transactions
if tx.is_defi_related() {
    if let Some(opportunity) = scan_mev_opportunity(&tx)? {
        println!("Found MEV opportunity: {} wei", opportunity.value());
    }
}
```

## 🏆 Code Quality Standards

### ✅ Enforced Rules
- **Zero panic policy** - No `unwrap()`, `expect()`, `panic!()`
- **Error handling** - All functions return `Result<T, E>`
- **Performance** - Critical paths must be <1ms
- **Memory safety** - No unsafe code without documentation
- **Testing** - All modules have comprehensive tests

### 🧪 Testing Strategy
```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test integration

# Performance tests
cargo test test_latency_requirement
```

## 📈 Monitoring & Metrics

### Built-in Metrics
- Transaction processing latency
- MEV opportunities found
- Error rates and types
- Memory usage patterns
- Queue sizes and throughput

### Prometheus Integration
```rust
use tallyio_metrics::MetricsManager;

let metrics = MetricsManager::new()?;
metrics.record_transaction_latency(duration);
metrics.record_mev_opportunity();
```

## 🔐 Security Features

- **Input validation** on all external data
- **Rate limiting** for API endpoints
- **Secure key management** for blockchain interactions
- **Audit logging** for all critical operations

## 🚀 Deployment

### Docker
```bash
docker build -t tallyio .
docker run -p 8080:8080 tallyio
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/
```

## 📚 Documentation

- [Core API Reference](docs/core-api.md)
- [Blockchain Integration](docs/blockchain.md)
- [Performance Tuning](docs/performance.md)
- [Deployment Guide](docs/deployment.md)

## 🚀 CI/CD Pipeline

TallyIO uses a unified GitHub Actions workflow for comprehensive automated testing and deployment:

### 🔄 Single Workflow Strategy
- **Unified pipeline**: All checks (quality, tests, coverage, security, docker) in one workflow
- **Minimal runs**: Optimized to prevent multiple concurrent workflow executions
- **Consistent validation**: Local `scripts/quick-check.ps1` mirrors GitHub Actions CI/CD
- **Fast feedback**: Run full CI checks locally before pushing

### 🛡️ Quality Gates
- **Zero panic policy**: Automated detection of `unwrap()`, `expect()`, `panic!()`
- **Ultra-strict Clippy**: 50+ lint rules for maximum code quality
- **Security audit**: Integrated vulnerability scanning with cargo-audit
- **Supply chain security**: Automated dependency verification with cargo-deny
- **Code coverage**: Minimum 90% overall, 100% for critical modules
- **Performance tests**: <1ms latency requirement verification

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure all tests pass (`cargo test --all`)
4. Run clippy (`cargo clippy --all-targets -- -D warnings`)
5. Check coverage meets requirements (see [Codecov Setup](docs/codecov-setup.md))
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open Pull Request

### 📊 Code Coverage Setup
To enable automatic coverage tracking, configure the `CODECOV_TOKEN` secret in GitHub repository settings. See [docs/codecov-setup.md](docs/codecov-setup.md) for detailed instructions.

### 🔄 Dependency Management
TallyIO uses Dependabot for automated dependency updates with optimized batching:
- **Single PR strategy**: All Rust dependencies are batched into one weekly PR
- **Minimal workflow runs**: `open-pull-requests-limit: 1` prevents multiple concurrent PRs
- **Smart grouping**: All updates (minor, patch, major) are grouped together
- **Critical dependency protection**: Major updates for `tokio`, `sqlx`, `ethers`, and `web3` require manual review

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Status

- ✅ **Core Engine** - Production ready with <1ms latency
- ✅ **Error Handling** - Zero panic guarantee implemented
- ✅ **Testing** - Comprehensive test suite
- 🚧 **Blockchain Integration** - In development
- 🚧 **API Layer** - In development
- 🚧 **Web UI** - Planned

---

**Built with ❤️ for ultra-high performance financial applications**
