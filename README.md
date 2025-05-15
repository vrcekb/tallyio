# TallyIO

[![Rust CI](https://github.com/vrcekb/tallyio/actions/workflows/rust.yml/badge.svg)](https://github.com/vrcekb/tallyio/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library for handling calculations with proper error handling and testing.

## Features

- Basic arithmetic operations
- Comprehensive error handling
- Unit and integration tests
- GitHub Actions CI/CD pipeline
- Code coverage with tarpaulin
- Documentation with examples

## Getting Started

### Prerequisites

- Rust (latest stable version recommended)
- Cargo (Rust's package manager)

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tallyio = { git = "https://github.com/vrcekb/tallyio.git" }
```

### Usage

```rust
use tallyio::{sestej, preveri_pozitivno, preberi_datoteko};

fn main() -> anyhow::Result<()> {
    // Basic arithmetic
    let vsota = sestej(5, 3);
    println!("5 + 3 = {}", vsota);
    
    // Validation
    preveri_pozitivno(42)?; // Returns Ok(())
    // preveri_pozitivno(-1)?; // Returns Err(TallyError::Validation(...))
    
    // File operations
    let vsebina = preberi_datoteko("nekaj.txt")?;
    println!("Prebrano: {}", vsebina);
    
    Ok(())
}
```

## Development

### Building

```bash
cargo build
```

### Testing

Run all tests:
```bash
cargo test
```

Run with detailed output:
```bash
cargo test -- --nocapture
```

### Code Formatting

```bash
cargo fmt
```

### Linting

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Documentation

Generate and open documentation:
```bash
cargo doc --open
```

## CI/CD

This project uses GitHub Actions for continuous integration. The following checks are performed on every push and pull request:

- Format checking with `rustfmt`
- Linting with `clippy`
- Running tests
- Generating documentation
- Checking for unused dependencies
- Security vulnerability scanning
- Test coverage with tarpaulin

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
