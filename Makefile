# 🚀 TallyIO Development Makefile
# Cross-platform commands for development workflow

.PHONY: help check quick ci fmt clippy test build clean install-tools

# 🎯 Default target
help:
	@echo "🚀 TallyIO Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Quick Commands:"
	@echo "  make quick     - Ultra-fast pre-commit checks (~30s)"
	@echo "  make check     - Full local CI pipeline (~5min)"
	@echo "  make fmt       - Format all code"
	@echo "  make clippy    - Run clippy linter"
	@echo "  make test      - Run all tests"
	@echo "  make build     - Build all crates"
	@echo ""
	@echo "Setup:"
	@echo "  make install-tools - Install required tools"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Platform-specific:"
	@echo "  Windows: Uses PowerShell scripts"
	@echo "  Unix:    Uses Bash scripts"

# 🏃 Ultra-fast pre-commit checks
quick:
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File scripts/quick-check.ps1
else
	@./scripts/quick-check.sh
endif

# 🔍 Full local CI pipeline
check:
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File scripts/local-ci.ps1
else
	@./scripts/local-ci.sh
endif

# 🔍 Fast CI (essential checks only)
ci-fast:
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File scripts/local-ci.ps1 -Fast
else
	@./scripts/local-ci.sh --fast
endif

# 🎨 Format code
fmt:
	@echo "🎨 Formatting code..."
	@cargo fmt --all

# 📎 Run clippy
clippy:
	@echo "📎 Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

# 🧪 Run tests
test:
	@echo "🧪 Running tests..."
	@cargo test --all --verbose

# 🔧 Build all crates
build:
	@echo "🔧 Building all crates..."
	@cargo build --all --verbose

# 🔒 Security audit
audit:
	@echo "🔒 Running security audit..."
	@cargo audit

# 📊 Generate coverage report
coverage:
	@echo "📊 Generating coverage report..."
	@cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@echo "Coverage report saved to: lcov.info"

# 🐳 Build Docker image
docker:
	@echo "🐳 Building Docker image..."
	@docker build -t tallyio:dev .

# 📈 Run benchmarks
bench:
	@echo "📈 Running benchmarks..."
	@cargo bench --all

# 🧹 Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean
	@rm -f lcov.info
	@rm -f benchmark_results.json

# 🔧 Install required tools
install-tools:
	@echo "🔧 Installing required tools..."
	@rustup component add rustfmt clippy
	@cargo install cargo-audit cargo-llvm-cov

# 🚀 Pre-commit workflow
pre-commit: fmt quick
	@echo "🚀 Pre-commit checks completed!"

# 🚀 Pre-push workflow  
pre-push: check
	@echo "🚀 Pre-push checks completed!"

# 🔄 Update dependencies
update:
	@echo "🔄 Updating dependencies..."
	@cargo update

# 📋 Show project status
status:
	@echo "📋 TallyIO Project Status"
	@echo "========================"
	@echo "Rust version: $$(rustc --version)"
	@echo "Cargo version: $$(cargo --version)"
	@echo "Project crates:"
	@find crates -name "Cargo.toml" -exec echo "  - {}" \;
	@echo ""
	@echo "Git status:"
	@git status --porcelain

# 🎯 TallyIO specific checks
tallyio-check:
	@echo "🎯 TallyIO specific checks..."
	@echo "Checking for prohibited patterns..."
	@! grep -r "unwrap\|expect\|panic!\|todo!\|unimplemented!" crates/ || (echo "❌ Found prohibited patterns!" && exit 1)
	@echo "✅ Zero panic policy verified"
	@echo "Checking latency requirements..."
	@cargo test --release test_latency_requirement
	@echo "✅ Latency requirements verified"
