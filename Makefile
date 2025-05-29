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

# 📎 Run ultra-strict clippy (TallyIO standards)
clippy:
	@echo "📎 Running ultra-strict clippy (TallyIO standards)..."
	@cargo clippy --all-targets --all-features -- \
		-D warnings \
		-D clippy::all \
		-D clippy::pedantic \
		-D clippy::nursery \
		-D clippy::correctness \
		-D clippy::suspicious \
		-D clippy::perf \
		-D clippy::redundant_allocation \
		-D clippy::needless_collect \
		-D clippy::suboptimal_flops \
		-A clippy::missing_docs_in_private_items \
		-D clippy::infinite_loop \
		-D clippy::while_immutable_condition \
		-D clippy::never_loop \
		-D for_loops_over_fallibles \
		-D clippy::manual_strip \
		-D clippy::needless_continue \
		-D clippy::match_same_arms \
		-D clippy::unwrap_used \
		-D clippy::expect_used \
		-D clippy::panic \
		-D clippy::large_stack_arrays \
		-D clippy::large_enum_variant \
		-D clippy::mut_mut \
		-D clippy::cast_possible_truncation \
		-D clippy::cast_sign_loss \
		-D clippy::cast_precision_loss \
		-D clippy::must_use_candidate \
		-D clippy::empty_loop \
		-D clippy::if_same_then_else \
		-D clippy::await_holding_lock \
		-D clippy::await_holding_refcell_ref \
		-D clippy::let_underscore_future \
		-D clippy::diverging_sub_expression \
		-D clippy::unreachable \
		-D clippy::default_numeric_fallback \
		-D clippy::redundant_pattern_matching \
		-D clippy::manual_let_else \
		-D clippy::blocks_in_conditions \
		-D clippy::needless_pass_by_value \
		-D clippy::single_match_else \
		-D clippy::branches_sharing_code \
		-D clippy::useless_asref \
		-D clippy::redundant_closure_for_method_calls \
		-v

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
