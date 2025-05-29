#!/bin/bash
# 🚀 TallyIO Local CI/CD Pipeline
# Runs the same checks as GitHub Actions locally

set -e  # Exit on any error

# 🎯 Configuration
SKIP_COVERAGE=false
SKIP_DOCKER=false
SKIP_BENCHMARKS=false
FAST_MODE=false

# 🎨 Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# 📋 Results tracking
declare -A results=(
    ["formatting"]=false
    ["clippy"]=false
    ["zero_panic"]=false
    ["build"]=false
    ["unit_tests"]=false
    ["integration_tests"]=false
    ["performance_tests"]=false
    ["security"]=false
    ["coverage"]=false
    ["docker"]=false
    ["benchmarks"]=false
)

function show_help() {
    cat << EOF
🚀 TallyIO Local CI/CD Pipeline

Usage: ./scripts/local-ci.sh [OPTIONS]

Options:
  --skip-coverage     Skip code coverage analysis
  --skip-docker       Skip Docker build test
  --skip-benchmarks   Skip performance benchmarks
  --fast              Run only essential checks (fmt, clippy, tests)
  --help              Show this help message

Examples:
  ./scripts/local-ci.sh                    # Full CI pipeline
  ./scripts/local-ci.sh --fast             # Quick checks only
  ./scripts/local-ci.sh --skip-coverage    # Skip coverage
EOF
}

function log_step() {
    echo -e "${YELLOW}🔄 $1${NC}"
}

function log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function log_error() {
    echo -e "${RED}❌ $1${NC}"
}

function log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function log_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

function check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

# 📝 Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-coverage)
            SKIP_COVERAGE=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --skip-benchmarks)
            SKIP_BENCHMARKS=true
            shift
            ;;
        --fast)
            FAST_MODE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# 🎯 TallyIO Performance Requirements
export CARGO_TERM_COLOR=always
export RUST_BACKTRACE=1
export TALLYIO_MAX_LATENCY_MS=1
export TALLYIO_ZERO_PANIC=true

START_TIME=$(date +%s)

echo -e "${CYAN}🚀 TallyIO Local CI/CD Pipeline Starting...${NC}"
echo -e "${GRAY}📅 Started at: $(date)${NC}"
echo ""

# 🔍 Prerequisites check
log_step "Checking prerequisites..."

if ! check_command cargo; then
    log_error "Cargo not found. Please install Rust toolchain."
    exit 1
fi

if ! check_command rustfmt; then
    log_warning "rustfmt not found. Installing..."
    rustup component add rustfmt
fi

if ! cargo clippy --version &> /dev/null; then
    log_warning "clippy not found. Installing..."
    rustup component add clippy
fi

log_success "Prerequisites OK"
echo ""

# 🎨 1. Code Formatting Check
log_step "Checking code formatting..."
if cargo fmt --all -- --check; then
    results["formatting"]=true
    log_success "Code formatting: PASSED"
else
    log_error "Code formatting: FAILED"
    echo -e "${GRAY}Run 'cargo fmt --all' to fix formatting issues.${NC}"
fi
echo ""

# 📎 2. Clippy Linting (Zero warnings policy)
log_step "Running Clippy (Zero warnings policy)..."
if cargo clippy --all-targets --all-features -- -D warnings; then
    results["clippy"]=true
    log_success "Clippy: PASSED"
else
    log_error "Clippy: FAILED"
    echo -e "${GRAY}Fix all clippy warnings before proceeding.${NC}"
fi
echo ""

# 🚨 3. TallyIO Zero Panic Check
log_step "TallyIO Zero Panic Policy Check..."
PANIC_COUNT=$(grep -r "unwrap\|expect\|panic!\|todo!\|unimplemented!" crates/ 2>/dev/null | wc -l || echo "0")
if [ "$PANIC_COUNT" -gt 0 ]; then
    log_error "Found $PANIC_COUNT prohibited patterns:"
    grep -rn "unwrap\|expect\|panic!\|todo!\|unimplemented!" crates/ 2>/dev/null || true
    results["zero_panic"]=false
else
    results["zero_panic"]=true
    log_success "Zero panic policy: PASSED"
fi
echo ""

if [ "$FAST_MODE" = true ]; then
    log_info "Fast mode enabled - skipping remaining checks"
    echo ""
    # Jump to summary
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo -e "${CYAN}📋 TallyIO Local CI/CD Summary${NC}"
    echo -e "${CYAN}================================${NC}"
    echo -e "${GRAY}⏱️  Duration: ${DURATION}s${NC}"
    echo ""
    
    PASSED=0
    TOTAL=0
    
    for check in "${!results[@]}"; do
        TOTAL=$((TOTAL + 1))
        if [ "${results[$check]}" = true ]; then
            PASSED=$((PASSED + 1))
            echo -e "${GREEN}✅ $check: PASSED${NC}"
        else
            echo -e "${RED}❌ $check: FAILED${NC}"
        fi
    done
    
    echo ""
    if [ $PASSED -eq $TOTAL ]; then
        echo -e "${GREEN}📊 Results: $PASSED/$TOTAL checks passed${NC}"
        echo -e "${GREEN}🎉 All checks passed! Ready for GitHub push.${NC}"
        exit 0
    else
        echo -e "${RED}📊 Results: $PASSED/$TOTAL checks passed${NC}"
        echo -e "${RED}💥 Some checks failed. Fix issues before pushing.${NC}"
        exit 1
    fi
fi

# 🔧 4. Build All Crates
log_step "Building all crates..."
if cargo build --all --verbose; then
    results["build"]=true
    log_success "Build: PASSED"
else
    log_error "Build: FAILED"
fi
echo ""

# 🧪 5. Unit Tests
log_step "Running unit tests..."
if cargo test --all --lib --verbose; then
    results["unit_tests"]=true
    log_success "Unit tests: PASSED"
else
    log_error "Unit tests: FAILED"
fi
echo ""

# 🔗 6. Integration Tests
log_step "Running integration tests..."
if cargo test --all --test '*' --verbose; then
    results["integration_tests"]=true
    log_success "Integration tests: PASSED"
else
    log_error "Integration tests: FAILED"
fi
echo ""

# ⚡ 7. Performance & Latency Tests
log_step "Running TallyIO performance tests..."
if cargo test --all --release test_latency_requirement -- --nocapture && \
   cargo test --all --release benchmark -- --nocapture; then
    results["performance_tests"]=true
    log_success "Performance tests: PASSED"
else
    log_error "Performance tests: FAILED"
fi
echo ""

# 🔒 8. Security Audit
log_step "Running security audit..."
if ! check_command cargo-audit; then
    log_warning "cargo-audit not found. Installing..."
    cargo install cargo-audit
fi

if cargo audit && cargo audit --deny warnings; then
    results["security"]=true
    log_success "Security audit: PASSED"
else
    log_error "Security audit: FAILED"
fi
echo ""

# 📊 9. Code Coverage (Optional)
if [ "$SKIP_COVERAGE" = false ]; then
    log_step "Generating code coverage report..."
    if ! check_command cargo-llvm-cov; then
        log_warning "cargo-llvm-cov not found. Installing..."
        cargo install cargo-llvm-cov
    fi
    
    if cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info; then
        results["coverage"]=true
        log_success "Code coverage: PASSED"
        echo -e "${GRAY}📊 Coverage report saved to: lcov.info${NC}"
    else
        log_error "Code coverage: FAILED"
    fi
    echo ""
fi

# 🐳 10. Docker Build (Optional)
if [ "$SKIP_DOCKER" = false ]; then
    log_step "Testing Docker build..."
    if ! check_command docker; then
        log_warning "Docker not found. Skipping Docker build test."
    else
        if docker build -t tallyio:local-test .; then
            results["docker"]=true
            log_success "Docker build: PASSED"
        else
            log_error "Docker build: FAILED"
        fi
    fi
    echo ""
fi

# 📈 11. Benchmarks (Optional)
if [ "$SKIP_BENCHMARKS" = false ]; then
    log_step "Running performance benchmarks..."
    if cargo bench --all; then
        results["benchmarks"]=true
        log_success "Benchmarks: PASSED"
    else
        log_error "Benchmarks: FAILED"
    fi
    echo ""
fi

# 📋 Summary Report
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "${CYAN}📋 TallyIO Local CI/CD Summary${NC}"
echo -e "${CYAN}================================${NC}"
echo -e "${GRAY}⏱️  Duration: ${DURATION}s${NC}"
echo ""

PASSED=0
TOTAL=0

for check in "${!results[@]}"; do
    TOTAL=$((TOTAL + 1))
    if [ "${results[$check]}" = true ]; then
        PASSED=$((PASSED + 1))
        echo -e "${GREEN}✅ $check: PASSED${NC}"
    else
        echo -e "${RED}❌ $check: FAILED${NC}"
    fi
done

echo ""
if [ $PASSED -eq $TOTAL ]; then
    echo -e "${GREEN}📊 Results: $PASSED/$TOTAL checks passed${NC}"
    echo -e "${GREEN}🎉 All checks passed! Ready for GitHub push.${NC}"
    exit 0
else
    echo -e "${RED}📊 Results: $PASSED/$TOTAL checks passed${NC}"
    echo -e "${RED}💥 Some checks failed. Fix issues before pushing.${NC}"
    exit 1
fi
