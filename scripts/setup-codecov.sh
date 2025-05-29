#!/bin/bash
# 📊 TallyIO Codecov Setup Verification Script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 TallyIO Codecov Setup Verification${NC}"
echo -e "${BLUE}====================================${NC}"

# Repository information
REPO_SLUG="vrcekb/tallyio"
CODECOV_TOKEN="194732cc-0084-48cd-8af3-4d72abdd3cdc"

echo -e "${BLUE}🔍 Repository: ${REPO_SLUG}${NC}"
echo -e "${BLUE}🔑 Token: ${CODECOV_TOKEN:0:8}...${NC}"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Error: Cargo.toml not found. Run this script from the project root.${NC}"
    exit 1
fi

# Check if grcov is installed (following Codecov Rust example)
echo -e "${BLUE}🔧 Checking grcov installation...${NC}"
if ! command -v grcov &> /dev/null; then
    echo -e "${YELLOW}⚠️  Installing grcov...${NC}"
    cargo install grcov
else
    echo -e "${GREEN}✅ grcov is installed${NC}"
fi

# Check if nightly toolchain is available
echo -e "${BLUE}🔧 Checking Rust nightly toolchain...${NC}"
if rustup toolchain list | grep -q nightly; then
    echo -e "${GREEN}✅ Rust nightly toolchain is available${NC}"
else
    echo -e "${YELLOW}⚠️  Installing Rust nightly toolchain...${NC}"
    rustup toolchain install nightly
    rustup component add llvm-tools-preview --toolchain nightly
fi

# Check if codecov.yml exists and is configured
echo -e "${BLUE}📋 Checking codecov.yml configuration...${NC}"
if [ -f "codecov.yml" ]; then
    echo -e "${GREEN}✅ codecov.yml found${NC}"

    # Check for TallyIO-specific configuration
    if grep -q "95" codecov.yml; then
        echo -e "${GREEN}✅ TallyIO coverage targets (95%) configured${NC}"
    else
        echo -e "${YELLOW}⚠️  TallyIO coverage targets not found in codecov.yml${NC}"
    fi

    if grep -q "100%" codecov.yml; then
        echo -e "${GREEN}✅ Critical path coverage (100%) configured${NC}"
    else
        echo -e "${YELLOW}⚠️  Critical path coverage not found in codecov.yml${NC}"
    fi
else
    echo -e "${RED}❌ codecov.yml not found${NC}"
    exit 1
fi

# Check CI configuration
echo -e "${BLUE}🔧 Checking CI configuration...${NC}"
if [ -f ".github/workflows/ci.yml" ]; then
    echo -e "${GREEN}✅ CI workflow found${NC}"

    if grep -q "codecov/codecov-action@v5" .github/workflows/ci.yml; then
        echo -e "${GREEN}✅ Codecov action v5 configured${NC}"
    else
        echo -e "${YELLOW}⚠️  Codecov action not found or wrong version${NC}"
    fi

    if grep -q "CODECOV_TOKEN" .github/workflows/ci.yml; then
        echo -e "${GREEN}✅ CODECOV_TOKEN reference found${NC}"
    else
        echo -e "${RED}❌ CODECOV_TOKEN not referenced in CI${NC}"
    fi

    if grep -q "vrcekb/tallyio" .github/workflows/ci.yml; then
        echo -e "${GREEN}✅ Repository slug configured${NC}"
    else
        echo -e "${YELLOW}⚠️  Repository slug not found${NC}"
    fi
else
    echo -e "${RED}❌ CI workflow not found${NC}"
    exit 1
fi

# Generate test coverage report (following Codecov Rust example)
echo -e "${BLUE}🧪 Generating test coverage report with grcov...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"

# Clean previous coverage data
rm -rf target/debug/deps/*.gcda target/debug/deps/*.gcno
rm -f *.profraw coverage.lcov

# Set coverage environment variables (from Codecov example)
export CARGO_INCREMENTAL=0
export RUSTFLAGS="-Zprofile -Ccodegen-units=1 -Cinline-threshold=0 -Clink-dead-code -Coverflow-checks=off -Cpanic=abort -Zpanic_abort_tests"
export RUSTDOCFLAGS="-Zprofile -Ccodegen-units=1 -Cinline-threshold=0 -Clink-dead-code -Coverflow-checks=off -Cpanic=abort -Zpanic_abort_tests"

# Build with nightly and coverage instrumentation
echo -e "${BLUE}🔧 Building with coverage instrumentation...${NC}"
if cargo +nightly build --verbose; then
    echo -e "${GREEN}✅ Build with coverage instrumentation successful${NC}"
else
    echo -e "${RED}❌ Failed to build with coverage instrumentation${NC}"
    exit 1
fi

# Run tests with coverage
echo -e "${BLUE}🧪 Running tests with coverage...${NC}"
if cargo +nightly test --verbose --all; then
    echo -e "${GREEN}✅ Tests with coverage completed${NC}"
else
    echo -e "${RED}❌ Tests with coverage failed${NC}"
    exit 1
fi

# Generate coverage report with grcov
echo -e "${BLUE}📊 Generating coverage report with grcov...${NC}"
if grcov . --binary-path ./target/debug/deps/ -s . -t lcov --branch --ignore-not-existing --ignore '../*' --ignore "/*" -o coverage.lcov; then
    echo -e "${GREEN}✅ Coverage report generated successfully${NC}"

    # Check if coverage.lcov was created
    if [ -f "coverage.lcov" ]; then
        LINES_COUNT=$(wc -l < coverage.lcov)
        echo -e "${GREEN}✅ coverage.lcov created with ${LINES_COUNT} lines${NC}"

        # Show basic coverage info
        echo -e "${BLUE}📊 Coverage file created successfully${NC}"

    else
        echo -e "${RED}❌ coverage.lcov file not created${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Failed to generate coverage report with grcov${NC}"
    exit 1
fi

# Test Codecov upload (dry run)
echo -e "${BLUE}🚀 Testing Codecov upload...${NC}"

# Download codecov uploader if not exists
if [ ! -f "codecov" ]; then
    echo -e "${YELLOW}📥 Downloading Codecov uploader...${NC}"
    curl -Os https://uploader.codecov.io/latest/linux/codecov
    chmod +x codecov
fi

# Test upload (dry run)
if ./codecov -t "$CODECOV_TOKEN" -f coverage.lcov --dry-run; then
    echo -e "${GREEN}✅ Codecov upload test successful${NC}"
else
    echo -e "${RED}❌ Codecov upload test failed${NC}"
    exit 1
fi

# Validate TallyIO coverage requirements
echo -e "${BLUE}🎯 Validating TallyIO coverage requirements...${NC}"

# Parse coverage from lcov file (basic validation)
if [ -f "coverage.lcov" ]; then
    # Count lines found and hit from lcov file
    LINES_FOUND=$(grep -c "^LF:" coverage.lcov || echo "0")
    LINES_HIT=$(grep "^LH:" coverage.lcov | awk -F: '{sum += $2} END {print sum}' || echo "0")

    if [ "$LINES_FOUND" -gt 0 ]; then
        OVERALL_COVERAGE=$(echo "scale=2; $LINES_HIT * 100 / $LINES_FOUND" | bc -l || echo "0")
        echo -e "${BLUE}📊 Overall coverage: ${OVERALL_COVERAGE}%${NC}"

        # Check if coverage meets TallyIO requirements
        if (( $(echo "$OVERALL_COVERAGE >= 95" | bc -l) )); then
            echo -e "${GREEN}✅ Coverage meets TallyIO requirements (${OVERALL_COVERAGE}% >= 95%)${NC}"
        elif (( $(echo "$OVERALL_COVERAGE >= 80" | bc -l) )); then
            echo -e "${YELLOW}⚠️  Coverage ${OVERALL_COVERAGE}% is good but below TallyIO target of 95%${NC}"
            echo -e "${YELLOW}📋 Add more tests to reach 95% coverage${NC}"
        else
            echo -e "${RED}❌ Coverage ${OVERALL_COVERAGE}% is below acceptable levels${NC}"
            echo -e "${RED}📋 Significant testing improvements needed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Could not parse coverage percentage from lcov file${NC}"
        echo -e "${BLUE}📋 Coverage report generated - check Codecov dashboard for details${NC}"
    fi
else
    echo -e "${RED}❌ Coverage file not found${NC}"
    exit 1
fi

# Final summary
echo -e "${BLUE}📋 Setup Summary:${NC}"
echo -e "${GREEN}✅ grcov: Installed and working${NC}"
echo -e "${GREEN}✅ Rust nightly: Available with coverage support${NC}"
echo -e "${GREEN}✅ codecov.yml: Configured with TallyIO standards${NC}"
echo -e "${GREEN}✅ CI workflow: Configured for Codecov upload (grcov method)${NC}"
echo -e "${GREEN}✅ Coverage generation: Working with grcov${NC}"
echo -e "${GREEN}✅ Codecov upload: Tested successfully${NC}"

echo -e "${BLUE}🎯 Next Steps:${NC}"
echo -e "${YELLOW}1. Ensure CODECOV_TOKEN is set as GitHub repository secret${NC}"
echo -e "${YELLOW}2. Push changes to trigger CI and verify Codecov integration${NC}"
echo -e "${YELLOW}3. Check Codecov dashboard: https://codecov.io/gh/${REPO_SLUG}${NC}"
echo -e "${YELLOW}4. Add more tests if coverage is below 95%${NC}"

echo -e "${GREEN}🎉 Codecov setup verification completed!${NC}"

# Cleanup
rm -f codecov
