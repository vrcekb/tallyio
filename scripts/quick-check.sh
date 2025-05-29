#!/bin/bash
# 🏃 TallyIO Quick Check
# Ultra-fast pre-commit checks (~30 seconds)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m'

echo -e "${CYAN}🏃 TallyIO Quick Check${NC}"
echo -e "${CYAN}=====================${NC}"
echo ""

START_TIME=$(date +%s)

# 🎨 1. Formatting
echo -e "${YELLOW}🎨 Checking formatting...${NC}"
if cargo fmt --all -- --check &> /dev/null; then
    echo -e "${GREEN}✅ Formatting: OK${NC}"
else
    echo -e "${RED}❌ Formatting: FAILED${NC}"
    echo -e "${GRAY}   Run: cargo fmt --all${NC}"
    exit 1
fi

# 📎 2. Clippy
echo -e "${YELLOW}📎 Running clippy...${NC}"
if cargo clippy --all-targets --all-features -- -D warnings &> /dev/null; then
    echo -e "${GREEN}✅ Clippy: OK${NC}"
else
    echo -e "${RED}❌ Clippy: FAILED${NC}"
    exit 1
fi

# 🚨 3. Zero Panic Check
echo -e "${YELLOW}🚨 Checking zero panic policy...${NC}"
PANIC_COUNT=$(grep -r "\.unwrap()\|\.expect(\|panic!\|todo!\|unimplemented!" crates/ 2>/dev/null | wc -l || echo "0")
if [ "$PANIC_COUNT" -gt 0 ]; then
    echo -e "${RED}❌ Zero panic: FAILED ($PANIC_COUNT violations)${NC}"
    grep -rn "\.unwrap()\|\.expect(\|panic!\|todo!\|unimplemented!" crates/ 2>/dev/null || true
    exit 1
else
    echo -e "${GREEN}✅ Zero panic: OK${NC}"
fi

# 🧪 4. Quick test
echo -e "${YELLOW}🧪 Running quick tests...${NC}"
if cargo test --lib &> /dev/null; then
    echo -e "${GREEN}✅ Tests: OK${NC}"
else
    echo -e "${RED}❌ Tests: FAILED${NC}"
    exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}🎉 All quick checks passed!${NC}"
echo -e "${GRAY}⏱️  Duration: ${DURATION}s${NC}"
echo -e "${CYAN}🚀 Ready for commit!${NC}"
