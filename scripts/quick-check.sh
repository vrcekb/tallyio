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

# 📎 2. Clippy (Ultra-strict TallyIO configuration)
echo -e "${YELLOW}📎 Running ultra-strict clippy...${NC}"
if cargo clippy --all-targets --all-features -- \
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
    -v &> /dev/null; then
    echo -e "${GREEN}✅ Ultra-strict Clippy: OK${NC}"
else
    echo -e "${RED}❌ Ultra-strict Clippy: FAILED${NC}"
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
