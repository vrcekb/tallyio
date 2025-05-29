#!/bin/bash
# 🚀 TallyIO Ultra-Strict Clippy Check Script
# Enforces production-ready code quality standards

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 TallyIO Ultra-Strict Clippy Check${NC}"
echo -e "${BLUE}====================================${NC}"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Error: Cargo.toml not found. Run this script from the project root.${NC}"
    exit 1
fi

# Check if clippy is installed
if ! command -v cargo-clippy &> /dev/null; then
    echo -e "${YELLOW}⚠️  Installing clippy...${NC}"
    rustup component add clippy
fi

echo -e "${BLUE}🔍 Running ultra-strict Clippy checks...${NC}"

# Run the comprehensive Clippy check
cargo clippy --all-targets --all-features -- \
    -D warnings \
    -D clippy::pedantic \
    -D clippy::nursery \
    -D clippy::correctness \
    -D clippy::suspicious \
    -D clippy::perf \
    -W clippy::redundant_allocation \
    -W clippy::needless_collect \
    -W clippy::suboptimal_flops \
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

CLIPPY_EXIT_CODE=$?

if [ $CLIPPY_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All Clippy checks passed!${NC}"
    echo -e "${GREEN}🏆 Code meets TallyIO ultra-strict standards${NC}"
else
    echo -e "${RED}❌ Clippy checks failed!${NC}"
    echo -e "${RED}🚨 Code does not meet TallyIO standards${NC}"
    exit $CLIPPY_EXIT_CODE
fi

# Additional zero-panic check
echo -e "${BLUE}🔍 Checking for prohibited patterns...${NC}"

PANIC_PATTERNS="unwrap\|expect\|panic!\|todo!\|unimplemented!"
PANIC_COUNT=$(grep -r "$PANIC_PATTERNS" src/ crates/ 2>/dev/null | wc -l || echo "0")

if [ "$PANIC_COUNT" -gt 0 ]; then
    echo -e "${RED}❌ Found $PANIC_COUNT prohibited patterns:${NC}"
    grep -rn "$PANIC_PATTERNS" src/ crates/ 2>/dev/null || true
    echo -e "${RED}🚨 TallyIO zero-panic policy violated!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Zero-panic policy verified${NC}"
fi

echo -e "${GREEN}🎉 All TallyIO quality checks passed!${NC}"
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "${GREEN}  ✅ Clippy ultra-strict checks: PASSED${NC}"
echo -e "${GREEN}  ✅ Zero-panic policy: VERIFIED${NC}"
echo -e "${GREEN}  ✅ Production-ready: CONFIRMED${NC}"
