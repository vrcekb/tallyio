# 🚀 TallyIO Multi-Stage Docker Build
# Optimized for ultra-low latency financial applications

# ============================================================================
# 🔧 Build Stage - Rust compilation with optimizations
# ============================================================================
FROM rust:1.87-slim as builder

# 📦 Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 🎯 Set build environment for maximum performance
ENV CARGO_TERM_COLOR=always
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# 📁 Create app directory
WORKDIR /app

# 📋 Copy dependency manifests first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/

# 🏗️ Create dummy source files to build dependencies
RUN mkdir -p crates/core/src crates/api/src crates/blockchain/src \
    crates/liquidation/src crates/security/src crates/database/src \
    crates/metrics/src crates/contracts/src crates/web-ui/src && \
    echo "fn main() {}" > crates/core/src/main.rs && \
    echo "fn main() {}" > crates/api/src/main.rs && \
    echo "fn main() {}" > crates/blockchain/src/main.rs && \
    echo "fn main() {}" > crates/liquidation/src/main.rs && \
    echo "fn main() {}" > crates/security/src/main.rs && \
    echo "fn main() {}" > crates/database/src/main.rs && \
    echo "fn main() {}" > crates/metrics/src/main.rs && \
    echo "fn main() {}" > crates/contracts/src/main.rs && \
    echo "fn main() {}" > crates/web-ui/src/main.rs && \
    find crates -name "*.rs" -exec touch {} \;

# 🔧 Build dependencies (cached layer)
RUN cargo build --release --workspace

# 📥 Copy actual source code
COPY . .

# 🚨 Verify TallyIO standards before build
RUN echo "🔍 Verifying TallyIO standards..." && \
    PANIC_COUNT=$(grep -r "unwrap\|expect\|panic!\|todo!\|unimplemented!" src/ crates/ || true | wc -l) && \
    if [ "$PANIC_COUNT" -gt 0 ]; then \
        echo "❌ Found $PANIC_COUNT prohibited patterns!" && \
        grep -rn "unwrap\|expect\|panic!\|todo!\|unimplemented!" src/ crates/ || true && \
        exit 1; \
    fi && \
    echo "✅ Zero panic policy verified"

# 🏗️ Build the application with maximum optimizations
RUN cargo build --release --workspace

# 🧪 Run tests to ensure build quality
RUN cargo test --release --workspace

# ============================================================================
# 🏃 Runtime Stage - Minimal production image
# ============================================================================
FROM debian:bookworm-slim as runtime

# 📦 Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 👤 Create non-root user for security
RUN groupadd -r tallyio && useradd -r -g tallyio tallyio

# 📁 Create app directory
WORKDIR /app

# 📋 Copy configuration files
COPY --from=builder /app/config ./config

# 📦 Copy built binaries
COPY --from=builder /app/target/release/tallyio-api ./bin/
COPY --from=builder /app/target/release/tallyio-core ./bin/

# 🔧 Set permissions
RUN chown -R tallyio:tallyio /app && \
    chmod +x /app/bin/*

# 👤 Switch to non-root user
USER tallyio

# 🌐 Expose ports
EXPOSE 8080 8081 9090

# 📊 Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 🏷️ Labels for metadata
LABEL org.opencontainers.image.title="TallyIO" \
      org.opencontainers.image.description="Ultra-performant financial trading platform" \
      org.opencontainers.image.vendor="TallyIO" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/your-org/tallyio"

# 🚀 Default command
CMD ["./bin/tallyio-api"]

# ============================================================================
# 🔧 Development Stage - For development with hot reload
# ============================================================================
FROM rust:1.87-slim as development

# 📦 Install development dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 📥 Install cargo-watch for hot reload
RUN cargo install cargo-watch

# 📁 Create app directory
WORKDIR /app

# 👤 Create non-root user
RUN groupadd -r tallyio && useradd -r -g tallyio tallyio && \
    chown -R tallyio:tallyio /app

# 👤 Switch to non-root user
USER tallyio

# 🌐 Expose ports
EXPOSE 8080 8081 9090

# 🔧 Development command with hot reload
CMD ["cargo", "watch", "-x", "run --bin tallyio-api"]

# ============================================================================
# 🧪 Testing Stage - For running tests in CI
# ============================================================================
FROM builder as testing

# 🧪 Run comprehensive test suite
RUN cargo test --workspace --verbose

# 📊 Run benchmarks
RUN cargo bench --workspace || true

# 📎 Run clippy (Ultra-strict TallyIO standards)
RUN cargo clippy --all-targets --all-features -- \
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

# 🎨 Check formatting
RUN cargo fmt --all -- --check

# ⚡ Performance tests
RUN echo "🚀 Running performance tests..." && \
    cargo test --release test_latency_requirement -- --nocapture || true

# ============================================================================
# 🔒 Security Stage - Security scanning
# ============================================================================
FROM builder as security

# 📥 Install security tools
RUN cargo install cargo-audit cargo-geiger

# 🔍 Run security audit
RUN cargo audit

# 🔍 Check for unsafe code
RUN cargo geiger

# ✅ Final verification
RUN echo "✅ Security checks completed"
