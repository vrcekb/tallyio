# 📊 TallyIO Codecov Integration Setup

**Complete guide for setting up code coverage analytics with ultra-strict requirements**

## 🎯 Overview

TallyIO uses Codecov for comprehensive code coverage analysis with the following requirements:
- **95% overall coverage minimum**
- **100% coverage for critical modules**
- **Automated coverage validation in CI/CD**
- **Pull request coverage reports**

## 🔧 Configuration Details

### Repository Information
- **Repository:** `vrcekb/tallyio`
- **Token:** `194732cc-0084-48cd-8af3-4d72abdd3cdc`
- **Language:** Rust
- **Coverage Tool:** `grcov` (following official Codecov Rust example)

## 📋 Setup Steps

### Step 1: Add GitHub Secret

**⚠️ REQUIRED:** Add the Codecov token as a GitHub repository secret:

1. Go to your GitHub repository: `https://github.com/vrcekb/tallyio`
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the secret:
   - **Name:** `CODECOV_TOKEN`
   - **Value:** `194732cc-0084-48cd-8af3-4d72abdd3cdc`

### Step 2: Verify CI Configuration

The CI pipeline is configured following the official Codecov Rust example:

```yaml
# .github/workflows/ci.yml
- name: 🦀 Setup Rust nightly (for coverage)
  uses: actions-rs/toolchain@v1
  with:
    toolchain: nightly
    override: true
    components: llvm-tools-preview

- name: 🔧 Build with coverage instrumentation
  run: cargo build --verbose
  env:
    CARGO_INCREMENTAL: '0'
    RUSTFLAGS: '-Zprofile -Ccodegen-units=1 -Cinline-threshold=0 -Clink-dead-code -Coverflow-checks=off -Cpanic=abort -Zpanic_abort_tests'

- name: 🧪 Run tests with coverage
  run: cargo test --verbose --all
  env:
    CARGO_INCREMENTAL: '0'
    RUSTFLAGS: '-Zprofile -Ccodegen-units=1 -Cinline-threshold=0 -Clink-dead-code -Coverflow-checks=off -Cpanic=abort -Zpanic_abort_tests'

- name: 📊 Generate coverage report with grcov
  uses: actions-rs/grcov@v0.1

- name: 📤 Upload coverage reports to Codecov
  uses: codecov/codecov-action@v5
  with:
    token: ${{ secrets.CODECOV_TOKEN }}
    slug: vrcekb/tallyio
    fail_ci_if_error: true
    verbose: true
```

### Step 3: Coverage Generation

TallyIO uses `grcov` following the official Codecov Rust example:

```bash
# Install grcov
cargo install grcov

# Set coverage environment variables
export CARGO_INCREMENTAL=0
export RUSTFLAGS="-Zprofile -Ccodegen-units=1 -Cinline-threshold=0 -Clink-dead-code -Coverflow-checks=off -Cpanic=abort -Zpanic_abort_tests"

# Build and test with coverage
cargo +nightly build --verbose
cargo +nightly test --verbose --all

# Generate coverage report
grcov . --binary-path ./target/debug/deps/ -s . -t lcov --branch --ignore-not-existing --ignore '../*' --ignore "/*" -o coverage.lcov
```

## 🎯 Coverage Requirements

### Strict Thresholds

| Component | Coverage Target | Threshold | Failure Policy |
|-----------|----------------|-----------|----------------|
| **Overall Project** | 95% | 1% | ❌ Fail CI |
| **Core Engine** | 100% | 0% | ❌ Fail CI |
| **Security** | 100% | 0% | ❌ Fail CI |
| **Liquidation** | 100% | 0% | ❌ Fail CI |
| **Blockchain** | 100% | 0% | ❌ Fail CI |
| **Smart Contracts** | 100% | 0% | ❌ Fail CI |
| **API Layer** | 95% | 1% | ❌ Fail CI |
| **Database** | 95% | 1% | ❌ Fail CI |
| **Metrics** | 95% | 1% | ❌ Fail CI |

### New Code Requirements

- **Patch Coverage:** 95% minimum
- **Critical Path Patches:** 100% required
- **No coverage regression** allowed

## 🔍 Local Coverage Testing

### Generate Coverage Locally

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Generate coverage report
cargo llvm-cov --all-features --workspace

# Generate HTML report
cargo llvm-cov --all-features --workspace --html

# Generate LCOV format (for Codecov)
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
```

### View Coverage Report

```bash
# Open HTML report
open target/llvm-cov/html/index.html

# View summary
cargo llvm-cov --all-features --workspace --summary-only
```

## 📊 Codecov Dashboard

### Access Your Dashboard

After setup, access your coverage dashboard at:
`https://codecov.io/gh/vrcekb/tallyio`

### Key Features

- **Coverage trends** over time
- **Pull request coverage** analysis
- **File-by-file coverage** breakdown
- **Critical path monitoring**
- **Coverage regression** detection

## 🚨 Quality Gates

### CI Validation

The pipeline automatically validates:

```bash
# Extract overall coverage percentage
OVERALL_COVERAGE=$(cargo llvm-cov --all-features --workspace --summary-only | grep "TOTAL" | awk '{print $4}' | sed 's/%//')

# Fail if below 95%
if (( $(echo "$OVERALL_COVERAGE < 95" | bc -l) )); then
  echo "❌ Coverage ${OVERALL_COVERAGE}% below required 95%!"
  exit 1
fi
```

### Pull Request Checks

Codecov will automatically:
- ✅ Comment on PRs with coverage analysis
- ✅ Show coverage diff for changed files
- ✅ Block merging if coverage drops below thresholds
- ✅ Highlight uncovered lines

## 🔧 Troubleshooting

### Common Issues

#### 1. Token Authentication Error
```
Error: Codecov token not found
```
**Solution:** Verify `CODECOV_TOKEN` secret is properly set in GitHub repository settings.

#### 2. Coverage Report Not Found
```
Error: No coverage reports found
```
**Solution:** Ensure `cargo llvm-cov` generates `lcov.info` file before upload step.

#### 3. Low Coverage Failure
```
Error: Coverage 89% is below target 95%
```
**Solution:** Add more tests to increase coverage or review coverage exclusions.

### Debug Commands

```bash
# Check if coverage file exists
ls -la lcov.info

# Validate LCOV format
head -20 lcov.info

# Test Codecov upload locally
curl -Os https://uploader.codecov.io/latest/linux/codecov
chmod +x codecov
./codecov -t 194732cc-0084-48cd-8af3-4d72abdd3cdc -f lcov.info
```

## 📈 Best Practices

### Writing Testable Code

```rust
// ✅ Good - Easy to test
pub fn calculate_fee(amount: u64, rate: f64) -> Result<u64, Error> {
    if rate < 0.0 || rate > 1.0 {
        return Err(Error::InvalidRate);
    }
    Ok((amount as f64 * rate) as u64)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_calculate_fee() -> Result<(), Error> {
        assert_eq!(calculate_fee(1000, 0.01)?, 10);
        assert!(calculate_fee(1000, -0.1).is_err());
        Ok(())
    }
}
```

### Coverage Optimization

1. **Test all error paths**
2. **Include edge cases**
3. **Test async code properly**
4. **Mock external dependencies**
5. **Use property-based testing**

## 🎉 Success Indicators

### When Setup is Complete

- ✅ GitHub secret `CODECOV_TOKEN` configured
- ✅ CI pipeline uploads coverage successfully
- ✅ Codecov dashboard shows repository data
- ✅ Pull requests show coverage comments
- ✅ Coverage thresholds enforced

### Monitoring

- 📊 **Daily:** Check coverage trends
- 📈 **Weekly:** Review uncovered critical paths
- 🎯 **Monthly:** Adjust coverage targets if needed

---

**🏆 With this setup, TallyIO maintains financial-grade code quality with comprehensive coverage analytics!**
