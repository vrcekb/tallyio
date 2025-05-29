# 📊 TallyIO Codecov Integration Setup

This guide explains how to set up Codecov integration for the TallyIO project to track code coverage automatically.

## 🚨 Current Status

The CI pipeline is configured for Codecov integration but requires the `CODECOV_TOKEN` secret to be configured in GitHub repository settings.

**Diagnostic Warning**: `Context access might be invalid: CODECOV_TOKEN` indicates the secret is not yet configured.

## 🔧 Setup Instructions

### Step 1: Create Codecov Account
1. Go to [https://codecov.io](https://codecov.io)
2. Sign up or log in with your GitHub account
3. Grant necessary permissions to access your repositories

### Step 2: Add Repository to Codecov
1. In Codecov dashboard, click "Add new repository"
2. Find and select your `tallyio` repository
3. Copy the repository token (starts with a UUID format)

### Step 3: Configure GitHub Secret
1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Set:
   - **Name**: `CODECOV_TOKEN`
   - **Value**: [paste the token from Codecov]
5. Click **"Add secret"**

### Step 4: Verify Integration
1. Push a commit or create a pull request
2. Check the CI workflow runs successfully
3. Verify coverage report appears in Codecov dashboard

## 🎯 TallyIO Coverage Requirements

The project enforces strict coverage requirements:

### Critical Modules (100% Coverage Required)
- `crates/core/` - Core engine
- `crates/security/` - Security modules
- `crates/liquidation/` - Liquidation strategies
- `crates/blockchain/` - Blockchain integration
- `crates/contracts/` - Smart contracts

### Other Modules (95% Coverage Required)
- `crates/database/` - Database abstractions
- `crates/metrics/` - Metrics & monitoring
- `crates/api/` - REST + WebSocket API

### Overall Project (95% Coverage Required)
- Minimum 95% overall coverage across all crates
- Coverage thresholds enforced by Codecov

## 🔍 Local Coverage Testing

You can test coverage locally using the provided script:

```bash
# Run the setup verification script
./scripts/setup-codecov.sh
```

This script will:
- ✅ Install required tools (grcov, nightly toolchain)
- ✅ Generate coverage report locally
- ✅ Test Codecov upload (dry run)
- ✅ Validate coverage meets TallyIO requirements

## 📊 Coverage Dashboard

Once configured, access your coverage dashboard at:
```
https://codecov.io/gh/[your-username]/tallyio
```

## 🚨 Troubleshooting

### "Context access might be invalid: CODECOV_TOKEN"
- **Cause**: GitHub secret not configured
- **Solution**: Follow Step 3 above to add the secret
- **VSCode Workaround**: The project includes `.vscode/settings.json` with YAML custom tags to reduce diagnostic noise

### "Codecov upload failed"
- **Cause**: Invalid token or network issues
- **Solution**: Verify token is correct and try again

### "Coverage below threshold"
- **Cause**: Insufficient test coverage
- **Solution**: Add more tests to reach required coverage levels

### CI Fails on Coverage Step
- **Cause**: Coverage requirements not met
- **Solution**: The CI is configured to continue even if Codecov fails, but you should still configure the token for proper tracking

### VSCode Diagnostic Warnings
- **Issue**: VSCode shows "Context access might be invalid" warnings
- **Status**: Expected behavior until secret is configured
- **Workaround**: Project includes VSCode settings to minimize diagnostic noise
- **Reference**: [GitHub Issue #222](https://github.com/github/vscode-github-actions/issues/222)

## 🔄 CI Workflow Behavior

The CI workflow is designed to be robust:

1. **With CODECOV_TOKEN**: Uploads coverage to Codecov dashboard
2. **Without CODECOV_TOKEN**: Generates coverage locally, shows setup instructions
3. **Upload Failure**: CI continues (doesn't fail the build)
4. **Coverage Validation**: Always runs regardless of upload status

## 📋 Next Steps

1. ✅ Configure `CODECOV_TOKEN` secret in GitHub
2. ✅ Push a commit to trigger CI
3. ✅ Verify coverage appears in Codecov dashboard
4. ✅ Add more tests if coverage is below requirements
5. ✅ Set up Codecov notifications and integrations as needed

## 🎉 Benefits

Once configured, you'll get:
- 📊 Automatic coverage tracking on every commit
- 📈 Coverage trends and history
- 🔍 Line-by-line coverage visualization
- 📝 Pull request coverage comments
- 🚨 Alerts when coverage drops below thresholds
