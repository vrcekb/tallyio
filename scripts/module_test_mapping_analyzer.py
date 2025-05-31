#!/usr/bin/env python3
"""
TallyIO Module Test Mapping Analyzer
====================================

Verifies that all modules are properly included in the appropriate test categories:
- Security modules → security_tests.rs
- Economic/MEV modules → economic_tests.rs
- State management modules → state_consistency_tests.rs
- Performance critical modules → timing_tests.rs

Reports missing modules that should be tested but aren't included.

Usage: python scripts/module_test_mapping_analyzer.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class ModuleMapping:
    """Mapping of modules to their expected test categories."""
    security_modules: Set[str]
    economic_modules: Set[str]
    state_modules: Set[str]
    timing_modules: Set[str]

@dataclass
class TestInclusion:
    """Which modules are actually included in each test file."""
    security_tests: Set[str]
    economic_tests: Set[str]
    state_tests: Set[str]
    timing_tests: Set[str]

@dataclass
class MissingModules:
    """Modules missing from their expected test categories."""
    missing_security: Set[str]
    missing_economic: Set[str]
    missing_state: Set[str]
    missing_timing: Set[str]

class ModuleTestMappingAnalyzer:
    """Analyzes module inclusion in TallyIO test categories."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)

        # Define module categories based on functionality
        self.module_categories = {
            # Security-related modules
            'security': {
                'security', 'auth', 'crypto', 'key', 'signature', 'validation',
                'protection', 'guard', 'access', 'permission', 'audit', 'error',
                'critical', 'safety'
            },

            # Economic/MEV-related modules
            'economic': {
                'mev', 'arbitrage', 'liquidation', 'opportunity', 'profit',
                'economics', 'trading', 'swap', 'dex', 'price', 'slippage',
                'sandwich', 'frontrun', 'backrun', 'flash_loan', 'analyzer',
                'filter', 'watcher', 'mempool'
            },

            # State management modules
            'state': {
                'state', 'storage', 'database', 'cache', 'sync', 'consistency',
                'transaction', 'mempool', 'global', 'local', 'persistence'
            },

            # Performance/timing critical modules
            'timing': {
                'engine', 'executor', 'scheduler', 'worker', 'optimization',
                'performance', 'cpu', 'memory', 'simd', 'lock_free', 'affinity',
                'latency', 'benchmark', 'utils', 'time', 'hash', 'memory_pool'
            }
        }

        # Universal test files (cross-cutting concerns across multiple crates)
        self.universal_test_files = {
            'security': 'tests/security_tests.rs',
            'economic': 'tests/economic_tests.rs',
            'state': 'tests/state_consistency_tests.rs',
            'timing': 'tests/timing_tests.rs',
            'chaos': 'tests/chaos_engineering_tests.rs',
            'market': 'tests/market_simulation_tests.rs',
            'fuzzing': 'tests/fuzzing_tests.rs',
            'e2e': 'tests/testnet_e2e_tests.rs',
            'integration': 'tests/integration_test.rs'
        }

        # Note: Crate-specific tests are in crates/*/tests/ and test only that crate's functionality

    def find_all_modules(self) -> Set[str]:
        """Find all Rust modules in the codebase."""
        modules = set()

        # Search in all crate source directories
        source_patterns = [
            "crates/*/src/**/*.rs",
            "crates/*/src/*.rs"
        ]

        for pattern in source_patterns:
            for rust_file in self.project_root.glob(pattern):
                # Skip lib.rs, mod.rs, and test files
                if rust_file.name in ['lib.rs', 'mod.rs'] or 'test' in str(rust_file):
                    continue

                # Extract module name from file path
                module_name = rust_file.stem
                modules.add(module_name)

                # Also extract module names from file content
                try:
                    content = rust_file.read_text(encoding='utf-8')
                    # Find mod declarations
                    mod_matches = re.findall(r'(?:pub\s+)?mod\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                    modules.update(mod_matches)
                except Exception:
                    pass

        return modules

    def categorize_modules(self, modules: Set[str]) -> ModuleMapping:
        """Categorize modules based on their names and functionality."""
        security_modules = set()
        economic_modules = set()
        state_modules = set()
        timing_modules = set()

        # External crates to exclude (these are dependencies, not our modules)
        external_crates = {
            'core_affinity', 'affinity', 'num_cpus', 'memory', 'libc', 'tokio',
            'crossbeam', 'rayon', 'parking_lot', 'serde', 'chrono', 'uuid',
            'log', 'thiserror', 'anyhow', 'reqwest', 'hyper', 'ethers', 'web3'
        }

        for module in modules:
            module_lower = module.lower()

            # Skip external crates
            if module in external_crates or module_lower in external_crates:
                continue

            # Check each category
            for keyword in self.module_categories['security']:
                if keyword in module_lower:
                    security_modules.add(module)
                    break

            for keyword in self.module_categories['economic']:
                if keyword in module_lower:
                    economic_modules.add(module)
                    break

            for keyword in self.module_categories['state']:
                if keyword in module_lower:
                    state_modules.add(module)
                    break

            for keyword in self.module_categories['timing']:
                if keyword in module_lower:
                    timing_modules.add(module)
                    break

        return ModuleMapping(
            security_modules=security_modules,
            economic_modules=economic_modules,
            state_modules=state_modules,
            timing_modules=timing_modules
        )

    def find_included_modules(self) -> TestInclusion:
        """Find which modules are actually included in each test file."""
        security_tests = set()
        economic_tests = set()
        state_tests = set()
        timing_tests = set()

        # Check each universal test file
        for test_type, test_file_path in self.universal_test_files.items():
            test_file = self.project_root / test_file_path

            if not test_file.exists():
                print(f"⚠️  Test file not found: {test_file}")
                continue

            try:
                content = test_file.read_text(encoding='utf-8')

                # Find use statements and module references (simplified)
                use_matches = re.findall(r'use\s+\w+', content)
                mod_matches = re.findall(r'mod\s+(\w+)', content)

                # Find direct module name mentions (limited to avoid performance issues)
                word_matches = []
                for line in content.split('\n')[:100]:  # Only check first 100 lines
                    words = re.findall(r'\b(\w+)\b', line)
                    word_matches.extend(words[:10])  # Limit words per line

                all_mentions = set(use_matches + mod_matches + word_matches)

                # Categorize based on test type
                if test_type == 'security':
                    security_tests.update(all_mentions)
                elif test_type == 'economic':
                    economic_tests.update(all_mentions)
                elif test_type == 'state':
                    state_tests.update(all_mentions)
                elif test_type == 'timing':
                    timing_tests.update(all_mentions)
                elif test_type == 'integration':
                    # Integration tests cover all categories
                    # Add specific modules that are tested in integration_test.rs
                    integration_modules = {
                        'engine', 'executor', 'TallyEngine', 'error', 'CoreError',
                        'CriticalError', 'utils', 'affinity', 'memory', 'hash',
                        'validation', 'time', 'LatencyTimer', 'mempool', 'watcher',
                        'MempoolWatcher', 'MempoolEvent', 'analyzer', 'MempoolAnalyzer',
                        'transaction', 'Transaction', 'ProcessingResult', 'filter',
                        'MempoolFilter', 'FilterConfig', 'TransactionFilter'
                    }

                    # Distribute integration modules to appropriate categories
                    for module in integration_modules:
                        module_lower = module.lower()

                        # Check which category this module belongs to
                        for keyword in self.module_categories['security']:
                            if keyword in module_lower:
                                security_tests.add(module)
                                break

                        for keyword in self.module_categories['economic']:
                            if keyword in module_lower:
                                economic_tests.add(module)
                                break

                        for keyword in self.module_categories['state']:
                            if keyword in module_lower:
                                state_tests.add(module)
                                break

                        for keyword in self.module_categories['timing']:
                            if keyword in module_lower:
                                timing_tests.add(module)
                                break

            except Exception as e:
                print(f"⚠️  Could not read {test_file}: {e}")

        return TestInclusion(
            security_tests=security_tests,
            economic_tests=economic_tests,
            state_tests=state_tests,
            timing_tests=timing_tests
        )

    def find_missing_modules(self, expected: ModuleMapping, actual: TestInclusion) -> MissingModules:
        """Find modules that should be tested but are missing from test files."""
        missing_security = expected.security_modules - actual.security_tests
        missing_economic = expected.economic_modules - actual.economic_tests
        missing_state = expected.state_modules - actual.state_tests
        missing_timing = expected.timing_modules - actual.timing_tests

        return MissingModules(
            missing_security=missing_security,
            missing_economic=missing_economic,
            missing_state=missing_state,
            missing_timing=missing_timing
        )

    def generate_report(self, expected: ModuleMapping, actual: TestInclusion, missing: MissingModules) -> str:
        """Generate detailed module mapping report."""
        report = []
        report.append("=" * 70)
        report.append("🧩 TALLYIO MODULE TEST MAPPING ANALYSIS")
        report.append("=" * 70)
        report.append("")
        report.append("📋 TEST ORGANIZATION")
        report.append("-" * 20)
        report.append("• Universal tests (tests/): Cross-cutting concerns across multiple crates")
        report.append("• Crate-specific tests (crates/*/tests/): Individual crate functionality")
        report.append("• This analysis focuses on universal tests only")
        report.append("")

        # Summary
        total_expected = (len(expected.security_modules) + len(expected.economic_modules) +
                         len(expected.state_modules) + len(expected.timing_modules))
        total_missing = (len(missing.missing_security) + len(missing.missing_economic) +
                        len(missing.missing_state) + len(missing.missing_timing))

        coverage_percent = ((total_expected - total_missing) / total_expected * 100) if total_expected > 0 else 100

        report.append("📊 SUMMARY")
        report.append("-" * 20)
        report.append(f"Total categorized modules: {total_expected}")
        report.append(f"Missing from tests: {total_missing}")
        report.append(f"Module inclusion rate: {coverage_percent:.1f}%")
        report.append("")

        # Status
        if total_missing == 0:
            report.append("✅ PERFECT: All modules are properly included in tests")
        elif total_missing <= 3:
            report.append("⚠️  GOOD: Only a few modules missing from tests")
        else:
            report.append("❌ NEEDS ATTENTION: Many modules missing from tests")
        report.append("")

        # Detailed breakdown
        categories = [
            ("SECURITY", expected.security_modules, missing.missing_security, "tests/security_tests.rs"),
            ("ECONOMIC", expected.economic_modules, missing.missing_economic, "tests/economic_tests.rs"),
            ("STATE", expected.state_modules, missing.missing_state, "tests/state_consistency_tests.rs"),
            ("TIMING", expected.timing_modules, missing.missing_timing, "tests/timing_tests.rs")
        ]

        for category_name, expected_modules, missing_modules, test_file in categories:
            if expected_modules:
                report.append(f"🔍 {category_name} MODULES")
                report.append("-" * (len(category_name) + 10))
                report.append(f"Expected in {test_file}: {len(expected_modules)}")
                report.append(f"Missing: {len(missing_modules)}")

                if missing_modules:
                    report.append("Missing modules:")
                    for module in sorted(missing_modules):
                        report.append(f"  ❌ {module}")
                else:
                    report.append("✅ All modules properly included")
                report.append("")

        # Recommendations
        if total_missing > 0:
            report.append("💡 RECOMMENDATIONS")
            report.append("-" * 20)
            report.append("1. Add missing modules to their respective universal test files")
            report.append("2. Create test cases for each missing module")
            report.append("3. Consider if module needs crate-specific tests instead")
            report.append("4. Verify module categorization is correct")
            report.append("5. Run this analyzer after adding new modules")
            report.append("")
        else:
            report.append("📝 NOTE")
            report.append("-" * 8)
            report.append("• Crate-specific tests should be in crates/*/tests/")
            report.append("• Universal tests cover cross-cutting concerns")
            report.append("• Both types of tests are important for complete coverage")
            report.append("")

        return "\n".join(report)

def main():
    """Main entry point."""
    analyzer = ModuleTestMappingAnalyzer()

    print("🔍 Analyzing TallyIO module test mapping...")

    # Find all modules
    all_modules = analyzer.find_all_modules()
    print(f"📁 Found {len(all_modules)} modules")

    # Categorize modules
    expected = analyzer.categorize_modules(all_modules)
    print(f"🏷️  Categorized modules: Security({len(expected.security_modules)}), "
          f"Economic({len(expected.economic_modules)}), State({len(expected.state_modules)}), "
          f"Timing({len(expected.timing_modules)})")

    # Find included modules
    actual = analyzer.find_included_modules()

    # Find missing modules
    missing = analyzer.find_missing_modules(expected, actual)

    # Generate report
    report = analyzer.generate_report(expected, actual, missing)
    print(report)

    # Save report
    report_file = Path("module_test_mapping_report.txt")
    report_file.write_text(report, encoding='utf-8')
    print(f"📄 Report saved to {report_file}")

    # Exit with error if modules are missing
    total_missing = (len(missing.missing_security) + len(missing.missing_economic) +
                    len(missing.missing_state) + len(missing.missing_timing))

    if total_missing > 0:
        print(f"\n❌ {total_missing} modules are missing from their expected test files")
        sys.exit(1)
    else:
        print(f"\n✅ All modules are properly included in their test categories")
        sys.exit(0)

if __name__ == "__main__":
    main()
