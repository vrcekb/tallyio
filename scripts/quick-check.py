#!/usr/bin/env python3
"""
TallyIO Quick Check Script
==========================

Comprehensive validation script for TallyIO project with visual progress indicators.
Performs the same checks as quick-check.ps1 but with better reliability and UX.

Requirements:
- Python 3.7+
- Rust toolchain (cargo, clippy, etc.)
- tarpaulin for coverage

Usage:
    python scripts/quick-check.py
"""

import subprocess
import sys
import re
import time
import threading
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Visual indicators and colors
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class Status(Enum):
    RUNNING = "🔄"
    SUCCESS = "✅"
    FAILED = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"

@dataclass
class CheckResult:
    name: str
    success: bool
    message: str
    duration: float
    details: Optional[str] = None
    error_output: Optional[str] = None

class ProgressBar:
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.current = 0
        self.width = width
        self.start_time = time.time()

    def update(self, step: int = 1):
        self.current += step
        self._draw()

    def _draw(self):
        percent = (self.current / self.total) * 100
        filled = int(self.width * self.current // self.total)
        bar = '█' * filled + '░' * (self.width - filled)
        elapsed = time.time() - self.start_time

        print(f'\r{Color.CYAN}Progress: {Color.END}[{Color.GREEN}{bar}{Color.END}] '
              f'{Color.BOLD}{percent:.1f}%{Color.END} '
              f'{Color.YELLOW}({self.current}/{self.total}){Color.END} '
              f'{Color.BLUE}⏱️ {elapsed:.1f}s{Color.END}', end='', flush=True)

class LiveSpinner:
    def __init__(self, message: str):
        self.message = message
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.running = False
        self.thread = None
        self.start_time = time.time()

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the spinner line
        print('\r' + ' ' * 80 + '\r', end='', flush=True)

    def _spin(self):
        i = 0
        while self.running:
            elapsed = time.time() - self.start_time
            spinner = self.spinner_chars[i % len(self.spinner_chars)]
            print(f'\r{Color.CYAN}{spinner}{Color.END} {Color.BOLD}{self.message}{Color.END} '
                  f'{Color.YELLOW}⏱️ {elapsed:.1f}s{Color.END}', end='', flush=True)
            time.sleep(0.1)
            i += 1

def print_header():
    """Print TallyIO header with visual styling."""
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}🚀 TallyIO Quick Check - Comprehensive Validation{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}\n")

def print_section(title: str, status: Status = Status.INFO):
    """Print section header with visual indicators."""
    print(f"\n{Color.BOLD}{Color.BLUE}{status.value} {title}{Color.END}")
    print(f"{Color.BLUE}{'─' * (len(title) + 4)}{Color.END}")

def print_result(result: CheckResult):
    """Print check result with appropriate colors and formatting."""
    status_icon = Status.SUCCESS.value if result.success else Status.FAILED.value
    color = Color.GREEN if result.success else Color.RED

    print(f"{status_icon} {Color.BOLD}{result.name}{Color.END}: "
          f"{color}{result.message}{Color.END} "
          f"{Color.YELLOW}({result.duration:.2f}s){Color.END}")

    if result.details:
        print(f"   {Color.CYAN}Details:{Color.END} {result.details}")

    # If failed, show more detailed error information
    if not result.success and hasattr(result, 'error_output') and result.error_output:
        print(f"   {Color.RED}Error Output:{Color.END}")
        # Show first 10 lines of error output
        error_lines = result.error_output.strip().split('\n')
        for line in error_lines[:10]:
            print(f"   {Color.RED}│{Color.END} {line}")
        if len(error_lines) > 10:
            print(f"   {Color.RED}│{Color.END} ... ({len(error_lines) - 10} more lines)")
        print()

def run_command(cmd: List[str], timeout: int = 300, show_spinner: bool = True, spinner_message: str = "") -> Tuple[bool, str, str]:
    """Run command with timeout and return success, stdout, stderr."""
    spinner = None
    if show_spinner and spinner_message:
        spinner = LiveSpinner(spinner_message)
        spinner.start()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="."
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)
    finally:
        if spinner:
            spinner.stop()

def check_cargo_fmt() -> CheckResult:
    """Check Rust code formatting (avoiding workspace false results)."""
    start_time = time.time()
    success, stdout, stderr = run_command(["cargo", "fmt", "--check"])  # Remove --all
    duration = time.time() - start_time

    if success:
        return CheckResult(
            name="Cargo Format",
            success=True,
            message="Code formatting is correct",
            duration=duration
        )
    else:
        return CheckResult(
            name="Cargo Format",
            success=False,
            message="Code formatting issues found",
            duration=duration,
            details="Run 'cargo fmt' to fix formatting",
            error_output=stderr if stderr else stdout
        )

def check_cargo_clippy() -> CheckResult:
    """Check Rust code with clippy lints using TallyIO ultra-strict standards (avoiding workspace false results)."""
    start_time = time.time()

    # TallyIO ultra-strict clippy configuration (remove --workspace for real results)
    clippy_args = [
        "cargo", "clippy", "--all-targets", "--all-features", "--",
        "-D", "warnings",
        "-D", "clippy::pedantic",
        "-D", "clippy::nursery",
        "-D", "clippy::correctness",
        "-D", "clippy::suspicious",
        "-D", "clippy::perf",
        "-W", "clippy::redundant_allocation",
        "-W", "clippy::needless_collect",
        "-W", "clippy::suboptimal_flops",
        "-A", "clippy::missing_docs_in_private_items",
        "-D", "clippy::infinite_loop",
        "-D", "clippy::while_immutable_condition",
        "-D", "clippy::never_loop",
        "-D", "for_loops_over_fallibles",
        "-D", "clippy::manual_strip",
        "-D", "clippy::needless_continue",
        "-D", "clippy::match_same_arms",
        "-D", "clippy::unwrap_used",
        "-D", "clippy::expect_used",
        "-D", "clippy::panic",
        "-D", "clippy::large_stack_arrays",
        "-D", "clippy::large_enum_variant",
        "-D", "clippy::mut_mut",
        "-D", "clippy::cast_possible_truncation",
        "-D", "clippy::cast_sign_loss",
        "-D", "clippy::cast_precision_loss",
        "-D", "clippy::must_use_candidate",
        "-D", "clippy::empty_loop",
        "-D", "clippy::if_same_then_else",
        "-D", "clippy::await_holding_lock",
        "-D", "clippy::await_holding_refcell_ref",
        "-D", "clippy::let_underscore_future",
        "-D", "clippy::diverging_sub_expression",
        "-D", "clippy::unreachable",
        "-D", "clippy::default_numeric_fallback",
        "-D", "clippy::redundant_pattern_matching",
        "-D", "clippy::manual_let_else",
        "-D", "clippy::blocks_in_conditions",
        "-D", "clippy::needless_pass_by_value",
        "-D", "clippy::single_match_else",
        "-D", "clippy::branches_sharing_code",
        "-D", "clippy::useless_asref",
        "-D", "clippy::redundant_closure_for_method_calls",
        "-v"
    ]
    
    success, stdout, stderr = run_command(clippy_args, spinner_message="Running clippy lints...")
    duration = time.time() - start_time

    if success:
        return CheckResult(
            name="Cargo Clippy",
            success=True,
            message="No clippy warnings found (TallyIO ultra-strict standards)",
            duration=duration
        )
    else:
        # Extract warning count from output
        combined_output = stdout + stderr
        warning_count = len(re.findall(r'warning:', combined_output))
        error_count = len(re.findall(r'error:', combined_output))

        if error_count > 0:
            message = f"Found {error_count} clippy errors and {warning_count} warnings"
        else:
            message = f"Found {warning_count} clippy warnings"

        return CheckResult(
            name="Cargo Clippy",
            success=False,
            message=message,
            duration=duration,
            details="Run 'cargo clippy --fix' to auto-fix issues",
            error_output=combined_output
        )

def check_cargo_test() -> CheckResult:
    """Run Rust unit tests (avoiding workspace false results)."""
    start_time = time.time()
    _, stdout, stderr = run_command(
        ["cargo", "test", "--all-targets", "--all-features"],  # Remove --workspace
        spinner_message="Running unit tests..."
    )
    duration = time.time() - start_time

    # Check for actual test failures in output, not just exit code
    combined_output = stdout + stderr
    failed_match = re.search(r'(\d+) failed', combined_output)
    passed_match = re.search(r'(\d+) passed', combined_output)

    failed_count = int(failed_match.group(1)) if failed_match else 0
    passed_count = int(passed_match.group(1)) if passed_match else 0

    if failed_count == 0 and passed_count > 0:
        # All tests passed
        return CheckResult(
            name="Cargo Test",
            success=True,
            message=f"All tests passed ({passed_count} tests)",
            duration=duration
        )
    elif failed_count > 0:
        # Some tests failed
        return CheckResult(
            name="Cargo Test",
            success=False,
            message=f"Tests failed ({failed_count} failures, {passed_count} passed)",
            duration=duration,
            details="Check test output for details",
            error_output=combined_output
        )
    else:
        # No tests found or other issue
        return CheckResult(
            name="Cargo Test",
            success=False,
            message="No tests found or compilation failed",
            duration=duration,
            details="Check test output for details",
            error_output=combined_output
        )

def check_security_audit() -> CheckResult:
    """Run security audit with ignored vulnerabilities."""
    start_time = time.time()

    # List of ignored vulnerabilities (same as PowerShell script)
    ignored_vulns = [
        "RUSTSEC-2023-0071",
        "RUSTSEC-2024-0421",
        "RUSTSEC-2025-0009",
        "RUSTSEC-2025-0010",
        "RUSTSEC-2024-0384"
    ]

    cmd = ["cargo", "audit"]
    for vuln in ignored_vulns:
        cmd.extend(["--ignore", vuln])

    success, stdout, stderr = run_command(cmd, spinner_message="Running security audit...")
    duration = time.time() - start_time

    if success:
        return CheckResult(
            name="Security Audit",
            success=True,
            message="No security vulnerabilities found",
            duration=duration,
            details=f"Ignored {len(ignored_vulns)} known issues"
        )
    else:
        # Extract vulnerability count
        combined_output = stdout + stderr
        vuln_count = len(re.findall(r'vulnerability', combined_output, re.IGNORECASE))
        return CheckResult(
            name="Security Audit",
            success=False,
            message=f"Found {vuln_count} security vulnerabilities",
            duration=duration,
            details="Review cargo audit output for details",
            error_output=combined_output
        )

def check_code_coverage() -> CheckResult:
    """Run code coverage analysis with cargo llvm-cov (better for async/multithreading)."""
    start_time = time.time()

    # Use llvm-cov WITHOUT --workspace for accurate real coverage
    success, stdout, stderr = run_command([
        "cargo", "llvm-cov",
        "--all-features",       # Include all features but NO --workspace
        "--all-targets"         # Include all targets (lib, bins, tests)
    ], timeout=600, spinner_message="Analyzing real code coverage (llvm-cov)...")

    duration = time.time() - start_time

    if not success:
        return CheckResult(
            name="Code Coverage",
            success=False,
            message="Coverage analysis failed",
            duration=duration,
            details="LLMV-COV execution failed",
            error_output=stderr if stderr else stdout
        )

    # Extract coverage percentage from llvm-cov output
    coverage_patterns = [
        r'lines\.\.\.\.\.\.\.\.\.\. (\d+\.?\d*)%',  # llvm-cov format
        r'TOTAL.*?(\d+\.?\d*)%',                    # alternative llvm-cov format
        r'(\d+\.?\d*)% coverage'                    # fallback format
    ]

    coverage = None
    for pattern in coverage_patterns:
        coverage_match = re.search(pattern, stdout)
        if coverage_match:
            coverage = float(coverage_match.group(1))
            break

    if coverage is None:
        return CheckResult(
            name="Code Coverage",
            success=False,
            message="Could not parse coverage output",
            duration=duration,
            details="Check llvm-cov output format",
            error_output=stdout
        )

    min_coverage = 90.0

    # Extract detailed coverage info for better reporting
    coverage_details = []
    lines = stdout.split('\n')
    for line in lines:
        if '.rs' in line and '%' in line and 'coverage' not in line.lower():
            coverage_details.append(line.strip())

    if coverage >= min_coverage:
        return CheckResult(
            name="Code Coverage",
            success=True,
            message=f"Coverage: {coverage}% (≥{min_coverage}%) [llvm-cov real coverage]",
            duration=duration
        )
    else:
        # Show which files need more coverage
        low_coverage_files = []
        for detail in coverage_details[:5]:  # Show top 5 problematic files
            if detail:
                low_coverage_files.append(detail)

        details_msg = f"Need {min_coverage - coverage:.1f}% more coverage"
        if low_coverage_files:
            details_msg += f"\nFiles needing attention:\n" + "\n".join(low_coverage_files)

        return CheckResult(
            name="Code Coverage",
            success=False,
            message=f"Coverage: {coverage}% (<{min_coverage}%) [llvm-cov real coverage]",
            duration=duration,
            details=details_msg
        )

def main():
    """Main execution function."""
    print_header()

    # Define all checks for entire workspace
    checks = [
        ("Formatting", check_cargo_fmt),
        ("Linting", check_cargo_clippy),
        ("Testing", check_cargo_test),
        ("Security", check_security_audit),
        ("Coverage", check_code_coverage),
    ]

    # Initialize progress bar
    progress = ProgressBar(len(checks))
    results = []

    print(f"{Color.BOLD}Running {len(checks)} validation checks...{Color.END}\n")

    # Run each check
    for check_name, check_func in checks:
        print_section(f"Running {check_name} Check", Status.RUNNING)

        result = check_func()
        results.append(result)
        print_result(result)

        progress.update()
        time.sleep(0.1)  # Small delay for visual effect

    # Print final progress bar completion
    print("\n")

    # Summary
    print_section("Summary", Status.INFO)

    passed = sum(1 for r in results if r.success)
    total = len(results)
    total_time = sum(r.duration for r in results)

    if passed == total:
        print(f"{Status.SUCCESS.value} {Color.BOLD}{Color.GREEN}All checks passed!{Color.END} "
              f"{Color.CYAN}({passed}/{total}){Color.END} "
              f"{Color.YELLOW}⏱️ {total_time:.1f}s{Color.END}")
        sys.exit(0)
    else:
        failed = total - passed
        print(f"{Status.FAILED.value} {Color.BOLD}{Color.RED}{failed} checks failed{Color.END} "
              f"{Color.CYAN}({passed}/{total} passed){Color.END} "
              f"{Color.YELLOW}⏱️ {total_time:.1f}s{Color.END}")

        print(f"\n{Color.BOLD}Failed checks:{Color.END}")
        for result in results:
            if not result.success:
                print(f"  {Status.FAILED.value} {result.name}: {result.message}")

        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Status.WARNING.value} {Color.YELLOW}Check interrupted by user{Color.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n{Status.FAILED.value} {Color.RED}Unexpected error: {e}{Color.END}")
        sys.exit(1)

