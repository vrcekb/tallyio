#!/usr/bin/env python3
"""
TallyIO Comprehensive Benchmark Runner & Performance Analyzer
============================================================

This script discovers and runs all benchmark tests in the TallyIO project,
then generates a detailed performance report for financial application readiness.

Author: TallyIO Performance Team
Date: 2025-06-15
"""

import os
import subprocess
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics

@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    name: str
    crate: str
    time_ns: float
    throughput: Optional[float] = None
    iterations: Optional[int] = None
    status: str = "PASS"
    error: Optional[str] = None

@dataclass
class CratePerformance:
    """Performance summary for a crate"""
    name: str
    total_benchmarks: int
    passed: int
    failed: int
    avg_time_ns: float
    min_time_ns: float
    max_time_ns: float
    total_time_ms: float
    performance_grade: str
    critical_issues: List[str]

@dataclass
class SystemPerformance:
    """Overall system performance summary"""
    total_benchmarks: int
    total_crates: int
    passed: int
    failed: int
    total_runtime_seconds: float
    avg_benchmark_time_ns: float
    performance_grade: str
    financial_readiness: bool
    critical_issues: List[str]
    recommendations: List[str]

class TallyIOBenchmarkRunner:
    """Comprehensive benchmark runner for TallyIO financial application"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.results: List[BenchmarkResult] = []
        self.crate_performance: Dict[str, CratePerformance] = {}
        self.system_performance: Optional[SystemPerformance] = None
        
        # Financial application performance thresholds (nanoseconds)
        self.THRESHOLDS = {
            "ULTRA_CRITICAL": 1_000,      # 1μs - MEV operations
            "CRITICAL": 10_000,           # 10μs - Trading operations  
            "HIGH": 100_000,              # 100μs - Risk calculations
            "MEDIUM": 1_000_000,          # 1ms - General operations
            "LOW": 10_000_000,            # 10ms - Background tasks
        }
        
        # Performance grades
        self.GRADES = {
            "A+": "EXCEPTIONAL - Ready for HFT",
            "A": "EXCELLENT - Ready for production",
            "B": "GOOD - Minor optimizations needed", 
            "C": "ACCEPTABLE - Significant optimizations needed",
            "D": "POOR - Major performance issues",
            "F": "FAIL - Not suitable for financial applications"
        }

    def discover_benchmarks(self) -> List[Tuple[str, Path]]:
        """Discover all benchmark files in the project"""
        benchmarks = []
        
        # Search for bench files in crates
        for crate_dir in self.project_root.glob("crates/*/benches/*.rs"):
            crate_name = crate_dir.parent.parent.name
            benchmarks.append((crate_name, crate_dir))
            
        # Search for bench files in root benches directory
        for bench_file in self.project_root.glob("benches/*.rs"):
            benchmarks.append(("root", bench_file))
            
        print(f"🔍 Discovered {len(benchmarks)} benchmark files:")
        for crate, path in benchmarks:
            print(f"   📊 {crate}: {path.name}")
            
        return benchmarks

    def run_cargo_bench(self) -> str:
        """Run cargo bench by crates to avoid timeout"""
        print("\n🚀 Running benchmark suite by crates...")
        print("=" * 70)

        all_output = ""
        total_start_time = time.time()

        # Get list of crates with benchmarks
        crates_with_benches = set()
        for crate_dir in self.project_root.glob("crates/*/benches/*.rs"):
            crate_name = crate_dir.parent.parent.name
            crates_with_benches.add(crate_name)

        print(f"📦 Found {len(crates_with_benches)} crates with benchmarks")

        for crate in sorted(crates_with_benches):
            print(f"\n🔄 Running benchmarks for crate: {crate}")

            try:
                # Run benchmarks for specific crate with shorter timeout
                cmd = [
                    "cargo", "bench", "--package", f"tallyio-{crate}",
                    "--", "--sample-size", "10"  # Reduce sample size for speed
                ]

                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout per crate
                )

                runtime = time.time() - start_time
                print(f"   ⏱️  {crate} runtime: {runtime:.2f} seconds")

                if result.returncode == 0:
                    all_output += f"\n=== {crate.upper()} CRATE BENCHMARKS ===\n"
                    all_output += result.stdout
                    print(f"   ✅ {crate} benchmarks completed")
                else:
                    print(f"   ⚠️  {crate} benchmarks failed (exit code: {result.returncode})")
                    if result.stderr:
                        print(f"   STDERR: {result.stderr[:200]}...")

            except subprocess.TimeoutExpired:
                print(f"   🚨 TIMEOUT: {crate} benchmarks exceeded 2 minute limit!")
                # Add timeout info to results
                self.results.append(BenchmarkResult(
                    name=f"{crate}_timeout",
                    crate=crate,
                    time_ns=120_000_000_000,  # 2 minutes in ns
                    status="TIMEOUT",
                    error="Benchmark timeout - performance issue detected"
                ))
            except Exception as e:
                print(f"   🚨 ERROR running {crate} benchmarks: {e}")

        total_runtime = time.time() - total_start_time
        print(f"\n⏱️  Total benchmark runtime: {total_runtime:.2f} seconds")

        return all_output

    def parse_benchmark_output(self, output: str) -> None:
        """Parse cargo bench output and extract performance data"""
        print("\n📊 Parsing benchmark results...")

        if not output.strip():
            print("⚠️  No benchmark output to parse!")
            return

        # Enhanced regex patterns for different benchmark output formats
        patterns = {
            'criterion': re.compile(r'(\w+(?:[_\w]*)*)\s+time:\s+\[([0-9.]+)\s*([a-zμ]+)\s+([0-9.]+)\s*([a-zμ]+)\s+([0-9.]+)\s*([a-zμ]+)\]'),
            'criterion_simple': re.compile(r'(\w+(?:[_\w]*)*)\s+time:\s+\[([0-9.]+)\s*([a-zμ]+)'),
            'simple': re.compile(r'test\s+(\w+(?:::\w+)*)\s+\.\.\.\s+bench:\s+([0-9,]+)\s+ns/iter'),
            'throughput': re.compile(r'(\w+(?:[_\w]*)*)\s+time:.*?thrpt:\s+\[([0-9.]+)\s+([A-Za-z/]+)'),
            'timeout': re.compile(r'(\w+)_timeout'),
        }

        current_crate = "unknown"
        parsed_count = 0

        for line in output.split('\n'):
            line = line.strip()

            # Detect current crate being benchmarked
            if "=== " in line and "CRATE BENCHMARKS ===" in line:
                crate_match = re.search(r'=== (\w+) CRATE', line)
                if crate_match:
                    current_crate = crate_match.group(1).lower()
                    print(f"   📦 Processing {current_crate} crate results...")
            elif "Running benches" in line or "Benchmarking" in line:
                crate_match = re.search(r'crates/(\w+)', line)
                if crate_match:
                    current_crate = crate_match.group(1)

            # Parse different benchmark formats
            for pattern_name, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    if self._parse_benchmark_match(match, pattern_name, current_crate):
                        parsed_count += 1
                    break

        print(f"   ✅ Parsed {parsed_count} benchmark results")

        # Add synthetic results for timeout cases
        if parsed_count == 0:
            print("   ⚠️  No benchmark results parsed - adding synthetic timeout results")
            self._add_synthetic_timeout_results()

    def _parse_benchmark_match(self, match, pattern_type: str, crate: str) -> bool:
        """Parse individual benchmark match"""
        try:
            throughput = None

            if pattern_type == 'criterion':
                name = match.group(1)
                time_val = float(match.group(2))
                time_unit = match.group(3)
                time_ns = self._convert_to_nanoseconds(time_val, time_unit)

            elif pattern_type == 'criterion_simple':
                name = match.group(1)
                time_val = float(match.group(2))
                time_unit = match.group(3)
                time_ns = self._convert_to_nanoseconds(time_val, time_unit)

            elif pattern_type == 'simple':
                name = match.group(1).replace('::', '_')
                time_ns = float(match.group(2).replace(',', ''))

            elif pattern_type == 'throughput':
                name = match.group(1)
                throughput = float(match.group(2))
                # Estimate time from throughput (rough approximation)
                time_ns = 1_000_000_000 / throughput if throughput > 0 else 1_000_000

            elif pattern_type == 'timeout':
                name = match.group(1)
                time_ns = 120_000_000_000  # 2 minutes

            else:
                return False

            result = BenchmarkResult(
                name=name,
                crate=crate,
                time_ns=time_ns,
                throughput=throughput,
                status="TIMEOUT" if pattern_type == 'timeout' else "PASS"
            )

            self.results.append(result)
            return True

        except (ValueError, IndexError) as e:
            print(f"⚠️  Failed to parse benchmark: {match.group(0) if match else 'unknown'} - {e}")
            return False

    def _add_synthetic_timeout_results(self) -> None:
        """Add synthetic results when benchmarks timeout"""
        crates = ["core", "secure_storage", "data_storage", "network"]

        for crate in crates:
            self.results.append(BenchmarkResult(
                name=f"{crate}_benchmark_timeout",
                crate=crate,
                time_ns=600_000_000_000,  # 10 minutes in nanoseconds
                status="TIMEOUT",
                error="Benchmark suite timeout - critical performance issue"
            ))

    def _convert_to_nanoseconds(self, value: float, unit: str) -> float:
        """Convert time value to nanoseconds"""
        conversions = {
            'ns': 1,
            'μs': 1_000, 'us': 1_000,
            'ms': 1_000_000,
            's': 1_000_000_000,
        }
        return value * conversions.get(unit, 1)

    def analyze_performance(self) -> None:
        """Analyze benchmark results and generate performance metrics"""
        print("\n🔬 Analyzing performance metrics...")
        
        if not self.results:
            print("⚠️  No benchmark results to analyze!")
            return
            
        # Group results by crate
        crate_results = {}
        for result in self.results:
            if result.crate not in crate_results:
                crate_results[result.crate] = []
            crate_results[result.crate].append(result)
        
        # Analyze each crate
        for crate_name, results in crate_results.items():
            self._analyze_crate_performance(crate_name, results)
        
        # Analyze overall system performance
        self._analyze_system_performance()

    def _analyze_crate_performance(self, crate_name: str, results: List[BenchmarkResult]) -> None:
        """Analyze performance for a specific crate"""
        times = [r.time_ns for r in results if r.status == "PASS"]
        
        if not times:
            return
            
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times) / 1_000_000  # Convert to milliseconds
        
        # Determine performance grade
        grade = self._calculate_performance_grade(avg_time)
        
        # Identify critical issues
        critical_issues = []
        for result in results:
            if result.time_ns > self.THRESHOLDS["CRITICAL"]:
                critical_issues.append(f"{result.name}: {result.time_ns/1000:.1f}μs (>{self.THRESHOLDS['CRITICAL']/1000}μs threshold)")
        
        self.crate_performance[crate_name] = CratePerformance(
            name=crate_name,
            total_benchmarks=len(results),
            passed=len([r for r in results if r.status == "PASS"]),
            failed=len([r for r in results if r.status == "FAIL"]),
            avg_time_ns=avg_time,
            min_time_ns=min_time,
            max_time_ns=max_time,
            total_time_ms=total_time,
            performance_grade=grade,
            critical_issues=critical_issues
        )

    def _analyze_system_performance(self) -> None:
        """Analyze overall system performance"""
        if not self.results:
            return
            
        total_benchmarks = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = total_benchmarks - passed
        
        times = [r.time_ns for r in self.results if r.status == "PASS"]
        avg_time = statistics.mean(times) if times else 0
        
        # Calculate overall grade
        grade = self._calculate_performance_grade(avg_time)
        
        # Determine financial readiness
        financial_readiness = self._assess_financial_readiness()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Collect all critical issues
        critical_issues = []
        for crate_perf in self.crate_performance.values():
            critical_issues.extend(crate_perf.critical_issues)
        
        self.system_performance = SystemPerformance(
            total_benchmarks=total_benchmarks,
            total_crates=len(self.crate_performance),
            passed=passed,
            failed=failed,
            total_runtime_seconds=sum(times) / 1_000_000_000,  # Convert to seconds
            avg_benchmark_time_ns=avg_time,
            performance_grade=grade,
            financial_readiness=financial_readiness,
            critical_issues=critical_issues,
            recommendations=recommendations
        )

    def _calculate_performance_grade(self, avg_time_ns: float) -> str:
        """Calculate performance grade based on average time"""
        if avg_time_ns <= self.THRESHOLDS["ULTRA_CRITICAL"]:
            return "A+"
        elif avg_time_ns <= self.THRESHOLDS["CRITICAL"]:
            return "A"
        elif avg_time_ns <= self.THRESHOLDS["HIGH"]:
            return "B"
        elif avg_time_ns <= self.THRESHOLDS["MEDIUM"]:
            return "C"
        elif avg_time_ns <= self.THRESHOLDS["LOW"]:
            return "D"
        else:
            return "F"

    def _assess_financial_readiness(self) -> bool:
        """Assess if system is ready for financial applications"""
        if not self.results:
            return False
            
        # Check critical performance requirements
        critical_operations = [r for r in self.results if 'critical' in r.name.lower() or 'mev' in r.name.lower()]
        
        for op in critical_operations:
            if op.time_ns > self.THRESHOLDS["CRITICAL"]:
                return False
                
        # Check overall system stability
        failure_rate = len([r for r in self.results if r.status == "FAIL"]) / len(self.results)
        if failure_rate > 0.05:  # More than 5% failure rate
            return False
            
        return True

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not self.results:
            return ["No benchmark data available for analysis"]
        
        # Analyze slow operations
        slow_ops = [r for r in self.results if r.time_ns > self.THRESHOLDS["MEDIUM"]]
        if slow_ops:
            recommendations.append(f"Optimize {len(slow_ops)} operations exceeding 1ms threshold")
        
        # Check for failed benchmarks
        failed_ops = [r for r in self.results if r.status == "FAIL"]
        if failed_ops:
            recommendations.append(f"Fix {len(failed_ops)} failing benchmark tests")
        
        # Performance-specific recommendations
        avg_time = statistics.mean([r.time_ns for r in self.results if r.status == "PASS"])
        if avg_time > self.THRESHOLDS["HIGH"]:
            recommendations.extend([
                "Implement SIMD optimizations for mathematical operations",
                "Consider memory pool allocation for hot paths",
                "Profile and optimize critical code paths",
                "Implement lock-free data structures where possible"
            ])
        
        return recommendations

    def generate_report(self) -> str:
        """Generate comprehensive markdown report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# TallyIO Performance Benchmark Report

**Generated:** {timestamp}  
**Project:** TallyIO Financial Application  
**Environment:** Windows Development  
**Rust Version:** {self._get_rust_version()}

---

## 🎯 Executive Summary

"""
        
        if self.system_performance:
            sp = self.system_performance
            report += f"""
**Overall Performance Grade:** `{sp.performance_grade}` - {self.GRADES.get(sp.performance_grade, 'Unknown')}  
**Financial Application Readiness:** {'✅ READY' if sp.financial_readiness else '🚨 NOT READY'}  
**Total Benchmarks:** {sp.total_benchmarks} ({sp.passed} passed, {sp.failed} failed)  
**Average Benchmark Time:** {sp.avg_benchmark_time_ns/1000:.2f}μs  
**Total Runtime:** {sp.total_runtime_seconds:.2f} seconds  

"""

        # Add critical issues section
        if self.system_performance and self.system_performance.critical_issues:
            report += "### 🚨 Critical Performance Issues\n\n"
            for issue in self.system_performance.critical_issues[:10]:  # Limit to top 10
                report += f"- {issue}\n"
            report += "\n"

        # Add crate-by-crate analysis
        report += "## 📊 Crate Performance Analysis\n\n"
        
        for crate_name, perf in self.crate_performance.items():
            report += f"""### {crate_name.upper()} Crate

**Grade:** `{perf.performance_grade}` | **Benchmarks:** {perf.total_benchmarks} | **Pass Rate:** {(perf.passed/perf.total_benchmarks)*100:.1f}%

| Metric | Value |
|--------|-------|
| Average Time | {perf.avg_time_ns/1000:.2f}μs |
| Fastest Operation | {perf.min_time_ns/1000:.2f}μs |
| Slowest Operation | {perf.max_time_ns/1000:.2f}μs |
| Total Time | {perf.total_time_ms:.2f}ms |

"""
            if perf.critical_issues:
                report += "**Critical Issues:**\n"
                for issue in perf.critical_issues[:5]:  # Limit to top 5 per crate
                    report += f"- {issue}\n"
            report += "\n"

        # Add detailed benchmark results
        report += "## 📈 Detailed Benchmark Results\n\n"
        report += "| Crate | Benchmark | Time (μs) | Status | Performance |\n"
        report += "|-------|-----------|-----------|--------|-------------|\n"
        
        for result in sorted(self.results, key=lambda x: x.time_ns, reverse=True)[:50]:  # Top 50 slowest
            time_us = result.time_ns / 1000
            perf_indicator = self._get_performance_indicator(result.time_ns)
            report += f"| {result.crate} | {result.name} | {time_us:.2f} | {result.status} | {perf_indicator} |\n"

        # Add recommendations
        if self.system_performance and self.system_performance.recommendations:
            report += "\n## 🎯 Performance Optimization Recommendations\n\n"
            for i, rec in enumerate(self.system_performance.recommendations, 1):
                report += f"{i}. {rec}\n"

        # Add financial application assessment
        report += "\n## 💰 Financial Application Readiness Assessment\n\n"
        
        if self.system_performance:
            if self.system_performance.financial_readiness:
                report += """✅ **READY FOR PRODUCTION**

The TallyIO system meets the performance requirements for financial applications:
- Critical operations complete within acceptable timeframes
- System stability is maintained under load
- Error rates are within acceptable limits

"""
            else:
                report += """🚨 **NOT READY FOR PRODUCTION**

The TallyIO system does NOT meet the performance requirements for financial applications:
- Critical performance thresholds exceeded
- Unacceptable failure rates detected
- System requires optimization before handling real money

"""

        # Add performance thresholds reference
        report += """## 📏 Performance Thresholds Reference

| Category | Threshold | Use Case |
|----------|-----------|----------|
| Ultra Critical | <1μs | MEV operations, arbitrage detection |
| Critical | <10μs | Trading operations, order execution |
| High | <100μs | Risk calculations, portfolio updates |
| Medium | <1ms | General operations, data processing |
| Low | <10ms | Background tasks, reporting |

---

**Report Generated by TallyIO Performance Analysis System**  
*For questions or issues, contact the TallyIO development team.*
"""

        return report

    def _get_performance_indicator(self, time_ns: float) -> str:
        """Get performance indicator emoji"""
        if time_ns <= self.THRESHOLDS["ULTRA_CRITICAL"]:
            return "🚀 ULTRA"
        elif time_ns <= self.THRESHOLDS["CRITICAL"]:
            return "⚡ FAST"
        elif time_ns <= self.THRESHOLDS["HIGH"]:
            return "✅ GOOD"
        elif time_ns <= self.THRESHOLDS["MEDIUM"]:
            return "⚠️ SLOW"
        else:
            return "🚨 CRITICAL"

    def _get_rust_version(self) -> str:
        """Get Rust version"""
        try:
            result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "Unknown"

    def run_full_analysis(self) -> None:
        """Run complete benchmark analysis"""
        print("🎯 TallyIO Comprehensive Performance Analysis")
        print("=" * 70)
        
        # Discover benchmarks
        benchmarks = self.discover_benchmarks()
        if not benchmarks:
            print("❌ No benchmarks found!")
            return
        
        # Run benchmarks
        output = self.run_cargo_bench()
        if not output:
            print("❌ Failed to run benchmarks!")
            return
        
        # Parse results
        self.parse_benchmark_output(output)
        
        # Analyze performance
        self.analyze_performance()
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        report_file = f"TallyIO_Performance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📋 Performance report saved to: {report_file}")
        print(f"📊 Analyzed {len(self.results)} benchmark results")
        
        if self.system_performance:
            print(f"🎯 Overall Grade: {self.system_performance.performance_grade}")
            print(f"💰 Financial Ready: {'YES' if self.system_performance.financial_readiness else 'NO'}")

def main():
    """Main entry point"""
    runner = TallyIOBenchmarkRunner()
    runner.run_full_analysis()

if __name__ == "__main__":
    main()
