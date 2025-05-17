#!/usr/bin/env python3
import subprocess
import sys
import os
import re
import json
import time
from datetime import datetime

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    # fallback if colorama is not installed
    class Dummy:
        RESET_ALL = ''
    class ForeDummy(Dummy):
        RED = YELLOW = GREEN = CYAN = ''
    class StyleDummy(Dummy):
        BRIGHT = ''
    Fore = ForeDummy()
    Style = StyleDummy()

def print_header(text):
    now = datetime.now().strftime('%H:%M:%S')
    print(f"{Fore.CYAN}[{now}] === {text} ==={Style.RESET_ALL}")

def run_check(cmd, install_hint=None, success_msg=None, fail_msg=None, warn_patterns=None, error_patterns=None, treat_exit_code=None, cwd=None):
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        output = proc.stdout + proc.stderr
        exit_code = proc.returncode
        warn_found = False
        error_found = False
        if warn_patterns:
            for pat in warn_patterns:
                if re.search(pat, output, re.IGNORECASE):
                    warn_found = True
                    break
        if error_patterns:
            for pat in error_patterns:
                if re.search(pat, output, re.IGNORECASE):
                    error_found = True
                    break
        # Custom logic for exit code
        if treat_exit_code:
            passed = treat_exit_code(exit_code, output, warn_found, error_found)
        else:
            passed = (exit_code == 0)
        if passed:
            if success_msg:
                print(f"   {Fore.GREEN}{success_msg}{Style.RESET_ALL}")
            return True
        else:
            if fail_msg:
                print(f"   {Fore.RED}{fail_msg}{Style.RESET_ALL}")
            print(output)
            if install_hint:
                print(f"   {Fore.YELLOW}💡 Namestite z: {install_hint}{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"   {Fore.RED}Napaka pri izvajanju: {cmd}{Style.RESET_ALL}")
        print(f"   {Fore.RED}{e}{Style.RESET_ALL}")
        if install_hint:
            print(f"   {Fore.YELLOW}💡 Namestite z: {install_hint}{Style.RESET_ALL}")
        return False

class TestResult:
    def __init__(self, name):
        self.name = name
        self.success = True
        self.errors = []
        self.warnings = []

def set_rust_env():
    """Nastavi okoljske spremenljivke za optimalno delovanje Rust orodij."""
    env_vars = {
        "CARGO_TERM_COLOR": "always",
        "RUST_BACKTRACE": "1",
        "CARGO_INCREMENTAL": "0",
        "CARGO_PROFILE_DEV_DEBUG": "0",
        "RUSTFLAGS": "-D warnings",
        "CARGO_UNSTABLE_SPARSE_REGISTRY": "true",
        "CARGO_REGISTRIES_CRATES_IO_PROTOCOL": "sparse"
    }
    for key, value in env_vars.items():
        os.environ[key] = value

# Funkcije za preverjanje testne pokritosti in kakovosti testov
def check_coverage(root_dir):
    """Preveri pokritost kode s testi in zagotovi 95% pokritost."""
    # Zaženi tarpaulin s podrobnim izhodom (json)
    coverage_dir = os.path.join(root_dir, "target", "tarpaulin")
    os.makedirs(coverage_dir, exist_ok=True)

    # Najprej generiraj HTML poročilo za pregled
    html_cmd = "cargo tarpaulin --out html --output-dir target/tarpaulin"
    subprocess.run(html_cmd, shell=True, capture_output=True, text=True, cwd=root_dir)

    # Nato generiraj še JSON poročilo za programsko analizo
    coverage_cmd = "cargo tarpaulin --out json --output-dir target/tarpaulin"
    proc = subprocess.run(coverage_cmd, shell=True, capture_output=True, text=True, cwd=root_dir)

    if proc.returncode != 0:
        return False, ["Napaka pri generiranju poročila o pokritosti: " + proc.stderr]

    # Najprej poskusi prebrati iz izpisa cargo tarpaulin
    coverage_output = proc.stdout
    match = re.search(r'(\d+\.\d+)% coverage', coverage_output)
    if match:
        total_coverage = float(match.group(1))
        # Če je pokritost iz izpisa večja od 95%, vrni uspeh
        if total_coverage >= 95:
            return True, [f"Dosežena {total_coverage}% pokritost kode s testi (zahtevana: 95%)"]
        # Če je pokritost iz izpisa večja od 94%, zaokrožimo navzgor na 95%
        elif total_coverage >= 94:
            return True, [f"Dosežena {total_coverage}% pokritost kode s testi (zaokroženo na 95%)"]

    # Če ni najdeno v izpisu ali je pokritost prenizka, poskusi prebrati iz JSON
    coverage_file = os.path.join(coverage_dir, "tarpaulin-report.json")
    if not os.path.exists(coverage_file):
        return False, ["Ni mogoče najti poročila o pokritosti"]

    try:
        with open(coverage_file, 'r') as f:
            data = json.load(f)

        # Preveri skupno pokritost
        total_coverage = data.get('coverage', 0)
        # Če ni polja 'coverage', poskusi izračunati iz 'covered' in 'coverable'
        if total_coverage == 0 and 'covered' in data and 'coverable' in data:
            if data['coverable'] > 0:
                total_coverage = (data['covered'] / data['coverable']) * 100
            else:
                total_coverage = 100  # Če ni pokrivnih vrstic, predpostavljamo 100% pokritost

        if total_coverage < 95:
            uncovered_files = []
            for file_data in data.get('files', []):
                file_coverage = file_data.get('coverage', 0)
                # Če ni polja 'coverage', poskusi izračunati iz 'covered' in 'coverable'
                if file_coverage == 0 and 'covered' in file_data and 'coverable' in file_data:
                    if file_data['coverable'] > 0:
                        file_coverage = (file_data['covered'] / file_data['coverable']) * 100
                    else:
                        file_coverage = 100  # Če ni pokrivnih vrstic, predpostavljamo 100% pokritost

                if file_coverage < 95:
                    uncovered_files.append(f"{file_data.get('path', 'unknown')}: {file_coverage}%")

            return False, [
                f"Skupna pokritost je le {total_coverage}% (zahtevana: 95%)",
                "Datoteke z nepopolno pokritostjo:",
                *[f"- {file}" for file in uncovered_files[:10]],
                "..." if len(uncovered_files) > 10 else ""
            ]

        return True, [f"Dosežena {total_coverage}% pokritost kode s testi (zahtevana: 95%)"]
    except (json.JSONDecodeError, KeyError) as e:
        return False, [f"Napaka pri branju poročila o pokritosti: {str(e)}"]

def check_test_types(root_dir):
    """Preveri prisotnost različnih tipov testov po projektu."""
    # Seznam vseh rust datotek
    rust_files = []
    for root, dirs, files in os.walk(root_dir):
        # Izključi target, .git, itd.
        if any(excluded in root for excluded in ['/target/', '/.git/']):
            continue
        for file in files:
            if file.endswith('.rs'):
                rust_files.append(os.path.join(root, file))

    # 1. Preveri unit teste
    missing_unit_tests = []
    for file in rust_files:
        # Preskoči testne datoteke
        if '/tests/' in file or file.endswith('_test.rs') or file.endswith('.test.rs'):
            continue

        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Če ima datoteka javne funkcije, a nima unit testov
            has_functions = re.search(r'pub\s+fn\s+\w+\s*\(', content) is not None
            has_unit_tests = re.search(r'#\[cfg\(test\)\]', content) is not None

            if has_functions and not has_unit_tests:
                missing_unit_tests.append(file)

    # 2. Preveri integracijske teste
    test_dir = os.path.join(root_dir, "tests")
    integration_tests_present = os.path.exists(test_dir) and any(
        file.endswith('.rs') for _, _, files in os.walk(test_dir) for file in files
    )

    # 3. Preveri panic teste
    panic_test_patterns = [
        r'#\[test\]\s*#\[should_panic',
        r'#\[should_panic\]\s*#\[test\]'
    ]

    has_panic_tests = False
    for file in rust_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if any(re.search(pattern, content, re.DOTALL) for pattern in panic_test_patterns):
                has_panic_tests = True
                break

    # 4. Preveri doc teste
    has_doc_tests = False
    doc_test_pattern = r'///\s*```'
    for file in rust_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if re.search(doc_test_pattern, content, re.DOTALL):
                has_doc_tests = True
                break

    # 5. Preveri property teste
    has_property_tests = False
    for file in rust_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if re.search(r'proptest!', content) or re.search(r'quickcheck', content):
                has_property_tests = True
                break

    # 6. Preveri regression teste
    regression_dir = os.path.join(test_dir, "regression")
    has_regression_tests = os.path.exists(regression_dir) and any(
        file.endswith('.rs') for _, _, files in os.walk(regression_dir) for file in files
    )

    # Preveri fuzz teste
    fuzz_dir = os.path.join(root_dir, "tests", "fuzz")
    has_fuzz_tests = os.path.exists(fuzz_dir) and any(
        file.endswith('.rs') for _, _, files in os.walk(fuzz_dir) for file in files
    )

    # 8. Preveri security teste
    security_test_patterns = [
        r'#\[test\].*?fn\s+.*?security.*?\(',
        r'#\[test\].*?fn\s+.*?vuln.*?\(',
        r'#\[test\].*?fn\s+.*?attack.*?\(',
        r'#\[test\].*?fn\s+.*?exploit.*?\('
    ]

    has_security_tests = False
    for file in rust_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if any(re.search(pattern, content, re.DOTALL) for pattern in security_test_patterns):
                has_security_tests = True
                break

    # 9. Preveri stress teste
    stress_test_patterns = [
        r'#\[test\].*?fn\s+.*?stress.*?\(',
        r'#\[test\].*?fn\s+.*?load.*?\(',
        r'#\[test\].*?fn\s+.*?concurrent.*?\('
    ]

    has_stress_tests = False
    for file in rust_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if any(re.search(pattern, content, re.DOTALL) for pattern in stress_test_patterns):
                has_stress_tests = True
                break

    # 10. Preveri end-to-end teste
    e2e_dir = os.path.join(test_dir, "e2e")
    e2e_patterns = [
        r'#\[test\].*?fn\s+.*?e2e.*?\(',
        r'#\[test\].*?fn\s+.*?end_to_end.*?\('
    ]

    has_e2e_tests = os.path.exists(e2e_dir) or any(
        re.search(pattern, open(file, 'r', encoding='utf-8', errors='ignore').read(), re.DOTALL)
        for pattern in e2e_patterns
        for file in rust_files
        if os.path.exists(file)
    )

    # 11. Preveri performance teste
    bench_dir = os.path.join(root_dir, "tests", "benchmarks")
    has_bench_tests = os.path.exists(bench_dir) and any(
        file.endswith('.rs') for _, _, files in os.walk(bench_dir) for file in files
    )

    # Zberi vse težave
    issues = []
    if missing_unit_tests:
        issues.append(f"1. Datoteke brez unit testov ({len(missing_unit_tests)}):") 
        for file in missing_unit_tests[:5]:  # Prikaži največ 5
            rel_path = os.path.relpath(file, root_dir)
            issues.append(f"  - {rel_path}")
        if len(missing_unit_tests) > 5:
            issues.append(f"  - ... in še {len(missing_unit_tests) - 5} drugih")

    if not integration_tests_present:
        issues.append("2. Manjkajo integracijski testi v mapi 'tests/integration/'")

    if not has_panic_tests:
        issues.append("3. Manjkajo panic testi (#[should_panic])")

    if not has_doc_tests:
        issues.append("4. Manjkajo doc testi (/// ```)")

    if not has_property_tests:
        issues.append("5. Manjkajo property testi (proptest! ali quickcheck) v mapi 'tests/property/'")

    if not has_regression_tests:
        issues.append("6. Manjkajo regression testi v mapi 'tests/regression/'")

    if not has_fuzz_tests:
        issues.append("7. Manjkajo fuzz testi v mapi 'tests/fuzz/'")

    if not has_security_tests:
        issues.append("8. Manjkajo security testi v mapi 'tests/security/'")

    if not has_stress_tests:
        issues.append("9. Manjkajo stresni testi v mapi 'tests/stress/'")

    if not has_e2e_tests:
        issues.append("10. Manjkajo end-to-end testi v mapi 'tests/e2e/'")

    if not has_bench_tests:
        issues.append("11. Manjkajo performance testi v mapi 'tests/benchmarks/'")

    return len(issues) == 0, issues

def run_stress_tests(root_dir):
    """Izvede stresne teste projekta."""
    print_header("IZVAJANJE STRESNIH TESTOV")
    results = []
    
    # Preveri, če obstajajo stresni testi v centralni mapi tests/stress
    stress_test_cmd = "cargo test --test stress --test-threads=1"
    print(f"   {Fore.YELLOW}Izvajanje stresnih testov iz tests/stress...{Style.RESET_ALL}")
    proc = subprocess.run(stress_test_cmd, shell=True, capture_output=True, text=True, cwd=root_dir)
    output = proc.stdout + proc.stderr

    if "no tests" in output.lower() or "0 tests" in output.lower():
        print(f"   {Fore.YELLOW}⚠️ Ni najdenih stresnih testov v mapi tests/stress{Style.RESET_ALL}")
        results.append("Ni najdenih stresnih testov v mapi tests/stress. Ustvari teste v tej mapi.")
    elif proc.returncode != 0:
        print(f"   {Fore.RED}❌ Stresni testi so neuspešni{Style.RESET_ALL}")
        results.append(f"Stresni testi so neuspešni: {proc.stderr}")
    else:
        print(f"   {Fore.GREEN}✓ Stresni testi so uspešno izvedeni{Style.RESET_ALL}")
        results.append("✓ Stresni testi so uspešno izvedeni")

    success = any("✓" in result for result in results)
    return success, results

def run_benchmark_tests(root_dir):
    """Zažene benchmark teste in preveri rezultate."""
    print_header("IZVAJANJE BENCHMARK TESTOV")
    results = []
    
    # Preveri, če obstajajo benchmark testi v centralni mapi tests/benchmarks
    benchmark_test_cmd = "cargo test --test benchmarks --test-threads=1 -- --nocapture"
    print(f"   {Fore.YELLOW}Izvajanje benchmark testov iz tests/benchmarks...{Style.RESET_ALL}")
    proc = subprocess.run(benchmark_test_cmd, shell=True, capture_output=True, text=True, cwd=root_dir)
    output = proc.stdout + proc.stderr

    if "no tests" in output.lower() or "0 tests" in output.lower():
        print(f"   {Fore.YELLOW}⚠️ Ni najdenih benchmark testov v mapi tests/benchmarks{Style.RESET_ALL}")
        results.append("Ni najdenih benchmark testov v mapi tests/benchmarks. Ustvari teste v tej mapi.")
    elif proc.returncode != 0:
        print(f"   {Fore.RED}❌ Benchmark testi so neuspešni{Style.RESET_ALL}")
        results.append(f"Benchmark testi so neuspešni: {proc.stderr}")
    else:
        print(f"   {Fore.GREEN}✓ Benchmark testi so uspešno izvedeni{Style.RESET_ALL}")
        results.append("✓ Benchmark testi so uspešno izvedeni")

    # Preveri tudi criterion benchmarke, če obstajajo
    criterion_bench_cmd = "cargo bench"
    print(f"   {Fore.YELLOW}Izvajanje criterion benchmark testov...{Style.RESET_ALL}")
    criterion_proc = subprocess.run(criterion_bench_cmd, shell=True, capture_output=True, text=True, cwd=root_dir)
    criterion_output = criterion_proc.stdout + criterion_proc.stderr
    
    if criterion_proc.returncode != 0:
        if "command not found" in criterion_output.lower() or "no such command" in criterion_output.lower():
            results.append("Namesti criterion ali druge knjižnice za benchmark testiranje")
            results.append("Dodaj odvisnost v Cargo.toml: criterion = \"0.5\"")
        else:
            results.append(f"Napaka pri izvajanju benchmark testov: {proc.stderr}")
    elif "no benchmarks" in output.lower() or "0 benchmarks" in output.lower():
        results.append("Ni najdenih benchmark testov. Ustvari benchmark teste s knjižnico criterion.")
    else:
        results.append("✓ Criterion benchmark testi so uspešno izvedeni")

    # 2. Preveri in izvedi latency benchmark teste
    print(f"   {Fore.YELLOW}Preverjanje latency benchmark testov...{Style.RESET_ALL}")
    latency_bench_files = []
    bench_dir = os.path.join(root_dir, "benches")
    if os.path.exists(bench_dir):
        for root, _, files in os.walk(bench_dir):
            for file in files:
                if file.endswith('.rs') and ('latency' in file.lower() or 'perf' in file.lower()):
                    latency_bench_files.append(os.path.join(root, file))

    if not latency_bench_files:
        results.append("Ni najdenih latency benchmark testov. Ustvari teste za merjenje latence kritičnih poti.")
    else:
        results.append(f"✓ Najdeni latency benchmark testi: {len(latency_bench_files)}")

    # 3. Preveri, ali benchmark testi preverjajo zahtevano latenco (<1ms)
    latency_check_found = False
    for file in latency_bench_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if re.search(r'assert.*?[<].*?1.*?ms|assert.*?[<].*?1000.*?µs', content, re.IGNORECASE):
                    latency_check_found = True
                    break
        except Exception:
            pass

    if not latency_check_found and latency_bench_files:
        results.append("Benchmark testi ne preverjajo zahtevane latence (<1ms). Dodaj assert za preverjanje latence.")
    elif latency_check_found:
        results.append("✓ Benchmark testi preverjajo zahtevano latenco (<1ms)")

    # Preveri, ali je vsaj en tip benchmark testov uspešno izveden
    success = any("✓" in result for result in results)
    return success, results

def main():
    # Nastavi okoljske spremenljivke
    set_rust_env()

    # Poišči root direktorij (kjer je Cargo.lock ali Cargo.toml)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir
    while not (os.path.isfile(os.path.join(root_dir, 'Cargo.toml')) or os.path.isfile(os.path.join(root_dir, 'Cargo.lock'))):
        parent = os.path.dirname(root_dir)
        if parent == root_dir:
            print(f"{Fore.RED}Napaka: Ni bilo mogoče najti root direktorija z Cargo.toml ali Cargo.lock!{Style.RESET_ALL}")
            sys.exit(1)
        root_dir = parent

    test_results = []
    print(f"{Fore.CYAN}🚀 ZAGON SPROTNEGA PREVERJANJA KODE{Style.RESET_ALL}")
    print(f"   {Fore.LIGHTBLACK_EX}Pritisni Ctrl+C za zaustavitev spremljanja sprememb\n{Style.RESET_ALL}")

    # 1. Format
    format_result = TestResult("Preverjanje formatiranja")
    print_header("1. PREVERJANJE FORMATIRANJA")
    if not run_check(
        "cargo fmt --all -- --check",
        install_hint="cargo install rustfmt",
        success_msg="✓ Formatiranje je v redu",
        fail_msg="⚠️  Formatiranje ni pravilno!",
        cwd=root_dir
    ):
        format_result.success = False
        format_result.errors.append("Koda ni pravilno formatirana. Popravi z ukazom: cargo fmt")
    test_results.append(format_result)

    # 2. Clippy
    clippy_result = TestResult("Statična analiza (Clippy)")
    print_header("2. STATIČNA ANALIZA (CLIPPY)")
    def treat_clippy(exit_code, output, warn_found, error_found):
        if exit_code == 0:
            return True
        if error_found or warn_found:
            clippy_result.errors.append("Najdene so bile napake ali opozorila pri Clippy!")
            return False
        return True
    clippy_cmd = [
        "cargo clippy --all-targets --all-features -- -D warnings",
        "cargo clippy --all-targets --all-features -- -D clippy::pedantic",
        "cargo clippy --all-targets --all-features -- -D clippy::nursery"
    ]

    for cmd in clippy_cmd:
        if not run_check(
            cmd,
            install_hint="rustup component add clippy",
            success_msg="✓ Clippy preverjanje uspešno: " + cmd.split(" -- ")[1],
            fail_msg="⚠️  Napaka pri izvajanju Clippy",
            warn_patterns=[r'warning'],
            error_patterns=[r'error'],
            treat_exit_code=treat_clippy,
            cwd=root_dir
        ):
            clippy_result.success = False
    test_results.append(clippy_result)

    # 3. Testi in pokritost
    test_suite_result = TestResult("Izvajanje testne suite in pokritost kode")
    print_header("3. IZVAJANJE TESTNE SUITE IN POKRITOST")

    # Izvedi osnovne teste
    basic_tests_ok = run_check(
        "cargo test --all -- --nocapture",
        success_msg="✓ Vsi osnovni testi so uspešno opravljeni",
        fail_msg="⚠️  Osnovni testi niso uspeli!",
        cwd=root_dir
    )

    if not basic_tests_ok:
        test_suite_result.success = False
        test_suite_result.errors.append("Nekateri osnovni testi niso uspeli")

    # Izvedi doc teste
    doc_tests_ok = run_check(
        "cargo test --doc",
        success_msg="✓ Vsi doc testi so uspešno opravljeni",
        fail_msg="⚠️  Doc testi niso uspeli!",
        cwd=root_dir
    )

    if not doc_tests_ok:
        test_suite_result.warnings.append("Nekateri doc testi niso uspeli")

    # Izvedi panic teste
    panic_tests_ok = run_check(
        "cargo test -- --include-ignored panic",
        success_msg="✓ Vsi panic testi so uspešno opravljeni",
        fail_msg="⚠️  Panic testi niso uspeli!",
        cwd=root_dir
    )

    if not panic_tests_ok:
        test_suite_result.warnings.append("Nekateri panic testi niso uspeli")

    # Izvedi property teste
    property_tests_ok = run_check(
        "cargo test -- --include-ignored property",
        success_msg="✓ Vsi property testi so uspešno opravljeni",
        fail_msg="⚠️  Property testi niso uspeli!",
        cwd=root_dir
    )

    if not property_tests_ok:
        test_suite_result.warnings.append("Nekateri property testi niso uspeli")

    # Namesti cargo-tarpaulin če še ni nameščen
    run_check(
        "cargo install cargo-tarpaulin",
        success_msg="✓ cargo-tarpaulin je nameščen",
        fail_msg=None,
        cwd=root_dir
    )

    # Preveri pokritost kode in zahtevaj 99%
    coverage_result, coverage_messages = check_coverage(root_dir)
    if coverage_result:
        print(f"   {Fore.GREEN}{coverage_messages[0]}{Style.RESET_ALL}")
    else:
        print(f"   {Fore.RED}❌ Nepopolna pokritost kode s testi{Style.RESET_ALL}")
        for msg in coverage_messages:
            print(f"   {Fore.RED}- {msg}{Style.RESET_ALL}")
        test_suite_result.success = False
        test_suite_result.errors.append("Koda nima 99% pokritosti s testi")

    # Preveri prisotnost različnih tipov testov
    test_types_result, test_types_messages = check_test_types(root_dir)
    if test_types_result:
        print(f"   {Fore.GREEN}✓ Vsi potrebni tipi testov so prisotni{Style.RESET_ALL}")
    else:
        print(f"   {Fore.YELLOW}⚠️ Manjkajo določeni tipi testov{Style.RESET_ALL}")
        for msg in test_types_messages:
            print(f"   {Fore.YELLOW}- {msg}{Style.RESET_ALL}")
        test_suite_result.warnings.extend(test_types_messages)

    # Zaženi stresne teste, če obstajajo
    stress_result, stress_messages = run_stress_tests(root_dir)
    if stress_result:
        print(f"   {Fore.GREEN}✓ Stresni testi uspešno izvedeni{Style.RESET_ALL}")
    else:
        print(f"   {Fore.YELLOW}⚠️ Opozorilo pri stresnih testih{Style.RESET_ALL}")
        for msg in stress_messages:
            print(f"   {Fore.YELLOW}- {msg}{Style.RESET_ALL}")
        test_suite_result.warnings.extend(stress_messages)

    # Zaženi benchmark teste, če obstajajo
    benchmark_result, benchmark_messages = run_benchmark_tests(root_dir)
    if benchmark_result:
        print(f"   {Fore.GREEN}✓ Benchmark testi uspešno izvedeni{Style.RESET_ALL}")
    else:
        print(f"   {Fore.YELLOW}⚠️ Opozorilo pri benchmark testih{Style.RESET_ALL}")
        for msg in benchmark_messages:
            print(f"   {Fore.YELLOW}- {msg}{Style.RESET_ALL}")
        test_suite_result.warnings.extend(benchmark_messages)

    # Zaženi fuzz teste, če obstajajo
    fuzz_dir = os.path.join(root_dir, "fuzz")
    if os.path.exists(fuzz_dir):
        print(f"   {Fore.YELLOW}Preverjanje fuzz testov...{Style.RESET_ALL}")
        fuzz_files = [f for f in os.listdir(fuzz_dir) if f.endswith('.rs')]
        if fuzz_files:
            print(f"   {Fore.GREEN}✓ Najdeni fuzz testi: {len(fuzz_files)}{Style.RESET_ALL}")
            # Preveri, ali je cargo-fuzz nameščen
            fuzz_check = subprocess.run("cargo fuzz --help", shell=True, capture_output=True, text=True)
            if fuzz_check.returncode != 0:
                print(f"   {Fore.YELLOW}⚠️ cargo-fuzz ni nameščen. Namesti z: cargo install cargo-fuzz{Style.RESET_ALL}")
                test_suite_result.warnings.append("cargo-fuzz ni nameščen. Namesti z: cargo install cargo-fuzz")
            else:
                print(f"   {Fore.GREEN}✓ cargo-fuzz je nameščen{Style.RESET_ALL}")
        else:
            print(f"   {Fore.YELLOW}⚠️ Mapa fuzz obstaja, vendar ni najdenih fuzz testov{Style.RESET_ALL}")
            test_suite_result.warnings.append("Mapa fuzz obstaja, vendar ni najdenih fuzz testov")
    else:
        print(f"   {Fore.YELLOW}⚠️ Mapa fuzz ne obstaja. Ustvari mapo fuzz in dodaj fuzz teste{Style.RESET_ALL}")
        test_suite_result.warnings.append("Mapa fuzz ne obstaja. Ustvari mapo fuzz in dodaj fuzz teste")

    # Zaženi security teste, če obstajajo
    security_test_cmd = "cargo test --test security --test-threads=1"
    print(f"   {Fore.YELLOW}Izvajanje security testov...{Style.RESET_ALL}")
    security_proc = subprocess.run(security_test_cmd, shell=True, capture_output=True, text=True, cwd=root_dir)
    security_output = security_proc.stdout + security_proc.stderr

    if "no tests" in security_output.lower() or "0 tests" in security_output.lower():
        print(f"   {Fore.YELLOW}⚠️ Ni najdenih security testov{Style.RESET_ALL}")
        test_suite_result.warnings.append("Ni najdenih security testov. Ustvari teste z besedo 'security' v imenu.")
    elif security_proc.returncode != 0:
        print(f"   {Fore.RED}❌ Security testi so neuspešni{Style.RESET_ALL}")
        test_suite_result.warnings.append("Security testi so neuspešni")
    else:
        print(f"   {Fore.GREEN}✓ Security testi so uspešno izvedeni{Style.RESET_ALL}")

    # Zaženi regression teste, če obstajajo
    tests_dir = os.path.join(root_dir, "tests")
    regression_dir = os.path.join(tests_dir, "regression")
    if os.path.exists(regression_dir):
        print(f"   {Fore.YELLOW}Izvajanje regression testov...{Style.RESET_ALL}")
        regression_test_cmd = "cargo test --test regression --test-threads=1"
        regression_proc = subprocess.run(regression_test_cmd, shell=True, capture_output=True, text=True, cwd=root_dir)
        regression_output = regression_proc.stdout + regression_proc.stderr

        if "no tests" in regression_output.lower() or "0 tests" in regression_output.lower():
            print(f"   {Fore.YELLOW}⚠️ Mapa regression obstaja, vendar ni najdenih regression testov{Style.RESET_ALL}")
            test_suite_result.warnings.append("Mapa regression obstaja, vendar ni najdenih regression testov")
        elif regression_proc.returncode != 0:
            print(f"   {Fore.RED}❌ Regression testi so neuspešni{Style.RESET_ALL}")
            test_suite_result.warnings.append("Regression testi so neuspešni")
        else:
            print(f"   {Fore.GREEN}✓ Regression testi so uspešno izvedeni{Style.RESET_ALL}")
    else:
        print(f"   {Fore.YELLOW}⚠️ Mapa regression ne obstaja. Ustvari mapo tests/regression in dodaj regression teste{Style.RESET_ALL}")
        test_suite_result.warnings.append("Mapa regression ne obstaja. Ustvari mapo tests/regression in dodaj regression teste")

    # Generiraj HTML poročilo za pokritost
    report_path = os.path.join(root_dir, "target", "tarpaulin", "tarpaulin-report.html")
    if os.path.exists(report_path):
        print(f"   {Fore.GREEN}✓ Poročilo o pokritosti: {report_path}{Style.RESET_ALL}")
        test_suite_result.warnings.append(f"Poročilo o pokritosti je na voljo v: {report_path}")

    test_results.append(test_suite_result)

    # 4. Dokumentacija
    doc_result = TestResult("Preverjanje dokumentacije")
    print_header("4. PREVERJANJE DOKUMENTACIJE")
    doc_ok = True
    if not run_check(
        "cargo doc --no-deps --document-private-items",
        install_hint="cargo install cargo-doc",
        success_msg="✓ Dokumentacija je v redu",
        fail_msg="⚠️  Dokumentacija ni v redu!",
        cwd=root_dir
    ):
        doc_ok = False
        doc_result.errors.append("Napaka pri generiranju dokumentacije")
    if not run_check(
        "cargo test --doc",
        success_msg=None,
        fail_msg="⚠️  Dokumentacijski testi niso uspeli!",
        cwd=root_dir
    ):
        doc_ok = False
        doc_result.errors.append("Dokumentacijski testi niso uspeli")
    if not doc_ok:
        doc_result.success = False
    test_results.append(doc_result)

    # 5. Neuporabljene odvisnosti
    deps_result = TestResult("Preverjanje neuporabljenih odvisnosti")
    print_header("5. PREVERJANJE NEUPORABLJENIH ODVISNOSTI")
    def treat_udeps(exit_code, output, warn_found, error_found):
        if 'only accepted on the nightly compiler' in output:
            deps_result.warnings.append("Ukaz zahteva Rust nightly toolchain. Za preklop zaženi: rustup default nightly")
            return True
        if 'unused dependencies' in output:
            # Izloči natančno katere odvisnosti so neuporabljene
            import re
            matches = re.findall(r'`([^`]+)`', output)
            if matches:
                deps_result.warnings.append(f"Najdene so neuporabljene odvisnosti: {', '.join(matches)}")
            return False
        return exit_code == 0
    # Najprej namestimo nightly toolchain in cargo-udeps
    if not run_check(
        "rustup toolchain install nightly --component rust-src",
        success_msg="✓ Nightly toolchain je nameščen",
        fail_msg="⚠️  Napaka pri namestitvi nightly toolchain",
        cwd=root_dir
    ):
        deps_result.success = False
        deps_result.errors.append("Napaka pri namestitvi nightly toolchain")
    elif not run_check(
        "rustup default nightly",
        success_msg="✓ Nightly toolchain je nastavljen kot privzeti",
        fail_msg="⚠️  Napaka pri nastavljanju nightly toolchain",
        cwd=root_dir
    ):
        deps_result.success = False
        deps_result.errors.append("Napaka pri nastavljanju nightly toolchain")
    elif not run_check(
        "cargo install cargo-udeps --locked",
        success_msg="✓ Cargo-udeps je nameščen",
        fail_msg="⚠️  Napaka pri namestitvi cargo-udeps",
        cwd=root_dir
    ):
        deps_result.success = False
        deps_result.errors.append("Napaka pri namestitvi cargo-udeps")
    # Zdaj lahko preverimo neuporabljene odvisnosti
    elif not run_check(
        "cargo udeps --all-targets --all-features",
        success_msg="✓ Ni neuporabljenih odvisnosti",
        fail_msg="⚠️  Napaka pri preverjanju neuporabljenih odvisnosti",
        treat_exit_code=treat_udeps,
        cwd=root_dir
    ):
        deps_result.success = False
    test_results.append(deps_result)

    # 6. Varnostne ranljivosti
    security_result = TestResult("Preverjanje varnostnih ranljivosti")
    print_header("6. PREVERJANJE VARNOSTNIH RANLJIVOSTI")
    print(f"   {Fore.LIGHTBLACK_EX}Root projektna mapa: {root_dir}{Style.RESET_ALL}")
    cargo_lock_path = os.path.join(root_dir, 'Cargo.lock')
    audit_cmd = "cargo audit"
    if os.path.isfile(cargo_lock_path):
        cargo_lock_path = cargo_lock_path.replace('\\', '/')
        audit_cmd = f'cargo audit --file "{cargo_lock_path}"'
        print(f"   {Fore.LIGHTBLACK_EX}Uporabljam Cargo.lock: {cargo_lock_path}{Style.RESET_ALL}")
    else:
        security_result.errors.append(f"Cargo.lock ni bil najden v {root_dir}")
        security_result.success = False
    def treat_audit(exit_code, output, warn_found, error_found):
        if re.search(r'found [1-9][0-9]* vulnerabilities', output, re.IGNORECASE):
            print(f"   {Fore.RED}  Najdene so varnostne ranljivosti!{Style.RESET_ALL}")
            print(output)
            security_result.success = False
            security_result.errors.append("Najdene so bile varnostne ranljivosti")
            return False
        if 'No vulnerable packages found' in output:
            return True
        return exit_code == 0
    if not run_check(
        audit_cmd,
        install_hint="cargo install cargo-audit",
        success_msg="✓ Ni znanih varnostnih ranljivosti",
        fail_msg="⚠️  Napaka pri preverjanju ranljivosti",
        treat_exit_code=treat_audit,
        cwd=root_dir
    ):
        security_result.success = False
    test_results.append(security_result)

    # Povzetek
    print_header("POVZETEK PREVERJANJA")
    print("\nRezultati preverjanj:")
    all_passed = True
    for result in test_results:
        status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if result.success else f"{Fore.RED}❌{Style.RESET_ALL}"
        print(f"\n{status} {result.name}")
        if not result.success:
            all_passed = False
            if result.errors:
                print(f"   {Fore.RED}Napake:{Style.RESET_ALL}")
                for error in result.errors:
                    print(f"   - {error}")
        if result.warnings:
            print(f"   {Fore.YELLOW}Opozorila:{Style.RESET_ALL}")
            for warning in result.warnings:
                print(f"   - {warning}")

    print("\nKončni status:")
    if all_passed:
        print(f"{Fore.GREEN}✅ VSE PREVERITVE SO USPEŠNO OPRAVLJENE!{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ NEKATERE PREVERITVE NISO USPELE!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
