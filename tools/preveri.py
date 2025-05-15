#!/usr/bin/env python3
import subprocess
import sys
import os
import re
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

def main():
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
    if not run_check(
        "cargo clippy --all-targets --all-features -- -D warnings",
        install_hint="rustup component add clippy",
        success_msg="✓ Vsa Clippy preverjanja so uspela",
        fail_msg="⚠️  Napaka pri izvajanju Clippy",
        warn_patterns=[r'warning'],
        error_patterns=[r'error'],
        treat_exit_code=treat_clippy,
        cwd=root_dir
    ):
        clippy_result.success = False
    test_results.append(clippy_result)

    # 3. Testi
    test_suite_result = TestResult("Izvajanje testne suite")
    print_header("3. IZVAJANJE TESTNE SUITE")
    if not run_check(
        "cargo test --all -- --nocapture",
        success_msg="✓ Vsi testi so uspešno opravljeni",
        fail_msg="⚠️  Testi niso uspeli!",
        cwd=root_dir
    ):
        test_suite_result.success = False
        test_suite_result.errors.append("Nekateri testi niso uspeli")
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
    if not run_check(
        "cargo udeps --all-targets --all-features",
        install_hint="cargo install cargo-udeps",
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
