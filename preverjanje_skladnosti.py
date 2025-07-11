import os
import re
import sys
from collections import defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.syntax import Syntax
except ImportError:
    print("Napaka: Knjižnica 'rich' ni nameščena. Prosim, namestite jo z ukazom: pip install rich")
    sys.exit(1)

# Pravila, enaka kot prej
PRAVILA = [
    (re.compile(r'\.unwrap\(\)'), "Prepovedana uporaba '.unwrap()'. Uporabi '?' operator."),
    (re.compile(r'\.expect\('), "Prepovedana uporaba '.expect()'. Uporabi '?' operator."),
    (re.compile(r'panic!\('), "Prepovedana uporaba 'panic!()'. Vrneš 'Err(anyhow!(\"...\"))'."),
    (re.compile(r'todo!\('), "Prepovedana uporaba 'todo!()'."),
    (re.compile(r'unimplemented!\('), "Prepovedana uporaba 'unimplemented!()'."),
    (re.compile(r'\b(f32|f64)\b'), "Prepovedana uporaba 'f32'/'f64'. Uporabi 'rust_decimal::Decimal'."),
    (re.compile(r'std::sync::Mutex'), "Neustrezen Mutex. V async kodi uporabi 'tokio::sync::Mutex'."),
    (re.compile(r'Vec::new\(\)'), "Neoptimalno. Uporabi 'Vec::with_capacity(N)'."),
    (re.compile(r'String::new\(\)'), "Neoptimalno. Uporabi 'String::with_capacity(N)'."),
    (re.compile(r'"[^"]*\\'), "Trdo kodirana pot. Uporabi 'std::path::PathBuf'.") # Išče dvojno poševnico
]

def preveri_skladnost(root_dir):
    """Pregleda vse .rs datoteke in izpiše rezultate v moderni tabeli."""
    console = Console(force_terminal=True, color_system="truecolor")
    console.print(Panel(Text(f"Pregledujem: {root_dir}", justify="center"), title="[bold cyan]TallyIO Skladnostni Preizkus[/bold cyan]", border_style="blue"))

    krsitve = defaultdict(list)
    stevilo_pregledanih_datotek = 0
    exclude_dirs = {os.path.normpath(os.path.join(root_dir, 'target'))}

    for subdir, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if os.path.join(subdir, d) not in exclude_dirs and d != '.git']
        for file in files:
            if file.endswith('.rs'):
                stevilo_pregledanih_datotek += 1
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    is_test_file = "#[cfg(test)]" in content or "/tests/" in file_path.replace('\\', '/')
                    is_build_file = file.endswith('build.rs')

                    for i, line in enumerate(content.splitlines(), 1):
                        if line.strip().startswith('//'):
                            continue
                        
                        for pravilo, sporocilo in PRAVILA:
                            if "f32" in sporocilo or "f64" in sporocilo:
                                if is_test_file:
                                    continue

                            # Izjema za unwrap/expect v test kodi
                            if ("unwrap" in sporocilo or "expect" in sporocilo) and is_test_file:
                                continue

                            # Izjema za trdo kodirane poti v build.rs datotekah
                            if "Trdo kodirana pot" in sporocilo and is_build_file:
                                continue

                            if pravilo.search(line):
                                krsitve[os.path.relpath(file_path, root_dir)].append({
                                    "vrstica": i,
                                    "koda": line.strip(),
                                    "problem": sporocilo
                                })
                except Exception as e:
                    console.print(f"[NAPAKA] Pri branju datoteke {file_path}: {e}")

    total_krsitev = sum(len(v) for v in krsitve.values())

    if total_krsitev == 0:
        msg = Text(f"✔ Pregledanih {stevilo_pregledanih_datotek} datotek. Nobena kršitev ni bila najdena.", justify="center")
        console.print(Panel(msg, title="[bold green]USPEH[/bold green]", border_style="green"))
        return 0

    msg = Text(f"❌ Najdenih {total_krsitev} kršitev v {len(krsitve)} datotekah.", justify="center")
    console.print(Panel(msg, title="[bold red]NEUSPEH[/bold red]", border_style="red"))

    table = Table(title="[bold]Podroben Seznam Kršitev[/bold]", border_style="red", show_header=True, header_style="bold magenta")
    table.add_column("Datoteka", style="cyan", no_wrap=True, width=40)
    table.add_column("Vrstica", style="magenta", width=7)
    table.add_column("Problem", style="red", width=40)
    table.add_column("Koda", style="yellow")

    for datoteka, seznam_krsitev in sorted(krsitve.items()):
        for i, krsitev in enumerate(seznam_krsitev):
            syntax = Syntax(krsitev['koda'], "rust", theme="monokai", line_numbers=False)
            if i == 0:
                table.add_row(datoteka, str(krsitev['vrstica']), krsitev['problem'], syntax)
            else:
                table.add_row("", str(krsitev['vrstica']), krsitev['problem'], syntax)
        if len(krsitve) > 1:
            table.add_row(end_section=True)

    console.print(table)

    # Shrani poročilo v datoteko za zanesljiv pregled
    try:
        with open("porocilo_o_skladnosti.txt", "w", encoding="utf-8") as f:
            f.write(f"TallyIO Skladnostni Preizkus - Poročilo\n")
            f.write(f"='*'*50\n")
            f.write(f"Najdenih {total_krsitev} kršitev v {len(krsitve)} datotekah.\n\n")
            for datoteka, seznam_krsitev in sorted(krsitve.items()):
                f.write(f"--- Datoteka: {datoteka} ---\n")
                for krsitev in seznam_krsitev:
                    f.write(f"  - Vrstica {krsitev['vrstica']}: {krsitev['problem']}\n")
                    f.write(f"    | Koda: {krsitev['koda']}\n")
                f.write("\n")
        console.print(Panel("Poročilo je bilo shranjeno v [bold]porocilo_o_skladnosti.txt[/bold]", style="green"))
    except Exception as e:
        console.print(Panel(f"Napaka pri shranjevanju poročila: {e}", style="red"))


    return total_krsitev

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    if preveri_skladnost(project_root) > 0:
        sys.exit(1)
    else:
        sys.exit(0)
