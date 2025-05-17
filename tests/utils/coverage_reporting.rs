//! Ogrodje za sledenje pokritosti kode za TallyIO platformo
//!
//! Ta modul implementira izboljšano sledenje pokritosti kode, ki deluje tudi 
//! v primerih, kjer standardna orodja kot je tarpaulin ne delujejo optimalno,
//! npr. pri asinhronih metodah ali kompleksnih kontrolnih tokovih.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use once_cell::sync::Lazy;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Globalna tabela za sledenje pokritosti
pub static COVERAGE_TRACKER: Lazy<Mutex<CoverageTracker>> = 
    Lazy::new(|| Mutex::new(CoverageTracker::new()));

/// Makro za sledenje izvedbe kode
#[macro_export]
macro_rules! track_line {
    ($module:expr, $line:expr) => {
        if cfg!(feature = "track-coverage") {
            $crate::utils::coverage_reporting::COVERAGE_TRACKER
                .lock()
                .unwrap()
                .track($module, $line);
        }
    };
}

/// Struktura za sledenje pokritosti kode
#[derive(Debug)]
pub struct CoverageTracker {
    /// Preslikava iz modulov in vrstic v števce izvedb
    counters: HashMap<ModuleLine, AtomicUsize>,
    /// Skupno število sledenih linij
    total_lines: usize,
    /// Število pokritih linij
    covered_lines: usize,
}

/// Modul in številka vrstice za sledenje
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct ModuleLine {
    /// Ime modula
    pub module: String,
    /// Oznaka vrstice za sledenje
    pub line: String,
}

impl fmt::Display for ModuleLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.module, self.line)
    }
}

impl CoverageTracker {
    /// Ustvari nov sledilnik pokritosti
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            total_lines: 0,
            covered_lines: 0,
        }
    }
    
    /// Registriraj linijo za sledenje
    pub fn register(&mut self, module: &str, line: &str) {
        let key = ModuleLine {
            module: module.to_string(),
            line: line.to_string(),
        };
        
        if !self.counters.contains_key(&key) {
            self.counters.insert(key, AtomicUsize::new(0));
            self.total_lines += 1;
        }
    }
    
    /// Zabeleži izvedbo kode
    pub fn track(&mut self, module: &str, line: &str) {
        let key = ModuleLine {
            module: module.to_string(),
            line: line.to_string(),
        };
        
        if let Some(counter) = self.counters.get(&key) {
            counter.fetch_add(1, Ordering::SeqCst);
            // Posodobi število pokritih linij, če je to prvi klic
            if counter.load(Ordering::SeqCst) == 1 {
                self.covered_lines += 1;
            }
        } else {
            // Registriraj in zabeleži, če linija še ni registrirana
            self.register(module, line);
            if let Some(counter) = self.counters.get(&key) {
                counter.fetch_add(1, Ordering::SeqCst);
                self.covered_lines += 1;
            }
        }
    }
    
    /// Vrne število izvedb določene linije
    pub fn get_count(&self, module: &str, line: &str) -> usize {
        let key = ModuleLine {
            module: module.to_string(),
            line: line.to_string(),
        };
        
        self.counters.get(&key)
            .map(|counter| counter.load(Ordering::SeqCst))
            .unwrap_or(0)
    }
    
    /// Vrne odstotek pokritosti
    pub fn coverage_percentage(&self) -> f64 {
        if self.total_lines == 0 {
            return 0.0;
        }
        
        (self.covered_lines as f64 / self.total_lines as f64) * 100.0
    }
    
    /// Vrne vse sledene vrstice s števci
    pub fn get_all_counters(&self) -> Vec<(ModuleLine, usize)> {
        self.counters
            .iter()
            .map(|(key, counter)| (key.clone(), counter.load(Ordering::SeqCst)))
            .collect()
    }
    
    /// Vrne nepokrite vrstice
    pub fn get_uncovered_lines(&self) -> Vec<ModuleLine> {
        self.counters
            .iter()
            .filter(|(_, counter)| counter.load(Ordering::SeqCst) == 0)
            .map(|(key, _)| key.clone())
            .collect()
    }
    
    /// Ponastavi vse števce
    pub fn reset(&mut self) {
        for counter in self.counters.values() {
            counter.store(0, Ordering::SeqCst);
        }
        self.covered_lines = 0;
    }
    
    /// Izpiše statistiko pokritosti
    pub fn print_statistics(&self) {
        println!("=== Code Coverage Statistics ===");
        println!("Total lines tracked: {}", self.total_lines);
        println!("Covered lines: {}", self.covered_lines);
        println!("Coverage percentage: {:.2}%", self.coverage_percentage());
        
        println!("\nModule Coverage:");
        
        // Izračun pokritosti po modulih
        let mut module_stats: HashMap<String, (usize, usize)> = HashMap::new();
        
        for (key, counter) in &self.counters {
            let count = counter.load(Ordering::SeqCst);
            let entry = module_stats.entry(key.module.clone()).or_insert((0, 0));
            
            // Povečaj skupno število linij
            entry.1 += 1;
            
            // Povečaj število pokritih linij, če je linija pokrita
            if count > 0 {
                entry.0 += 1;
            }
        }
        
        // Izpiši statistiko po modulih
        for (module, (covered, total)) in module_stats {
            let percentage = if total > 0 {
                (covered as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            
            println!("  {}: {}/{} lines covered ({:.2}%)", 
                     module, covered, total, percentage);
        }
        
        println!("\nUncovered Lines:");
        
        for line in self.get_uncovered_lines() {
            println!("  {}", line);
        }
        
        println!("===============================");
    }
    
    /// Generira HTML poročilo o pokritosti
    pub fn generate_html_report<P: AsRef<Path>>(&self, output_path: P) -> std::io::Result<()> {
        let mut file = File::create(output_path)?;
        
        // Osnovni HTML
        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html lang=\"en\">")?;
        writeln!(file, "<head>")?;
        writeln!(file, "  <meta charset=\"UTF-8\">")?;
        writeln!(file, "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")?;
        writeln!(file, "  <title>TallyIO Coverage Report</title>")?;
        writeln!(file, "  <style>")?;
        writeln!(file, "    body {{ font-family: Arial, sans-serif; margin: 20px; }}")?;
        writeln!(file, "    h1 {{ color: #333; }}")?;
        writeln!(file, "    .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}")?;
        writeln!(file, "    .progress {{ height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }}")?;
        writeln!(file, "    .progress-bar {{ height: 100%; background-color: #4caf50; text-align: center; color: white; line-height: 20px; }}")?;
        writeln!(file, "    .module {{ margin-bottom: 30px; }}")?;
        writeln!(file, "    .module-name {{ font-weight: bold; margin-bottom: 10px; }}")?;
        writeln!(file, "    table {{ width: 100%; border-collapse: collapse; }}")?;
        writeln!(file, "    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}")?;
        writeln!(file, "    tr:nth-child(even) {{ background-color: #f2f2f2; }}")?;
        writeln!(file, "    th {{ background-color: #4caf50; color: white; }}")?;
        writeln!(file, "    .uncovered {{ background-color: #ffcccc; }}")?;
        writeln!(file, "  </style>")?;
        writeln!(file, "</head>")?;
        writeln!(file, "<body>")?;
        
        // Glava in povzetek
        writeln!(file, "  <h1>TallyIO Coverage Report</h1>")?;
        writeln!(file, "  <div class=\"summary\">")?;
        writeln!(file, "    <h2>Coverage Summary</h2>")?;
        writeln!(file, "    <p>Total lines tracked: {}</p>", self.total_lines)?;
        writeln!(file, "    <p>Covered lines: {}</p>", self.covered_lines)?;
        writeln!(file, "    <div class=\"progress\">")?;
        writeln!(file, "      <div class=\"progress-bar\" style=\"width: {:.2}%\">{:.2}%</div>", 
                 self.coverage_percentage(), self.coverage_percentage())?;
        writeln!(file, "    </div>")?;
        writeln!(file, "  </div>")?;
        
        // Statistika po modulih
        writeln!(file, "  <h2>Module Coverage</h2>")?;
        
        let mut module_lines: HashMap<String, Vec<(String, usize)>> = HashMap::new();
        
        for (key, counter) in &self.counters {
            let count = counter.load(Ordering::SeqCst);
            let entry = module_lines.entry(key.module.clone()).or_insert_with(Vec::new);
            
            entry.push((key.line.clone(), count));
        }
        
        for (module, lines) in &module_lines {
            let total = lines.len();
            let covered = lines.iter().filter(|(_, count)| *count > 0).count();
            let percentage = if total > 0 {
                (covered as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            
            writeln!(file, "  <div class=\"module\">")?;
            writeln!(file, "    <div class=\"module-name\">{}</div>", module)?;
            writeln!(file, "    <p>{}/{} lines covered ({:.2}%)</p>", covered, total, percentage)?;
            writeln!(file, "    <div class=\"progress\">")?;
            writeln!(file, "      <div class=\"progress-bar\" style=\"width: {:.2}%\">{:.2}%</div>", 
                     percentage, percentage)?;
            writeln!(file, "    </div>")?;
            
            writeln!(file, "    <table>")?;
            writeln!(file, "      <tr><th>Line</th><th>Execution Count</th></tr>")?;
            
            // Sortiraj vrstice za lažje branje
            let mut sorted_lines = lines.clone();
            sorted_lines.sort_by(|a, b| a.0.cmp(&b.0));
            
            for (line, count) in sorted_lines {
                let class = if count == 0 { " class=\"uncovered\"" } else { "" };
                writeln!(file, "      <tr{}><td>{}</td><td>{}</td></tr>", class, line, count)?;
            }
            
            writeln!(file, "    </table>")?;
            writeln!(file, "  </div>")?;
        }
        
        // Zaključek
        writeln!(file, "  <h2>Uncovered Lines</h2>")?;
        writeln!(file, "  <ul>")?;
        
        for line in self.get_uncovered_lines() {
            writeln!(file, "    <li>{}</li>", line)?;
        }
        
        writeln!(file, "  </ul>")?;
        writeln!(file, "</body>")?;
        writeln!(file, "</html>")?;
        
        Ok(())
    }
}

/// Pomožna struktura za zbiranje pokritosti testov
pub struct TestCoverageCollector {
    /// Ime tega kolektorja (za identifikacijo)
    name: String,
    /// Ali naj izpiše rezultate ob zaključku
    print_on_drop: bool,
    /// Ali naj ustvari HTML poročilo
    generate_html: bool,
    /// Pot za HTML poročilo
    html_path: Option<String>,
}

impl TestCoverageCollector {
    /// Ustvari nov kolektor za pokritost
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            print_on_drop: true,
            generate_html: false,
            html_path: None,
        }
    }
    
    /// Nastavi, ali naj se izpiše poročilo ob zaključku
    pub fn print_on_drop(mut self, print: bool) -> Self {
        self.print_on_drop = print;
        self
    }
    
    /// Nastavi, ali naj se generira HTML poročilo
    pub fn generate_html(mut self, generate: bool, path: Option<&str>) -> Self {
        self.generate_html = generate;
        self.html_path = path.map(String::from);
        self
    }
    
    /// Ponastavi števce
    pub fn reset(&self) {
        let mut tracker = COVERAGE_TRACKER.lock().unwrap();
        tracker.reset();
    }
    
    /// Ročno izpiše poročilo
    pub fn print_report(&self) {
        let tracker = COVERAGE_TRACKER.lock().unwrap();
        println!("Coverage Report for {}", self.name);
        tracker.print_statistics();
    }
    
    /// Ročno generiraj HTML poročilo
    pub fn generate_html_report(&self, path: &str) -> std::io::Result<()> {
        let tracker = COVERAGE_TRACKER.lock().unwrap();
        tracker.generate_html_report(path)
    }
}

impl Drop for TestCoverageCollector {
    fn drop(&mut self) {
        if self.print_on_drop {
            self.print_report();
        }
        
        if self.generate_html {
            if let Some(ref path) = self.html_path {
                if let Err(e) = self.generate_html_report(path) {
                    eprintln!("Failed to generate HTML report: {}", e);
                }
            } else {
                let default_path = format!("coverage_report_{}.html", self.name);
                if let Err(e) = self.generate_html_report(&default_path) {
                    eprintln!("Failed to generate HTML report: {}", e);
                }
            }
        }
    }
}

/// Registriraj linijo za sledenje pokritosti
pub fn register_line(module: &str, line: &str) {
    let mut tracker = COVERAGE_TRACKER.lock().unwrap();
    tracker.register(module, line);
}

/// Zabeleži izvajanje linije
pub fn track_line(module: &str, line: &str) {
    let mut tracker = COVERAGE_TRACKER.lock().unwrap();
    tracker.track(module, line);
}

/// Vrne trenutni odstotek pokritosti
pub fn current_coverage_percentage() -> f64 {
    let tracker = COVERAGE_TRACKER.lock().unwrap();
    tracker.coverage_percentage()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_coverage_tracking() {
        let mut tracker = CoverageTracker::new();
        
        // Registriraj nekaj vrstic
        tracker.register("test_module", "line1");
        tracker.register("test_module", "line2");
        tracker.register("test_module", "line3");
        
        // Zabeleži nekaj izvedb
        tracker.track("test_module", "line1");
        tracker.track("test_module", "line1");
        tracker.track("test_module", "line2");
        
        // Preveri števce
        assert_eq!(tracker.get_count("test_module", "line1"), 2);
        assert_eq!(tracker.get_count("test_module", "line2"), 1);
        assert_eq!(tracker.get_count("test_module", "line3"), 0);
        
        // Preveri nepokrite vrstice
        let uncovered = tracker.get_uncovered_lines();
        assert_eq!(uncovered.len(), 1);
        assert_eq!(uncovered[0].line, "line3");
        
        // Preveri odstotek pokritosti
        assert_eq!(tracker.coverage_percentage(), 2.0 / 3.0 * 100.0);
    }
    
    #[test]
    fn test_module_statistics() {
        let mut tracker = CoverageTracker::new();
        
        // Dva modula
        tracker.register("module1", "line1");
        tracker.register("module1", "line2");
        tracker.register("module2", "line1");
        tracker.register("module2", "line2");
        
        // Zabeleži nekaj izvedb
        tracker.track("module1", "line1");
        tracker.track("module2", "line1");
        tracker.track("module2", "line2");
        
        // Preveri statistiko
        assert_eq!(tracker.total_lines, 4);
        assert_eq!(tracker.covered_lines, 3);
        assert_eq!(tracker.coverage_percentage(), 75.0);
    }
}
