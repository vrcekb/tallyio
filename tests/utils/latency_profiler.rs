//! Napredno orodje za profiliranje latence kritičnih poti v MEV sistemu
//! 
//! Ta modul omogoča natančno profiliranje kritičnih poti v MEV sistemu, vključno
//! z mikro-benchmarki, flamegraphs, in avtomatsko identifikacijo ozkih grl.
//! Ključno za MEV operacije je odkrivanje in odpravljanje tudi najmanjših 
//! virov latence, ki lahko povzročijo izgubo priložnosti.

use std::collections::{HashMap, BTreeMap};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use once_cell::sync::Lazy;
use std::cell::RefCell;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Globalni profilirnik za uporabo po celotnem projektu
pub static GLOBAL_PROFILER: Lazy<Arc<Mutex<LatencyProfiler>>> = 
    Lazy::new(|| Arc::new(Mutex::new(LatencyProfiler::new())));

/// Struktura za enkratno merjenje
pub struct Measurement {
    /// Ime merjenja
    pub name: String,
    /// Čas začetka
    start: Instant,
    /// Oznaka za prikaz v hierarhiji
    parent: Option<String>,
    /// Globina gnezdenja
    depth: usize,
}

impl Measurement {
    /// Ustvari novo merjenje
    fn new(name: &str, parent: Option<String>, depth: usize) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
            parent,
            depth,
        }
    }
    
    /// Zaključi merjenje in vrni rezultat
    fn end(self) -> MeasurementResult {
        let elapsed = self.start.elapsed();
        MeasurementResult {
            name: self.name,
            duration: elapsed,
            parent: self.parent,
            depth: self.depth,
        }
    }
}

/// Rezultat posameznega merjenja
#[derive(Debug, Clone)]
pub struct MeasurementResult {
    /// Ime merjenja
    pub name: String,
    /// Izmerjeni čas
    pub duration: Duration,
    /// Starš v gnezdenju
    pub parent: Option<String>,
    /// Globina gnezdenja
    pub depth: usize,
}

/// Statistika za skupino meritev
#[derive(Debug, Clone)]
pub struct MeasurementStats {
    /// Ime skupine meritev
    pub name: String,
    /// Število meritev
    pub count: usize,
    /// Minimalni čas
    pub min: Duration,
    /// Maksimalni čas
    pub max: Duration,
    /// Povprečni čas
    pub avg: Duration,
    /// Mediana
    pub median: Duration,
    /// 95. percentil
    pub p95: Duration,
    /// 99. percentil
    pub p99: Duration,
    /// Standardna deviacija
    pub std_dev: Duration,
    /// Skupni čas vseh meritev
    pub total: Duration,
    /// Delež v skupnem času
    pub percentage: f64,
}

/// Nit-lokalno sledenje trenutnim meritvam
thread_local! {
    static CURRENT_MEASUREMENTS: RefCell<Vec<Measurement>> = RefCell::new(Vec::new());
}

/// Profilirnik latence za MEV kritične poti
#[derive(Debug)]
pub struct LatencyProfiler {
    /// Vsa izvedena merjenja
    measurements: Vec<MeasurementResult>,
    /// Ali je profiliranje omogočeno
    enabled: bool,
    /// Dodatne opombe za vsako merjenje
    annotations: HashMap<String, String>,
    /// Skladišče za flamegraph podatke
    flamegraph_data: Vec<(String, Duration, usize)>,
}

impl LatencyProfiler {
    /// Ustvari nov profilirnik
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            enabled: true,
            annotations: HashMap::new(),
            flamegraph_data: Vec::new(),
        }
    }
    
    /// Omogoči ali onemogoči profiliranje
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Začni merjenje odseka kode
    pub fn start_measurement(&mut self, name: &str) -> Measurement {
        if !self.enabled {
            return Measurement::new(name, None, 0);
        }
        
        let (parent, depth) = CURRENT_MEASUREMENTS.with(|measurements| {
            let measurements = measurements.borrow();
            let parent = measurements.last().map(|m| m.name.clone());
            let depth = measurements.len();
            (parent, depth)
        });
        
        let measurement = Measurement::new(name, parent, depth);
        
        CURRENT_MEASUREMENTS.with(|measurements| {
            measurements.borrow_mut().push(measurement.clone());
        });
        
        measurement
    }
    
    /// Končaj merjenje in zabeleži rezultat
    pub fn end_measurement(&mut self, measurement: Measurement) {
        if !self.enabled {
            return;
        }
        
        let result = measurement.end();
        
        CURRENT_MEASUREMENTS.with(|measurements| {
            let mut measurements = measurements.borrow_mut();
            if let Some(last) = measurements.last() {
                if last.name == result.name {
                    measurements.pop();
                }
            }
        });
        
        // Zabeleži podatke za flamegraph
        self.flamegraph_data.push((
            result.name.clone(),
            result.duration,
            result.depth
        ));
        
        self.measurements.push(result);
    }
    
    /// Dodaj opombo za določeno merjenje
    pub fn annotate(&mut self, measurement_name: &str, note: &str) {
        self.annotations.insert(measurement_name.to_string(), note.to_string());
    }
    
    /// Ponastavi vse meritve
    pub fn reset(&mut self) {
        self.measurements.clear();
        self.annotations.clear();
        self.flamegraph_data.clear();
    }
    
    /// Vrni vse rezultate meritev
    pub fn get_measurements(&self) -> &[MeasurementResult] {
        &self.measurements
    }
    
    /// Vrni statistiko za vsako skupino meritev
    pub fn get_stats(&self) -> HashMap<String, MeasurementStats> {
        let mut stats = HashMap::new();
        
        // Najprej grupiramo meritve po imenu
        let mut grouped: HashMap<String, Vec<Duration>> = HashMap::new();
        for measurement in &self.measurements {
            let entry = grouped.entry(measurement.name.clone()).or_insert_with(Vec::new);
            entry.push(measurement.duration);
        }
        
        // Izračunamo skupni čas vseh meritev
        let total_time: Duration = self.measurements.iter()
            .filter(|m| m.depth == 0) // Samo korenski klici
            .map(|m| m.duration)
            .sum();
        
        // Izračunamo statistiko za vsako skupino
        for (name, durations) in grouped {
            if durations.is_empty() {
                continue;
            }
            
            let count = durations.len();
            let mut sorted = durations.clone();
            sorted.sort();
            
            let min = sorted[0];
            let max = sorted[count - 1];
            
            let total: Duration = durations.iter().sum();
            let avg = total / count as u32;
            
            let median_idx = count / 2;
            let median = sorted[median_idx];
            
            let p95_idx = (count as f64 * 0.95) as usize;
            let p95 = sorted[p95_idx.min(count - 1)];
            
            let p99_idx = (count as f64 * 0.99) as usize;
            let p99 = sorted[p99_idx.min(count - 1)];
            
            // Izračun standardne deviacije
            let variance = durations.iter()
                .map(|&d| {
                    let diff = if d > avg { d - avg } else { avg - d };
                    diff.as_nanos().pow(2)
                })
                .sum::<u128>() / count as u128;
            
            let std_dev = Duration::from_nanos((variance as f64).sqrt() as u64);
            
            // Izračun deleža skupnega časa
            let percentage = if total_time.as_nanos() > 0 {
                total.as_nanos() as f64 / total_time.as_nanos() as f64 * 100.0
            } else {
                0.0
            };
            
            stats.insert(name.clone(), MeasurementStats {
                name,
                count,
                min,
                max,
                avg,
                median,
                p95,
                p99,
                std_dev,
                total,
                percentage,
            });
        }
        
        stats
    }
    
    /// Izpiši poročilo o performanci
    pub fn print_report(&self) {
        if self.measurements.is_empty() {
            println!("Ni meritev za prikaz.");
            return;
        }
        
        println!("\n===== POROČILO O LATENCI MEV OPERACIJ =====");
        
        let stats = self.get_stats();
        
        // Razvrsti po skupnem času (padajoče)
        let mut sorted_stats: Vec<&MeasurementStats> = stats.values().collect();
        sorted_stats.sort_by(|a, b| b.total.cmp(&a.total));
        
        println!("\n{:<30} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8}",
                 "Operacija", "Št. klicev", "Povpr. [µs]", "P95 [µs]", "P99 [µs]", "Max [µs]", "Delež [%]");
        println!("{:-<110}", "");
        
        for stat in sorted_stats {
            println!("{:<30} | {:>8} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2} | {:>8.2}",
                     stat.name,
                     stat.count,
                     stat.avg.as_micros() as f64,
                     stat.p95.as_micros() as f64,
                     stat.p99.as_micros() as f64,
                     stat.max.as_micros() as f64,
                     stat.percentage);
            
            // Dodaj opombe, če obstajajo
            if let Some(note) = self.annotations.get(&stat.name) {
                println!("   OPOMBA: {}", note);
            }
        }
        
        // Prikaži kritične poti
        println!("\n----- KRITIČNE POTI ZA MEV OPTIMIZACIJO -----");
        
        // Identificiraj meritve, ki presegajo prag
        let critical_paths: Vec<&MeasurementStats> = sorted_stats.iter()
            .filter(|&stat| {
                stat.p99.as_micros() > 100 || // presega 100µs
                stat.percentage > 10.0        // ali predstavlja >10% skupnega časa
            })
            .copied()
            .collect();
        
        if critical_paths.is_empty() {
            println!("Ni identificiranih kritičnih poti. Vse operacije so znotraj MEV zahtev!");
        } else {
            for (i, path) in critical_paths.iter().enumerate() {
                println!("{}. {} - P99: {}µs, Delež: {:.2}%", 
                         i + 1, 
                         path.name, 
                         path.p99.as_micros(), 
                         path.percentage);
                
                // Prikaži predloge za optimizacijo
                self.print_optimization_suggestions(&path.name);
            }
        }
        
        println!("\n===========================================");
    }
    
    /// Prikaži predloge za optimizacijo določene poti
    fn print_optimization_suggestions(&self, path_name: &str) {
        let suggestions = match path_name {
            name if name.contains("blockchain") || name.contains("tx_") => vec![
                "Uporabi batched RPC klice namesto posameznih",
                "Vzpostavi stalne WebSocket povezave za hitrejši dostop",
                "Implementiraj lokalnega cache za blockchain podatke",
                "Optimiziraj serijalizacijo/deserializacijo transakcij"
            ],
            name if name.contains("storage") || name.contains("load") => vec![
                "Uporabi zero-copy pristop pri branju podatkov",
                "Implementiraj predpomnilnik za pogosto dostopane podatke",
                "Optimiziraj shemo podatkov za čim manj disk access"
            ],
            name if name.contains("crypto") || name.contains("sign") => vec![
                "Uporabi batch podpisovanje, kjer je mogoče",
                "Premisli o asinhronem podpisovanju za nekritične poti",
                "Optimiziraj crypto operacije z uporabo Rust-SIMD"
            ],
            name if name.contains("arbitrage") || name.contains("opportunity") => vec![
                "Paralelno procesiranje priložnosti",
                "Predizračun pogostih poti za arbitražo",
                "Optimiziraj math operacije z uporabo približkov"
            ],
            name if name.contains("mempool") || name.contains("monitor") => vec![
                "Implementiraj filtrirani monitoring samo za relevantne transakcije",
                "Uporabi bloom filter za hitro preverjanje potencialnih MEV transakcij",
                "Optimiziraj parsanje in prioritizacijo transakcij"
            ],
            _ => vec![
                "Profiliraj z flamegraph za identifikacijo ozkih grl",
                "Preveri alokacije spomina v tej funkciji",
                "Razmisli o paralelizaciji operacij"
            ]
        };
        
        println!("   PREDLOGI ZA OPTIMIZACIJO:");
        for suggestion in suggestions {
            println!("    * {}", suggestion);
        }
    }
    
    /// Generiraj flamegraph za vizualizacijo časa izvajanja
    pub fn generate_flamegraph(&self, output_path: impl AsRef<Path>) -> std::io::Result<()> {
        if self.flamegraph_data.is_empty() {
            return Ok(());
        }
        
        // Ustvari hierarhično strukturo za flamegraph
        let mut call_tree: BTreeMap<String, (Duration, Vec<String>)> = BTreeMap::new();
        
        // Gradnja drevesa
        for (name, duration, depth) in &self.flamegraph_data {
            let full_path = if *depth > 0 {
                // Poišči starše
                let mut parent_path = String::new();
                let mut current_depth = *depth - 1;
                
                for (p_name, _, p_depth) in self.flamegraph_data.iter().rev() {
                    if *p_depth == current_depth {
                        if parent_path.is_empty() {
                            parent_path = p_name.clone();
                        } else {
                            parent_path = format!("{};{}", p_name, parent_path);
                        }
                        
                        if current_depth == 0 {
                            break;
                        }
                        
                        current_depth -= 1;
                    }
                }
                
                if parent_path.is_empty() {
                    name.clone()
                } else {
                    format!("{};{}", parent_path, name)
                }
            } else {
                name.clone()
            };
            
            let entry = call_tree.entry(full_path.clone()).or_insert_with(|| (Duration::new(0, 0), Vec::new()));
            entry.0 += *duration;
        }
        
        // Ustvari flamegraph format datoteko
        let mut file = File::create(output_path)?;
        
        for (path, (duration, _)) in call_tree {
            writeln!(file, "{} {}", path, duration.as_micros())?;
        }
        
        Ok(())
    }
    
    /// Ustvari JSON poročilo o meritvah za analizo
    pub fn generate_json(&self, output_path: impl AsRef<Path>) -> std::io::Result<()> {
        use serde_json::{json, to_string_pretty};
        
        let stats = self.get_stats();
        
        let json_data = json!({
            "measurements": self.measurements.iter().map(|m| {
                json!({
                    "name": m.name,
                    "duration_micros": m.duration.as_micros(),
                    "parent": m.parent,
                    "depth": m.depth
                })
            }).collect::<Vec<_>>(),
            "stats": stats.values().map(|s| {
                json!({
                    "name": s.name,
                    "count": s.count,
                    "min_micros": s.min.as_micros(),
                    "max_micros": s.max.as_micros(),
                    "avg_micros": s.avg.as_micros(),
                    "median_micros": s.median.as_micros(),
                    "p95_micros": s.p95.as_micros(),
                    "p99_micros": s.p99.as_micros(),
                    "std_dev_micros": s.std_dev.as_micros(),
                    "total_micros": s.total.as_micros(),
                    "percentage": s.percentage
                })
            }).collect::<Vec<_>>(),
            "annotations": self.annotations
        });
        
        let json_string = to_string_pretty(&json_data)?;
        let mut file = File::create(output_path)?;
        file.write_all(json_string.as_bytes())?;
        
        Ok(())
    }
    
    /// Analiziraj pomnilniško delo in alokacije
    pub fn analyze_allocations(&self) {
        // Ta funkcija bi bila implementirana z uporabo instrumentacije
        // spominske porabe, kar zahteva dodatno integracijo
        // V pravi implementaciji bi beležila alokacije med meritvami
        println!("Analiza alokacij spomina ni implementirana v tej verziji.");
    }
}

/// Razred za avtomatsko merjenje odseka kode
pub struct ScopedMeasurement {
    name: String,
    start_time: Instant,
}

impl ScopedMeasurement {
    /// Ustvari novo avtomatsko merjenje
    pub fn new(name: &str) -> Self {
        let profiler = GLOBAL_PROFILER.lock().unwrap();
        if !profiler.enabled {
            return Self {
                name: name.to_string(),
                start_time: Instant::now(),
            };
        }
        drop(profiler);
        
        println!("MEV PROFILER: Začetek merjenja '{}'", name);
        
        Self {
            name: name.to_string(),
            start_time: Instant::now(),
        }
    }
}

impl Drop for ScopedMeasurement {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        println!("MEV PROFILER: Konec merjenja '{}' - {:?}", self.name, duration);
        
        let mut profiler = GLOBAL_PROFILER.lock().unwrap();
        if profiler.enabled {
            profiler.measurements.push(MeasurementResult {
                name: self.name.clone(),
                duration,
                parent: None,
                depth: 0,
            });
        }
    }
}

/// Makro za enostavno merjenje odseka kode
#[macro_export]
macro_rules! measure {
    ($name:expr, $code:block) => {{
        let profiler = $crate::utils::latency_profiler::GLOBAL_PROFILER.clone();
        let measurement = profiler.lock().unwrap().start_measurement($name);
        let result = { $code };
        profiler.lock().unwrap().end_measurement(measurement);
        result
    }};
}

/// Makro za avtomatsko merjenje funkcije
#[macro_export]
macro_rules! measure_fn {
    () => {
        let _measurement = $crate::utils::latency_profiler::ScopedMeasurement::new(
            &format!("{}::{}", module_path!(), function_name!()));
    };
    ($name:expr) => {
        let _measurement = $crate::utils::latency_profiler::ScopedMeasurement::new($name);
    };
}

/// Pomožna struktura za programsko uporabo profilirnika
pub struct ProfilerSession {
    /// Ime seje
    name: String,
    /// Čas začetka
    start_time: Instant,
    /// Shrani rezultate v flamegraph
    save_flamegraph: bool,
    /// Shrani rezultate v JSON
    save_json: bool,
    /// Pot za shranjevanje
    output_path: Option<String>,
}

impl ProfilerSession {
    /// Ustvari novo sejo profiliranja
    pub fn new(name: &str) -> Self {
        // Resetira globalni profilirnik
        GLOBAL_PROFILER.lock().unwrap().reset();
        
        Self {
            name: name.to_string(),
            start_time: Instant::now(),
            save_flamegraph: false,
            save_json: false,
            output_path: None,
        }
    }
    
    /// Omogoči shranjevanje flamegraph
    pub fn with_flamegraph(mut self, enabled: bool) -> Self {
        self.save_flamegraph = enabled;
        self
    }
    
    /// Omogoči shranjevanje JSON poročila
    pub fn with_json(mut self, enabled: bool) -> Self {
        self.save_json = enabled;
        self
    }
    
    /// Nastavi pot za shranjevanje
    pub fn with_output_path(mut self, path: &str) -> Self {
        self.output_path = Some(path.to_string());
        self
    }
    
    /// Ročno zaključi sejo
    pub fn end(self) {
        let total_time = self.start_time.elapsed();
        
        println!("\n----- MEV PROFILER: Seja '{}' zaključena v {:?} -----", self.name, total_time);
        
        let profiler = GLOBAL_PROFILER.lock().unwrap();
        profiler.print_report();
        
        if self.save_flamegraph || self.save_json {
            let base_path = self.output_path.unwrap_or_else(|| "target/profile".to_string());
            
            if self.save_flamegraph {
                let flamegraph_path = format!("{}/{}_flamegraph.txt", base_path, self.name);
                if let Err(e) = profiler.generate_flamegraph(&flamegraph_path) {
                    eprintln!("Napaka pri ustvarjanju flamegraph: {}", e);
                } else {
                    println!("Flamegraph shranjen v: {}", flamegraph_path);
                }
            }
            
            if self.save_json {
                let json_path = format!("{}/{}_profile.json", base_path, self.name);
                if let Err(e) = profiler.generate_json(&json_path) {
                    eprintln!("Napaka pri ustvarjanju JSON poročila: {}", e);
                } else {
                    println!("JSON poročilo shranjeno v: {}", json_path);
                }
            }
        }
    }
}

impl Drop for ProfilerSession {
    fn drop(&mut self) {
        // Avtomatsko zaključi sejo, če ni bila ročno zaključena
        if self.start_time.elapsed() < Duration::from_secs(1800) { // Max 30 minut
            println!("\n----- MEV PROFILER: Seja '{}' avtomatsko zaključena -----", self.name);
            
            let profiler = GLOBAL_PROFILER.lock().unwrap();
            profiler.print_report();
        }
    }
}

// Izvozi glavne funkcije za enostavno uporabo
pub use self::GLOBAL_PROFILER as profiler;

/// Ustvari novo sejo profiliranja
pub fn profile_session(name: &str) -> ProfilerSession {
    ProfilerSession::new(name)
}

/// Globalno omogoči/onemogoči profiliranje
pub fn set_profiling_enabled(enabled: bool) {
    GLOBAL_PROFILER.lock().unwrap().set_enabled(enabled);
}

/// Resetiraj vse meritve
pub fn reset_measurements() {
    GLOBAL_PROFILER.lock().unwrap().reset();
}

/// Izpiši poročilo o trenutnih meritvah
pub fn print_report() {
    GLOBAL_PROFILER.lock().unwrap().print_report();
}

/// Generiraj flamegraph za trenutne meritve
pub fn generate_flamegraph(output_path: &str) -> std::io::Result<()> {
    GLOBAL_PROFILER.lock().unwrap().generate_flamegraph(output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_measurement() {
        let mut profiler = LatencyProfiler::new();
        
        let measurement = profiler.start_measurement("test_operation");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_measurement(measurement);
        
        let measurements = profiler.get_measurements();
        assert_eq!(measurements.len(), 1);
        assert_eq!(measurements[0].name, "test_operation");
        assert!(measurements[0].duration >= Duration::from_millis(10));
    }
    
    #[test]
    fn test_nested_measurements() {
        let mut profiler = LatencyProfiler::new();
        
        let outer = profiler.start_measurement("outer");
        std::thread::sleep(Duration::from_millis(5));
        
        let inner = profiler.start_measurement("inner");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_measurement(inner);
        
        std::thread::sleep(Duration::from_millis(5));
        profiler.end_measurement(outer);
        
        let measurements = profiler.get_measurements();
        assert_eq!(measurements.len(), 2);
        
        // Preverimo gnezdenje
        let inner_measurement = measurements.iter().find(|m| m.name == "inner").unwrap();
        assert_eq!(inner_measurement.parent, Some("outer".to_string()));
        assert_eq!(inner_measurement.depth, 1);
        
        let outer_measurement = measurements.iter().find(|m| m.name == "outer").unwrap();
        assert_eq!(outer_measurement.parent, None);
        assert_eq!(outer_measurement.depth, 0);
    }
    
    #[test]
    fn test_measurement_stats() {
        let mut profiler = LatencyProfiler::new();
        
        // Dodaj več meritev z različnimi časi
        for i in 0..100 {
            let measurement = profiler.start_measurement("test_operation");
            std::thread::sleep(Duration::from_micros(i * 10));
            profiler.end_measurement(measurement);
        }
        
        let stats = profiler.get_stats();
        assert!(stats.contains_key("test_operation"));
        
        let op_stats = &stats["test_operation"];
        assert_eq!(op_stats.count, 100);
        assert!(op_stats.min < op_stats.max);
        assert!(op_stats.p95 > op_stats.median);
        assert!(op_stats.p99 > op_stats.p95);
    }
    
    #[test]
    fn test_profiler_session() {
        let session = ProfilerSession::new("test_session")
            .with_flamegraph(false)
            .with_json(false);
        
        // Izvedi nekaj meritev
        {
            let _m = measure!("test_op", {
                std::thread::sleep(Duration::from_millis(10));
            });
        }
        
        session.end();
        
        // Preverimo, da je bil profiler resetiran po koncu seje
        let measurements = GLOBAL_PROFILER.lock().unwrap().get_measurements();
        assert_eq!(measurements.len(), 1);
    }
}
