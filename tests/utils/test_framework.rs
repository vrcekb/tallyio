//! Modularno testno ogrodje za TallyIO platformo
//!
//! Ta modul zagotavlja standardizirane gradnike za testiranje vseh TallyIO komponent.
//! Namen je zmanjšati podvajanje kode in zagotoviti konsistentno metodologijo 
//! testiranja skozi celotno MEV platformo.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::{tempdir, TempDir};
use tokio::runtime::Runtime;

/// Konfiguracijski parametri za teste
#[derive(Debug)]
pub struct TestConfig {
    /// Časovna omejitev za posamezen test
    pub timeout: Duration,
    /// Ali naj se izvedejo počasni testi
    pub run_slow_tests: bool,
    /// Ali naj se izvedejo stress testi
    pub run_stress_tests: bool,
    /// Nivo log output-a
    pub log_level: LogLevel,
}

/// Log nivoji za teste
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Silent,
    Error,
    Info,
    Debug,
    Trace,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            run_slow_tests: false,
            run_stress_tests: false,
            log_level: LogLevel::Error,
        }
    }
}

/// Struktura, ki drži testno okolje
pub struct TestEnvironment {
    /// Začasni direktorij za teste
    pub temp_dir: TempDir,
    /// Runtime za asinhrone teste
    pub runtime: Runtime,
    /// Konfiguracijski parametri
    pub config: TestConfig,
}

impl TestEnvironment {
    /// Ustvari novo testno okolje s privzetimi nastavitvami
    pub fn new() -> Self {
        Self::with_config(TestConfig::default())
    }
    
    /// Ustvari testno okolje s specifično konfiguracijo
    pub fn with_config(config: TestConfig) -> Self {
        let temp_dir = tempdir().expect("Failed to create temp directory for tests");
        let runtime = Runtime::new().expect("Failed to create tokio runtime for tests");
        
        Self {
            temp_dir,
            runtime,
            config,
        }
    }
    
    /// Vrne pot do začasnega direktorija
    pub fn temp_path(&self) -> PathBuf {
        self.temp_dir.path().to_path_buf()
    }
    
    /// Ustvari novo datoteko v začasnem direktoriju s podanim imenom
    pub fn temp_file(&self, name: &str) -> PathBuf {
        self.temp_dir.path().join(name)
    }
    
    /// Izvede asinhron blok kode in meri čas izvajanja
    pub fn run_async<F, R>(&self, func: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.runtime.block_on(async {
            let start = Instant::now();
            let result = tokio::task::spawn_blocking(func).await.expect("Task failed");
            let duration = start.elapsed();
            
            if self.config.log_level >= LogLevel::Debug {
                println!("Task completed in {:?}", duration);
            }
            
            result
        })
    }
    
    /// Izvede asinhron test z časovno omejitvijo
    pub fn run_async_with_timeout<F, Fut, R>(&self, func: F) -> R
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        self.runtime.block_on(async {
            let timeout = self.config.timeout;
            tokio::time::timeout(timeout, func()).await.expect("Test timed out")
        })
    }
    
    /// Pomožna metoda za izpis informacij
    pub fn log(&self, level: LogLevel, message: &str) {
        if level <= self.config.log_level {
            match level {
                LogLevel::Silent => {}
                LogLevel::Error => eprintln!("ERROR: {}", message),
                LogLevel::Info => println!("INFO: {}", message),
                LogLevel::Debug => println!("DEBUG: {}", message),
                LogLevel::Trace => println!("TRACE: {}", message),
            }
        }
    }
}

/// Struktura za merjenje performanc in statističnih podatkov
pub struct PerformanceMetrics {
    pub operation_name: String,
    pub iterations: usize,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub durations: Vec<Duration>,
}

impl PerformanceMetrics {
    /// Ustvari novo instanco za podano operacijo
    pub fn new(operation_name: &str) -> Self {
        Self {
            operation_name: operation_name.to_string(),
            iterations: 0,
            total_duration: Duration::default(),
            min_duration: Duration::from_secs(u64::MAX),
            max_duration: Duration::default(),
            durations: Vec::new(),
        }
    }
    
    /// Zabeleži trajanje operacije
    pub fn record_duration(&mut self, duration: Duration) {
        self.iterations += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.durations.push(duration);
    }
    
    /// Vrne povprečen čas trajanja operacije
    pub fn average_duration(&self) -> Duration {
        if self.iterations == 0 {
            return Duration::default();
        }
        self.total_duration / self.iterations as u32
    }
    
    /// Izračuna standardno deviacijo časov
    pub fn std_deviation(&self) -> f64 {
        if self.iterations <= 1 {
            return 0.0;
        }
        
        let avg_nanos = self.average_duration().as_nanos() as f64;
        let variance: f64 = self.durations.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - avg_nanos;
                diff * diff
            })
            .sum::<f64>() / (self.iterations as f64);
        
        variance.sqrt()
    }
    
    /// Izpiše statistiko v čitljivi obliki
    pub fn print_statistics(&self) {
        println!("Performance statistics for: {}", self.operation_name);
        println!("  Iterations:      {}", self.iterations);
        println!("  Total duration:  {:?}", self.total_duration);
        println!("  Average:         {:?}", self.average_duration());
        println!("  Min duration:    {:?}", self.min_duration);
        println!("  Max duration:    {:?}", self.max_duration);
        println!("  Std Deviation:   {:.2} ns", self.std_deviation());
    }
}

/// Generator testnih podatkov
pub struct TestDataGenerator;

impl TestDataGenerator {
    /// Generira podatke različnih velikosti
    pub fn generate_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push((i % 256) as u8);
        }
        data
    }
    
    /// Generira naključne podatke
    pub fn generate_random_data(size: usize) -> Vec<u8> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(rng.gen::<u8>());
        }
        data
    }
}

/// Asinhron pomožni razred za testiranje vzporednih operacij
pub struct ConcurrencyTest {
    task_count: usize,
}

impl ConcurrencyTest {
    /// Ustvari nov test vzporednosti
    pub fn new(task_count: usize) -> Self {
        Self { task_count }
    }
    
    /// Izvede enak test v vzporednih nitih in vrne rezultate
    pub async fn run<F, Fut, R>(&self, task_factory: F) -> Vec<R>
    where
        F: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let task_factory = Arc::new(task_factory);
        let mut handles = Vec::with_capacity(self.task_count);
        
        for i in 0..self.task_count {
            let task_factory = Arc::clone(&task_factory);
            let handle = tokio::spawn(async move {
                task_factory(i).await
            });
            handles.push(handle);
        }
        
        let mut results = Vec::with_capacity(self.task_count);
        for handle in handles {
            results.push(handle.await.expect("Task failed"));
        }
        
        results
    }
}

/// Modul za napredne validacije in asercije
pub mod assertions {
    use std::time::Duration;
    
    /// Preveri, da je operacija hitrejša od določenega limita
    pub fn assert_faster_than<F>(limit: Duration, func: F)
    where
        F: FnOnce(),
    {
        let start = std::time::Instant::now();
        func();
        let duration = start.elapsed();
        
        assert!(
            duration <= limit,
            "Operation took {:?}, which exceeds the limit of {:?}",
            duration,
            limit
        );
    }
    
    /// Preveri, da je razlika med dvema časoma manjša od tolerance
    pub fn assert_duration_similar(a: Duration, b: Duration, tolerance_percentage: f64) {
        let a_nanos = a.as_nanos() as f64;
        let b_nanos = b.as_nanos() as f64;
        let max = a_nanos.max(b_nanos);
        let min = a_nanos.min(b_nanos);
        
        let difference_percentage = (max - min) / min * 100.0;
        
        assert!(
            difference_percentage <= tolerance_percentage,
            "Durations differ by {:.2}%, which exceeds tolerance of {:.2}%: {:?} vs {:?}",
            difference_percentage,
            tolerance_percentage,
            a,
            b
        );
    }
}

/// Modul za simulacijo različnih napak in robnih primerov
pub mod fault_simulation {
    use std::path::Path;
    use std::fs::{self, File, OpenOptions};
    use std::io::{self, Read, Seek, SeekFrom, Write};
    use rand::Rng;
    
    /// Simulira različne napake datotečnega sistema
    pub struct FsSimulator;
    
    impl FsSimulator {
        /// Pokvari del datoteke z naključnimi podatki
        pub fn corrupt_file_partially<P: AsRef<Path>>(path: P, corrupt_percentage: f64) -> io::Result<()> {
            let path = path.as_ref();
            if !path.exists() {
                return Err(io::Error::new(io::ErrorKind::NotFound, "File not found"));
            }
            
            let metadata = fs::metadata(path)?;
            let file_size = metadata.len();
            
            let corrupt_size = (file_size as f64 * corrupt_percentage / 100.0) as usize;
            if corrupt_size == 0 {
                return Ok(());
            }
            
            let mut rng = rand::thread_rng();
            let start_pos = rng.gen_range(0..(file_size as usize - corrupt_size));
            
            let mut file = OpenOptions::new().read(true).write(true).open(path)?;
            file.seek(SeekFrom::Start(start_pos as u64))?;
            
            let random_data: Vec<u8> = (0..corrupt_size).map(|_| rng.gen()).collect();
            file.write_all(&random_data)?;
            
            Ok(())
        }
        
        /// Popolnoma pokvari datoteko
        pub fn corrupt_file_completely<P: AsRef<Path>>(path: P) -> io::Result<()> {
            let path = path.as_ref();
            if !path.exists() {
                return Err(io::Error::new(io::ErrorKind::NotFound, "File not found"));
            }
            
            let mut file = OpenOptions::new().write(true).truncate(true).open(path)?;
            let mut rng = rand::thread_rng();
            
            let random_size = rng.gen_range(10..1000);
            let random_data: Vec<u8> = (0..random_size).map(|_| rng.gen()).collect();
            file.write_all(&random_data)?;
            
            Ok(())
        }
        
        /// Nadomesti del datoteke z ničlami
        pub fn zero_out_file_section<P: AsRef<Path>>(path: P, start_percentage: f64, end_percentage: f64) -> io::Result<()> {
            let path = path.as_ref();
            if !path.exists() {
                return Err(io::Error::new(io::ErrorKind::NotFound, "File not found"));
            }
            
            let metadata = fs::metadata(path)?;
            let file_size = metadata.len();
            
            let start_pos = (file_size as f64 * start_percentage / 100.0) as u64;
            let end_pos = (file_size as f64 * end_percentage / 100.0) as u64;
            let zero_size = (end_pos - start_pos) as usize;
            
            if zero_size == 0 {
                return Ok(());
            }
            
            let mut file = OpenOptions::new().read(true).write(true).open(path)?;
            file.seek(SeekFrom::Start(start_pos))?;
            
            let zeros = vec![0u8; zero_size];
            file.write_all(&zeros)?;
            
            Ok(())
        }
    }
    
    /// Simulira sistemske napake
    pub struct SystemSimulator;
    
    impl SystemSimulator {
        /// Simulira visoko obremenitev CPU
        pub fn simulate_high_cpu_load(duration_ms: u64, thread_count: usize) {
            use std::thread;
            use std::time::{Duration, Instant};
            
            let handles: Vec<_> = (0..thread_count)
                .map(|_| {
                    thread::spawn(move || {
                        let start = Instant::now();
                        while start.elapsed().as_millis() < duration_ms as u128 {
                            // Intenzivna CPU operacija
                            let mut x = 0.0;
                            for i in 0..10_000 {
                                x += (i as f64).sqrt();
                            }
                            // Preprečimo da compiler odstrani izračun
                            if x < 0.0 {
                                panic!("Impossible");
                            }
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                let _ = handle.join();
            }
        }
        
        /// Simulira omrežno latenco
        pub async fn simulate_network_latency(latency_ms: u64) {
            tokio::time::sleep(Duration::from_millis(latency_ms)).await;
        }
        
        /// Simulira omrežne napake (kot da omrežje ni na voljo)
        pub fn simulate_network_unavailable() -> bool {
            // V testem okolju samo simuliramo napako
            false
        }
    }
}
