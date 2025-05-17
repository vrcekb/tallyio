//! Skupne komponente za integracijske teste
//!
//! Ta modul vsebuje pomožne funkcije in strukture, ki so skupne
//! vsem integracijskim testom v TallyIO platformi.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tempfile::{tempdir, TempDir};

/// Ustvari začasno mapo za teste, ki se avtomatsko pobriše po uporabi
pub fn create_test_dir() -> TempDir {
    tempdir().expect("Failed to create temporary directory for tests")
}

/// Vrne pot do skupne testne mape za integracijske teste
pub fn common_test_data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("common")
        .join("test_data")
}

/// Generator edinstvenih imen za teste
pub fn unique_test_name(prefix: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_nanos();
    
    format!("{}_{}_{}", prefix, std::process::id(), timestamp)
}

/// Struktura za inicializacijo testnih odvisnosti
pub struct TestDependencies {
    /// Tokio runtime za testiranje
    pub runtime: Runtime,
    /// Začasna mapa
    pub temp_dir: TempDir,
    /// Shrani druge instance, ki jih mora test vrniti
    _stored_instances: Vec<Box<dyn std::any::Any + Send + Sync>>,
}

impl TestDependencies {
    /// Ustvari nove testne odvisnosti
    pub fn new() -> Self {
        let runtime = Runtime::new().expect("Failed to create tokio runtime");
        let temp_dir = create_test_dir();
        
        Self {
            runtime,
            temp_dir,
            _stored_instances: Vec::new(),
        }
    }
    
    /// Shrani instanco za kasnejšo uporabo
    pub fn store<T: 'static + Send + Sync>(&mut self, instance: T) {
        self._stored_instances.push(Box::new(instance));
    }
    
    /// Vrne pot do začasne mape
    pub fn temp_path(&self) -> PathBuf {
        self.temp_dir.path().to_path_buf()
    }
    
    /// Izvedi asinhrono funkcijo v tokio runtime
    pub fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }
}

/// Obvesti, da se začenja nov test
pub fn log_test_start(test_name: &str) {
    println!("\n[TEST START] {} {}", test_name, "=".repeat(50 - test_name.len()));
}

/// Obvesti, da se test zaključuje
pub fn log_test_end(test_name: &str, success: bool) {
    let status = if success { "PASS" } else { "FAIL" };
    println!("[TEST END] {} - {} {}\n", test_name, status, "=".repeat(40 - test_name.len()));
}

/// Makro za merjenje časa izvajanja bloka kode
#[macro_export]
macro_rules! measure_time {
    ($name:expr, $body:block) => {{
        let start = std::time::Instant::now();
        let result = $body;
        let duration = start.elapsed();
        println!("[TIMING] {} took {:?}", $name, duration);
        (result, duration)
    }};
}

/// Makro za spremljanje časa latence kritičnih operacij
#[macro_export]
macro_rules! track_latency {
    ($report:expr, $name:expr, $body:block) => {{
        let start = std::time::Instant::now();
        let result = $body;
        let duration = start.elapsed();
        $report.add_operation($name, duration);
        result
    }};
}

/// Pomožna struktura za poganjanje testov z dodanim čiščenjem
pub struct TestRunner {
    /// Ime testa
    name: String,
    /// Seznam funkcij za čiščenje
    cleanup_fns: Vec<Box<dyn FnOnce() + Send>>,
}

impl TestRunner {
    /// Ustvari novega poganjača testov
    pub fn new(name: &str) -> Self {
        log_test_start(name);
        Self {
            name: name.to_string(),
            cleanup_fns: Vec::new(),
        }
    }
    
    /// Dodaj funkcijo za čiščenje, ki se bo izvedla ob koncu
    pub fn add_cleanup<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce() + Send + 'static,
    {
        self.cleanup_fns.push(Box::new(f));
        self
    }
    
    /// Izvedi čiščenje
    fn cleanup(&mut self) {
        for f in self.cleanup_fns.drain(..) {
            f();
        }
    }
}

impl Drop for TestRunner {
    fn drop(&mut self) {
        self.cleanup();
        log_test_end(&self.name, std::thread::panicking());
    }
}
