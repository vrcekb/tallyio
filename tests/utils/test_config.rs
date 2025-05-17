//! Centralna konfiguracijska datoteka za TallyIO testno ogrodje
//!
//! Ta modul definira konfiguracijske strukture, s katerimi je mogoče
//! prilagoditi obnašanje testnega ogrodja TallyIO za različne module in scenarije.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Nivoji beleženja (logging)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    /// Brez beleženja
    None,
    /// Samo napake
    Error,
    /// Napake in opozorila
    Warning,
    /// Osnovno beleženje (napake, opozorila, informacije)
    Info,
    /// Podrobno beleženje (vse zgornje plus debug)
    Debug,
    /// Najpodrobnejše beleženje za sledenje
    Trace,
}

/// Nivoji zahtevnosti performančnih testov
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTestLevel {
    /// Samo hitri, osnovni testi
    Basic,
    /// Standardni nabor testov
    Standard,
    /// Razširjeni nabor testov
    Extended,
    /// Celoviti stresni testi
    Stress,
}

/// Nivoji zahtevnosti varnostnih testov
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityTestLevel {
    /// Osnovni varnostni testi
    Basic,
    /// Standardni varnostni testi
    Standard,
    /// Napredni varnostni testi
    Advanced,
    /// Celoviti penetracijski testi
    Pentesting,
}

/// Glavna konfiguracijska struktura za teste
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Ime testne konfiguracije
    pub name: String,
    /// Nivo beleženja
    pub log_level: LogLevel,
    /// Pot do začasne mape za teste
    pub temp_dir: Option<PathBuf>,
    /// Časovna omejitev za teste (v sekundah)
    pub timeout_seconds: u64,
    /// Ali naj se izvajajo počasni testi
    pub run_slow_tests: bool,
    /// Ali naj se izvajajo stresni testi
    pub run_stress_tests: bool,
    /// Ali naj se izvajajo testi v vzporednih nitih
    pub parallel_execution: bool,
    /// Nivo za performančne teste
    pub performance_test_level: PerformanceTestLevel,
    /// Nivo za varnostne teste
    pub security_test_level: SecurityTestLevel,
    /// Ali naj se generira poročilo o pokritosti
    pub generate_coverage_report: bool,
    /// Dodatni parametri (specifični za posamezne module)
    pub extra_params: HashMap<String, String>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            log_level: LogLevel::Error,
            temp_dir: None,
            timeout_seconds: 60,
            run_slow_tests: false,
            run_stress_tests: false,
            parallel_execution: true,
            performance_test_level: PerformanceTestLevel::Standard,
            security_test_level: SecurityTestLevel::Standard,
            generate_coverage_report: true,
            extra_params: HashMap::new(),
        }
    }
}

impl TestConfig {
    /// Ustvari novo testno konfiguracijo z imenom
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }
    
    /// Nastavi nivo beleženja
    pub fn with_log_level(mut self, log_level: LogLevel) -> Self {
        self.log_level = log_level;
        self
    }
    
    /// Nastavi začasno mapo
    pub fn with_temp_dir(mut self, temp_dir: PathBuf) -> Self {
        self.temp_dir = Some(temp_dir);
        self
    }
    
    /// Nastavi časovno omejitev
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }
    
    /// Omogoči ali onemogoči počasne teste
    pub fn with_slow_tests(mut self, enabled: bool) -> Self {
        self.run_slow_tests = enabled;
        self
    }
    
    /// Omogoči ali onemogoči stresne teste
    pub fn with_stress_tests(mut self, enabled: bool) -> Self {
        self.run_stress_tests = enabled;
        self
    }
    
    /// Omogoči ali onemogoči vzporedno izvajanje
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }
    
    /// Nastavi nivo performančnih testov
    pub fn with_performance_level(mut self, level: PerformanceTestLevel) -> Self {
        self.performance_test_level = level;
        self
    }
    
    /// Nastavi nivo varnostnih testov
    pub fn with_security_level(mut self, level: SecurityTestLevel) -> Self {
        self.security_test_level = level;
        self
    }
    
    /// Omogoči ali onemogoči poročanje o pokritosti
    pub fn with_coverage_report(mut self, enabled: bool) -> Self {
        self.generate_coverage_report = enabled;
        self
    }
    
    /// Dodaj dodatni parameter
    pub fn with_extra_param(mut self, key: &str, value: &str) -> Self {
        self.extra_params.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Vrne časovno omejitev kot Duration
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_seconds)
    }
}

/// Vrne konfiguracijo za unit teste secure_storage modula
pub fn secure_storage_unit_config() -> TestConfig {
    TestConfig::new("secure_storage_unit")
        .with_log_level(LogLevel::Debug)
        .with_timeout(30)
        .with_parallel_execution(true)
        .with_coverage_report(true)
        .with_extra_param("encryption_rounds", "1000")
        .with_extra_param("validate_encryption", "true")
}

/// Vrne konfiguracijo za performančne teste secure_storage modula
pub fn secure_storage_performance_config() -> TestConfig {
    TestConfig::new("secure_storage_performance")
        .with_log_level(LogLevel::Info)
        .with_timeout(120)
        .with_performance_level(if cfg!(feature = "extended-tests") {
            PerformanceTestLevel::Extended
        } else {
            PerformanceTestLevel::Standard
        })
        .with_extra_param("concurrency_level", "32")
        .with_extra_param("test_duration_seconds", "60")
}

/// Vrne konfiguracijo za varnostne teste secure_storage modula
pub fn secure_storage_security_config() -> TestConfig {
    TestConfig::new("secure_storage_security")
        .with_log_level(LogLevel::Debug)
        .with_timeout(180)
        .with_security_level(SecurityTestLevel::Advanced)
        .with_slow_tests(true)
        .with_extra_param("fuzz_iterations", "1000")
        .with_extra_param("test_corruption", "true")
}

/// Vrne konfiguracijo za MEV specifične teste
pub fn mev_test_config() -> TestConfig {
    TestConfig::new("mev_tests")
        .with_log_level(LogLevel::Debug)
        .with_timeout(120)
        .with_performance_level(PerformanceTestLevel::Extended)
        .with_extra_param("simulate_blocks", "100")
        .with_extra_param("simulate_txs_per_block", "50")
        .with_extra_param("opportunity_types", "DexArbitrage,Liquidation,Sandwich")
}

/// Vrne integrirano konfiguracijo za celovito testiranje projekta
pub fn full_project_test_config() -> TestConfig {
    let mut config = TestConfig::new("full_project");
    
    // Nastavi parametre glede na okolje
    if cfg!(feature = "ci") {
        // CI okolje - fokus na hitrosti in popolni pokritosti
        config = config
            .with_log_level(LogLevel::Info)
            .with_parallel_execution(true)
            .with_slow_tests(false)
            .with_stress_tests(false)
            .with_performance_level(PerformanceTestLevel::Basic)
            .with_security_level(SecurityTestLevel::Standard)
            .with_coverage_report(true);
    } else if cfg!(feature = "nightly") {
        // Nočno testiranje - izvedi vse teste
        config = config
            .with_log_level(LogLevel::Debug)
            .with_parallel_execution(true)
            .with_slow_tests(true)
            .with_stress_tests(true)
            .with_performance_level(PerformanceTestLevel::Stress)
            .with_security_level(SecurityTestLevel::Pentesting)
            .with_coverage_report(true);
    } else if cfg!(feature = "dev") {
        // Razvojno okolje - hiter feedback loop
        config = config
            .with_log_level(LogLevel::Debug)
            .with_parallel_execution(true)
            .with_slow_tests(false)
            .with_stress_tests(false)
            .with_performance_level(PerformanceTestLevel::Basic)
            .with_security_level(SecurityTestLevel::Basic)
            .with_coverage_report(true);
    }
    
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_builder() {
        let config = TestConfig::new("test")
            .with_log_level(LogLevel::Debug)
            .with_timeout(120)
            .with_slow_tests(true);
        
        assert_eq!(config.name, "test");
        assert_eq!(config.log_level, LogLevel::Debug);
        assert_eq!(config.timeout_seconds, 120);
        assert!(config.run_slow_tests);
    }
    
    #[test]
    fn test_predefined_configs() {
        let unit_config = secure_storage_unit_config();
        let perf_config = secure_storage_performance_config();
        let sec_config = secure_storage_security_config();
        
        assert_eq!(unit_config.name, "secure_storage_unit");
        assert_eq!(perf_config.performance_test_level, PerformanceTestLevel::Standard);
        assert_eq!(sec_config.security_test_level, SecurityTestLevel::Advanced);
    }
}
