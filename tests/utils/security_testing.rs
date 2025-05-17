//! Specializirano ogrodje za varnostno testiranje MEV platforme
//!
//! Ta modul zagotavlja orodja za temeljito testiranje varnostnih lastnosti
//! komponent TallyIO MEV platforme, s poudarkom na robustnosti, odpornosti
//! na napade in zaščiti občutljivih podatkov.

use std::path::{Path, PathBuf};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use rand::{Rng, thread_rng};
use tokio::runtime::Runtime;
use std::sync::Arc;

/// Vrste varnostnih testov za MEV platformo
#[derive(Debug, Clone, Copy)]
pub enum SecurityTestType {
    /// Testiranje integritete podatkov
    DataIntegrity,
    /// Testiranje odpornosti na korupcijo
    CorruptionResistance,
    /// Testiranje šifriranja in dešifriranja
    Encryption,
    /// Testiranje upravljanja s ključi
    KeyManagement,
    /// Testiranje avtentikacije in avtorizacije
    Authentication,
    /// Testiranje odpornosti na vnos podatkov (input validation)
    InputValidation,
    /// Testiranje zaščite pred časovnimi napadi (timing attacks)
    TimingAttacks,
    /// Testiranje obnašanja ob nepričakovanih pogojih
    FaultTolerance,
}

/// Rezultat varnostnega testa
#[derive(Debug)]
pub struct SecurityTestResult {
    /// Tip varnostnega testa
    pub test_type: SecurityTestType,
    /// Opis testa
    pub description: String,
    /// Ali je test uspel
    pub success: bool,
    /// Podrobnosti o rezultatu
    pub details: String,
    /// Priporočila za izboljšave (če je relevantno)
    pub recommendations: Option<String>,
}

impl SecurityTestResult {
    /// Ustvari nov rezultat uspešnega testa
    pub fn success(test_type: SecurityTestType, description: &str, details: &str) -> Self {
        Self {
            test_type,
            description: description.to_string(),
            success: true,
            details: details.to_string(),
            recommendations: None,
        }
    }
    
    /// Ustvari nov rezultat neuspešnega testa
    pub fn failure(test_type: SecurityTestType, description: &str, details: &str, recommendations: Option<&str>) -> Self {
        Self {
            test_type,
            description: description.to_string(),
            success: false,
            details: details.to_string(),
            recommendations: recommendations.map(String::from),
        }
    }
    
    /// Izpiše rezultat testa v čitljivi obliki
    pub fn print(&self) {
        let status = if self.success { "PASSED" } else { "FAILED" };
        println!("=== Security Test: {:?} - {} ===", self.test_type, status);
        println!("Description: {}", self.description);
        println!("Details: {}", self.details);
        
        if let Some(ref recommendations) = self.recommendations {
            println!("Recommendations: {}", recommendations);
        }
        
        println!("=====================================");
    }
}

/// Struktura za izvajanje varnostnih testov
pub struct SecurityTester {
    /// Začasni direktorij za testiranje
    temp_dir: TempDir,
    /// Runtime za asinhrone teste
    runtime: Runtime,
}

impl SecurityTester {
    /// Ustvari novo instanco SecurityTester
    pub fn new() -> Self {
        let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
        let runtime = Runtime::new().expect("Failed to create tokio runtime");
        
        Self {
            temp_dir,
            runtime,
        }
    }
    
    /// Vrne pot do začasnega direktorija
    pub fn temp_path(&self) -> PathBuf {
        self.temp_dir.path().to_path_buf()
    }
    
    /// Ustvari začasno datoteko z naključno vsebino
    pub fn create_random_file(&self, filename: &str, size_bytes: usize) -> io::Result<PathBuf> {
        let path = self.temp_dir.path().join(filename);
        let mut file = File::create(&path)?;
        
        let mut rng = thread_rng();
        let mut buffer = vec![0u8; 1024.min(size_bytes)];
        
        let mut remaining = size_bytes;
        while remaining > 0 {
            let chunk_size = buffer.len().min(remaining);
            for b in buffer.iter_mut().take(chunk_size) {
                *b = rng.gen();
            }
            
            file.write_all(&buffer[0..chunk_size])?;
            remaining -= chunk_size;
        }
        
        Ok(path)
    }
    
    /// Pokvari datoteko na določen način
    pub fn corrupt_file<P: AsRef<Path>>(&self, path: P, corruption_type: CorruptionType) -> io::Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "File not found"));
        }
        
        match corruption_type {
            CorruptionType::RandomBytes { percentage, offset_percentage } => {
                // Pokvari naključne bajte
                let metadata = fs::metadata(path)?;
                let file_size = metadata.len();
                
                let corrupt_size = (file_size as f64 * percentage / 100.0) as usize;
                if corrupt_size == 0 {
                    return Ok(());
                }
                
                let start_offset = (file_size as f64 * offset_percentage / 100.0) as u64;
                let end_offset = (start_offset + corrupt_size as u64).min(file_size);
                
                let mut file = OpenOptions::new().read(true).write(true).open(path)?;
                file.seek(SeekFrom::Start(start_offset))?;
                
                let mut rng = thread_rng();
                let mut buffer = vec![0u8; (end_offset - start_offset) as usize];
                
                for b in &mut buffer {
                    *b = rng.gen();
                }
                
                file.write_all(&buffer)?;
            },
            CorruptionType::Truncate { percentage } => {
                // Skrajšaj datoteko
                let metadata = fs::metadata(path)?;
                let file_size = metadata.len();
                
                let new_size = (file_size as f64 * percentage / 100.0) as u64;
                let file = OpenOptions::new().write(true).open(path)?;
                file.set_len(new_size)?;
            },
            CorruptionType::ZeroOut { start_percentage, end_percentage } => {
                // Nastavi na ničle
                let metadata = fs::metadata(path)?;
                let file_size = metadata.len();
                
                let start_pos = (file_size as f64 * start_percentage / 100.0) as u64;
                let end_pos = (file_size as f64 * end_percentage / 100.0) as u64;
                
                if end_pos <= start_pos {
                    return Ok(());
                }
                
                let zero_size = (end_pos - start_pos) as usize;
                let mut file = OpenOptions::new().write(true).open(path)?;
                file.seek(SeekFrom::Start(start_pos))?;
                
                let zeros = vec![0u8; zero_size];
                file.write_all(&zeros)?;
            },
            CorruptionType::BitFlip { num_bits } => {
                // Obrni določeno število bitov
                let metadata = fs::metadata(path)?;
                let file_size = metadata.len();
                
                if file_size == 0 {
                    return Ok(());
                }
                
                let mut file = OpenOptions::new().read(true).write(true).open(path)?;
                let mut content = vec![0u8; file_size as usize];
                file.read_exact(&mut content)?;
                
                let mut rng = thread_rng();
                
                for _ in 0..num_bits {
                    let byte_idx = rng.gen_range(0..file_size as usize);
                    let bit_idx = rng.gen_range(0..8);
                    content[byte_idx] ^= 1 << bit_idx;
                }
                
                file.seek(SeekFrom::Start(0))?;
                file.write_all(&content)?;
            },
        }
        
        Ok(())
    }
    
    /// Izvede asinhron varnostni test
    pub fn run_async_test<F, Fut, R>(&self, func: F) -> R
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        self.runtime.block_on(func())
    }
    
    /// Izvede varnostni test za šifriranje
    pub fn test_encryption<F, G, H>(&self, encrypt_fn: F, decrypt_fn: G, verify_fn: H) -> SecurityTestResult
    where
        F: Fn(&[u8]) -> Result<Vec<u8>, String>,
        G: Fn(&[u8]) -> Result<Vec<u8>, String>,
        H: Fn(&[u8], &[u8]) -> bool,
    {
        let test_type = SecurityTestType::Encryption;
        let description = "Testing encryption and decryption capabilities";
        
        // Pripravimo testne podatke različnih velikosti
        let test_cases = vec![
            vec![],                        // Prazen buffer
            vec![1, 2, 3, 4, 5],           // Majhen buffer
            vec![42; 1024],                // Srednji buffer
            vec![255; 10 * 1024],          // Velik buffer
            thread_rng()                   // Naključen buffer
                .sample_iter(rand::distributions::Standard)
                .take(1024)
                .collect(),
        ];
        
        for (i, original_data) in test_cases.iter().enumerate() {
            match encrypt_fn(original_data) {
                Ok(encrypted) => {
                    match decrypt_fn(&encrypted) {
                        Ok(decrypted) => {
                            if !verify_fn(original_data, &decrypted) {
                                return SecurityTestResult::failure(
                                    test_type,
                                    description,
                                    &format!("Test case {}: Data integrity check failed after decryption", i),
                                    Some("Verify that decryption correctly reverses encryption"),
                                );
                            }
                        },
                        Err(e) => {
                            return SecurityTestResult::failure(
                                test_type,
                                description,
                                &format!("Test case {}: Failed to decrypt data: {}", i, e),
                                Some("Check decryption implementation for errors"),
                            );
                        }
                    }
                },
                Err(e) => {
                    return SecurityTestResult::failure(
                        test_type,
                        description,
                        &format!("Test case {}: Failed to encrypt data: {}", i, e),
                        Some("Check encryption implementation for errors"),
                    );
                }
            }
        }
        
        // Poskusimo dekriptirati poškodovane podatke
        if let Ok(original_encrypted) = encrypt_fn(&[1, 2, 3, 4, 5]) {
            // Poškoduj šifrirane podatke
            let mut corrupted = original_encrypted.clone();
            if !corrupted.is_empty() {
                let idx = thread_rng().gen_range(0..corrupted.len());
                corrupted[idx] = corrupted[idx].wrapping_add(1);
                
                // Dekriptiranje poškodovanih podatkov mora vrniti napako ali zagotoviti
                // integriteto (npr. z uporabo AEAD - Authenticated Encryption with Associated Data)
                match decrypt_fn(&corrupted) {
                    Ok(decrypted) => {
                        // Če je dekriptiranje uspešno, preverimo, da podatki niso poškodovani
                        if verify_fn(&[1, 2, 3, 4, 5], &decrypted) {
                            // Če so podatki enaki, je šifriranje morda pomanjkljivo v pogledu avtentikacije
                            return SecurityTestResult::failure(
                                test_type,
                                description,
                                "Decryption of corrupted data succeeded and produced apparently valid data",
                                Some("Consider using authenticated encryption (AEAD) for stronger security"),
                            );
                        }
                    },
                    Err(_) => {
                        // To je pričakovano obnašanje - dekriptiranje poškodovanih podatkov mora spodleteti
                    }
                }
            }
        }
        
        SecurityTestResult::success(
            test_type,
            description,
            "All encryption and decryption tests passed successfully",
        )
    }
    
    /// Izvede varnostni test za preverjanje vhodnih podatkov
    pub fn test_input_validation<F>(&self, validate_fn: F) -> SecurityTestResult
    where
        F: Fn(&str) -> bool,
    {
        let test_type = SecurityTestType::InputValidation;
        let description = "Testing input validation for injection and malicious inputs";
        
        // Seznam potencialno nevarnih vhodov
        let dangerous_inputs = vec![
            // SQL injection poskusi
            "' OR 1=1 --",
            "'; DROP TABLE users; --",
            // XSS poskusi
            "<script>alert('XSS')</script>",
            // Command injection poskusi
            "$(rm -rf /)",
            "`rm -rf /`",
            // Path traversal poskusi
            "../../../etc/passwd",
            // Null byte injection
            "test\0.jpg",
            // Oversized input
            &"A".repeat(10000),
            // Unicode edge cases
            "ñá",
            "你好",
            // Special characters
            "!@#$%^&*()_+<>?:\"{}|",
        ];
        
        for (i, input) in dangerous_inputs.iter().enumerate() {
            // Validacija mora biti striktna in zavrniti nevarne vhode
            if validate_fn(input) {
                return SecurityTestResult::failure(
                    test_type,
                    description,
                    &format!("Dangerous input #{} was incorrectly validated as safe: {}", i, input),
                    Some("Make input validation more strict and reject known malicious patterns"),
                );
            }
        }
        
        // Seznam veljavnih vhodov
        let valid_inputs = vec![
            "simple_text",
            "user123",
            "valid-input",
            "example.com",
            "John Doe",
        ];
        
        for (i, input) in valid_inputs.iter().enumerate() {
            // Validacija mora sprejeti veljavne vhode
            if !validate_fn(input) {
                return SecurityTestResult::failure(
                    test_type,
                    description,
                    &format!("Valid input #{} was incorrectly rejected: {}", i, input),
                    Some("Review input validation to ensure it accepts valid inputs"),
                );
            }
        }
        
        SecurityTestResult::success(
            test_type,
            description,
            "Input validation correctly identifies safe and dangerous inputs",
        )
    }
    
    /// Izvede varnostni test za časovne napade
    pub fn test_timing_attacks<F>(&self, operation_fn: F, iterations: usize) -> SecurityTestResult
    where
        F: Fn(bool) -> Result<(), String>,
    {
        let test_type = SecurityTestType::TimingAttacks;
        let description = "Testing resistance to timing attacks on sensitive operations";
        
        let mut true_durations = Vec::with_capacity(iterations);
        let mut false_durations = Vec::with_capacity(iterations);
        
        // Izmeri čas za različne vhode
        for _ in 0..iterations {
            // Meri čas za 'true' primere
            let start = Instant::now();
            if let Err(e) = operation_fn(true) {
                return SecurityTestResult::failure(
                    test_type,
                    description,
                    &format!("Operation failed for true input: {}", e),
                    Some("Fix operation function to handle true cases"),
                );
            }
            true_durations.push(start.elapsed());
            
            // Meri čas za 'false' primere
            let start = Instant::now();
            if let Err(e) = operation_fn(false) {
                return SecurityTestResult::failure(
                    test_type,
                    description,
                    &format!("Operation failed for false input: {}", e),
                    Some("Fix operation function to handle false cases"),
                );
            }
            false_durations.push(start.elapsed());
        }
        
        // Izračun statistike
        fn mean_duration(durations: &[Duration]) -> Duration {
            let sum: Duration = durations.iter().sum();
            sum / durations.len() as u32
        }
        
        fn stddev_duration(durations: &[Duration], mean: Duration) -> f64 {
            let variance: f64 = durations
                .iter()
                .map(|d| {
                    let diff = d.as_nanos() as f64 - mean.as_nanos() as f64;
                    diff * diff
                })
                .sum::<f64>() / durations.len() as f64;
            
            variance.sqrt()
        }
        
        let true_mean = mean_duration(&true_durations);
        let false_mean = mean_duration(&false_durations);
        
        let true_stddev = stddev_duration(&true_durations, true_mean);
        let false_stddev = stddev_duration(&false_durations, false_mean);
        
        // T-test za ugotavljanje statistične značilnosti
        let n1 = true_durations.len() as f64;
        let n2 = false_durations.len() as f64;
        
        let pooled_stddev = ((n1 - 1.0) * true_stddev * true_stddev + (n2 - 1.0) * false_stddev * false_stddev)
            / (n1 + n2 - 2.0);
        
        let t_statistic = (true_mean.as_nanos() as f64 - false_mean.as_nanos() as f64)
            / (pooled_stddev * (1.0 / n1 + 1.0 / n2).sqrt());
        
        let t_statistic = t_statistic.abs();
        
        // Za velike vzorce, t > 1.96 pomeni p < 0.05 (statistično značilno)
        let timing_difference_significant = t_statistic > 1.96;
        
        if timing_difference_significant {
            let which_is_faster = if true_mean < false_mean { "true" } else { "false" };
            let time_diff_percentage = 
                ((true_mean.as_nanos().max(false_mean.as_nanos()) as f64) / 
                 (true_mean.as_nanos().min(false_mean.as_nanos()) as f64) - 1.0) * 100.0;
            
            return SecurityTestResult::failure(
                test_type,
                description,
                &format!(
                    "Potential timing attack vulnerability detected. '{}' operations are {:. 2}% faster. t-statistic: {:.2}",
                    which_is_faster,
                    time_diff_percentage,
                    t_statistic
                ),
                Some("Implement constant-time operations for security-sensitive code"),
            );
        }
        
        SecurityTestResult::success(
            test_type,
            description,
            &format!(
                "No significant timing differences detected. t-statistic: {:.2} (should be < 1.96)",
                t_statistic
            ),
        )
    }
}

/// Vrste poškodbe datotek za testiranje
#[derive(Debug, Clone)]
pub enum CorruptionType {
    /// Pokvari naključne byte v datoteki
    RandomBytes {
        /// Procent bytov za poškodovanje (0-100)
        percentage: f64,
        /// Procent za začetek poškodbe (odmik od začetka)
        offset_percentage: f64,
    },
    /// Skrajšaj datoteko
    Truncate {
        /// Procent originalne velikosti za ohranitev (0-100)
        percentage: f64,
    },
    /// Postavi določen del datoteke na ničle
    ZeroOut {
        /// Procent za začetek ničel (0-100)
        start_percentage: f64,
        /// Procent za konec ničel (0-100)
        end_percentage: f64,
    },
    /// Naključno obrni določeno število bitov v datoteki
    BitFlip {
        /// Število bitov za obračanje
        num_bits: usize,
    },
}
