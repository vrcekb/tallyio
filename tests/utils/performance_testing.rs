//! Specializirano ogrodje za testiranje zmogljivosti MEV platforme
//!
//! Ta modul zagotavlja specializirane funkcije in strukture za natančno merjenje
//! in analizo zmogljivosti kritičnih delov MEV platforme, kjer je latenca
//! ključnega pomena za uspešnost operacij.

use std::time::{Duration, Instant};
use tokio::sync::Barrier;
use tokio::task;
use std::sync::Arc;
use std::future::Future;
use std::pin::Pin;
use std::collections::HashMap;

/// Treshold latence za različne tipe MEV operacij (v mikrosekundah)
pub enum LatencyClass {
    /// Ultra nizka latenca (<10μs) - za kritične odločitve in arbitražo
    Critical = 10,
    /// Zelo nizka latenca (<100μs) - za procesiranje mempool transakcij
    VeryLow = 100,
    /// Nizka latenca (<500μs) - za večino MEV operacij
    Low = 500,
    /// Srednja latenca (<1ms) - za manj časovno kritične operacije
    Medium = 1_000,
    /// Visoka latenca (<10ms) - za nekritične operacije
    High = 10_000,
}

impl LatencyClass {
    /// Vrne maksimalno dovoljeno latenco v mikrosekundah
    pub fn max_latency_us(&self) -> u64 {
        match self {
            Self::Critical => 10,
            Self::VeryLow => 100,
            Self::Low => 500,
            Self::Medium => 1_000,
            Self::High => 10_000,
        }
    }
    
    /// Vrne maksimalno dovoljeno latenco kot Duration
    pub fn as_duration(&self) -> Duration {
        Duration::from_micros(self.max_latency_us())
    }
}

/// Struktura za merjenje zmogljivosti MEV operacij
pub struct LatencyBenchmark {
    name: String,
    samples: Vec<Duration>,
    latency_class: LatencyClass,
}

impl LatencyBenchmark {
    /// Ustvari novo instanco LatencyBenchmark
    pub fn new(name: &str, latency_class: LatencyClass) -> Self {
        Self {
            name: name.to_string(),
            samples: Vec::new(),
            latency_class,
        }
    }
    
    /// Izmeri latenco funkcije in vrne rezultat
    pub fn measure<F, R>(&mut self, func: F) -> R 
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = func();
        let duration = start.elapsed();
        
        self.samples.push(duration);
        result
    }
    
    /// Izmeri latenco asinhrone funkcije in vrne rezultat
    pub async fn measure_async<F, Fut, R>(&mut self, func: F) -> R 
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = R>,
    {
        let start = Instant::now();
        let result = func().await;
        let duration = start.elapsed();
        
        self.samples.push(duration);
        result
    }
    
    /// Vrne povprečno latenco vseh meritev
    pub fn average_latency(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::default();
        }
        
        let total: Duration = self.samples.iter().sum();
        total / self.samples.len() as u32
    }
    
    /// Vrne minimalno latenco vseh meritev
    pub fn min_latency(&self) -> Duration {
        self.samples.iter().min().copied().unwrap_or_default()
    }
    
    /// Vrne maksimalno latenco vseh meritev
    pub fn max_latency(&self) -> Duration {
        self.samples.iter().max().copied().unwrap_or_default()
    }
    
    /// Preveri, ali je povprečna latenca znotraj določenega razreda
    pub fn is_within_class(&self) -> bool {
        self.average_latency() <= self.latency_class.as_duration()
    }
    
    /// Vrne 99. percentil latence
    pub fn percentile_99(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::default();
        }
        
        let mut sorted = self.samples.clone();
        sorted.sort();
        
        let idx = (sorted.len() as f64 * 0.99).ceil() as usize - 1;
        sorted.get(idx).copied().unwrap_or_default()
    }
    
    /// Izpiše statistiko latence
    pub fn print_statistics(&self) {
        println!("=== Latency Benchmark: {} ===", self.name);
        println!("Target class: {:?} (max {}μs)", self.latency_class, self.latency_class.max_latency_us());
        println!("Samples: {}", self.samples.len());
        
        if !self.samples.is_empty() {
            let avg = self.average_latency();
            let min = self.min_latency();
            let max = self.max_latency();
            let p99 = self.percentile_99();
            
            println!("Average: {:?} ({} μs)", avg, avg.as_micros());
            println!("Min: {:?} ({} μs)", min, min.as_micros());
            println!("Max: {:?} ({} μs)", max, max.as_micros());
            println!("p99: {:?} ({} μs)", p99, p99.as_micros());
            
            println!("Within class: {}", self.is_within_class());
        } else {
            println!("No samples recorded");
        }
        
        println!("=====================================");
    }
    
    /// Ustvari poročilo o zmogljivosti
    pub fn generate_report(&self) -> String {
        if self.samples.is_empty() {
            return format!("Benchmark {}: No samples recorded", self.name);
        }
        
        let avg = self.average_latency();
        let min = self.min_latency();
        let max = self.max_latency();
        let p99 = self.percentile_99();
        
        format!(
            "Benchmark: {}\n\
             Target class: {:?} (max {}μs)\n\
             Samples: {}\n\
             Average: {}μs\n\
             Min: {}μs\n\
             Max: {}μs\n\
             p99: {}μs\n\
             Within class: {}",
            self.name,
            self.latency_class,
            self.latency_class.max_latency_us(),
            self.samples.len(),
            avg.as_micros(),
            min.as_micros(),
            max.as_micros(),
            p99.as_micros(),
            self.is_within_class()
        )
    }
}

/// Struktura za testiranje zmogljivosti ob povečevanju obremenitve
pub struct LoadScalingTest<T: Clone> {
    name: String,
    loads: Vec<usize>,
    test_data: T,
}

impl<T: Clone + Send + Sync + 'static> LoadScalingTest<T> {
    /// Ustvari novo instanco LoadScalingTest
    pub fn new(name: &str, test_data: T, loads: Vec<usize>) -> Self {
        Self {
            name: name.to_string(),
            loads,
            test_data,
        }
    }
    
    /// Izvede test skalabilnosti in vrne rezultate
    pub async fn run<F, Fut>(&self, test_func: F) -> HashMap<usize, Duration>
    where
        F: Fn(T, usize) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let mut results = HashMap::new();
        let test_func = Arc::new(test_func);
        
        for &load in &self.loads {
            println!("Running {} with load {}", self.name, load);
            
            let test_data = self.test_data.clone();
            let test_func = Arc::clone(&test_func);
            
            let start = Instant::now();
            
            let barrier = Arc::new(Barrier::new(load));
            let mut handles = Vec::with_capacity(load);
            
            for i in 0..load {
                let test_data = test_data.clone();
                let test_func = Arc::clone(&test_func);
                let barrier = Arc::clone(&barrier);
                
                let handle = tokio::spawn(async move {
                    barrier.wait().await;
                    test_func(test_data, i).await;
                });
                
                handles.push(handle);
            }
            
            for handle in handles {
                let _ = handle.await;
            }
            
            let duration = start.elapsed();
            results.insert(load, duration);
            
            println!("Completed {} with load {} in {:?}", self.name, load, duration);
        }
        
        results
    }
    
    /// Izpiše rezultate testa
    pub fn print_results(&self, results: &HashMap<usize, Duration>) {
        println!("=== Load Scaling Test: {} ===", self.name);
        
        let mut loads: Vec<_> = results.keys().copied().collect();
        loads.sort();
        
        println!("{:<10} | {:<15} | {:<15}", "Load", "Total Time", "Time per Op");
        println!("{:-<10}-+-{:-<15}-+-{:-<15}", "", "", "");
        
        for load in loads {
            if let Some(&duration) = results.get(&load) {
                let per_op = if load > 0 {
                    duration / load as u32
                } else {
                    Duration::default()
                };
                
                println!(
                    "{:<10} | {:<15?} | {:<15?}",
                    load,
                    duration,
                    per_op
                );
            }
        }
        
        println!("=====================================");
    }
}

/// Struktura za primerjalno testiranje (konkurenčne implementacije)
pub struct ComparativeBenchmark {
    name: String,
    iterations: usize,
}

impl ComparativeBenchmark {
    /// Ustvari novo instanco za primerjalno testiranje
    pub fn new(name: &str, iterations: usize) -> Self {
        Self {
            name: name.to_string(),
            iterations,
        }
    }
    
    /// Izmeri in primerja dve implementaciji
    pub fn compare<F1, F2, R>(&self, name1: &str, func1: F1, name2: &str, func2: F2) -> (Duration, Duration)
    where
        F1: Fn() -> R,
        F2: Fn() -> R,
    {
        let mut total1 = Duration::default();
        let mut total2 = Duration::default();
        
        // Izvajamo izmenično, da zagotovimo podobne pogoje
        for i in 0..self.iterations {
            let start = Instant::now();
            let _result1 = func1();
            let duration1 = start.elapsed();
            total1 += duration1;
            
            let start = Instant::now();
            let _result2 = func2();
            let duration2 = start.elapsed();
            total2 += duration2;
            
            if i % 10 == 0 {
                println!("Iteration {}/{}", i + 1, self.iterations);
            }
        }
        
        let avg1 = total1 / self.iterations as u32;
        let avg2 = total2 / self.iterations as u32;
        
        println!("=== Comparative Benchmark: {} ===", self.name);
        println!("Iterations: {}", self.iterations);
        println!("{}: avg {:?}", name1, avg1);
        println!("{}: avg {:?}", name2, avg2);
        
        if avg1 < avg2 {
            let speedup = avg2.as_nanos() as f64 / avg1.as_nanos() as f64;
            println!("{} is {:.2}x faster than {}", name1, speedup, name2);
        } else if avg2 < avg1 {
            let speedup = avg1.as_nanos() as f64 / avg2.as_nanos() as f64;
            println!("{} is {:.2}x faster than {}", name2, speedup, name1);
        } else {
            println!("Both implementations have equal performance");
        }
        
        println!("=====================================");
        
        (avg1, avg2)
    }
    
    /// Izmeri in primerja več implementacij
    pub fn compare_many<F, R>(&self, implementations: Vec<(&str, F)>) -> Vec<(&str, Duration)>
    where
        F: Fn() -> R,
    {
        if implementations.is_empty() {
            return Vec::new();
        }
        
        let mut results = Vec::with_capacity(implementations.len());
        
        for (name, func) in &implementations {
            let mut total = Duration::default();
            
            for i in 0..self.iterations {
                let start = Instant::now();
                let _result = func();
                let duration = start.elapsed();
                total += duration;
                
                if i % 10 == 0 {
                    println!("Testing {} - Iteration {}/{}", name, i + 1, self.iterations);
                }
            }
            
            let avg = total / self.iterations as u32;
            results.push((*name, avg));
        }
        
        // Sortiramo po času (najhitrejši prvi)
        results.sort_by_key(|&(_, duration)| duration);
        
        println!("=== Multi-implementation Benchmark: {} ===", self.name);
        println!("Iterations per implementation: {}", self.iterations);
        println!("{:<30} | {:<15}", "Implementation", "Avg Time");
        println!("{:-<30}-+-{:-<15}", "", "");
        
        for (name, duration) in &results {
            println!("{:<30} | {:<15?}", name, duration);
        }
        
        // Prikažemo relativno hitrost glede na najhitrejšo implementacijo
        if !results.is_empty() {
            let (fastest_name, fastest_time) = results[0];
            println!("\nRelative Performance (compared to fastest):");
            
            for (name, duration) in &results {
                if *name != fastest_name {
                    let slowdown = duration.as_nanos() as f64 / fastest_time.as_nanos() as f64;
                    println!("{:<30} | {:.2}x slower", name, slowdown);
                } else {
                    println!("{:<30} | baseline (fastest)", name);
                }
            }
        }
        
        println!("=====================================");
        
        results
    }
}
