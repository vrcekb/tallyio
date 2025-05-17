//! Tests za core metrics modul
//! 
//! Ti testi zagotavljajo pravilno delovanje metrik, ki so kritične za
//! spremljanje performanc v MEV sistemu in odkrivanje ozkih grl.

use core::metrics::{Metrics, MetricsGuard};
use std::time::{Duration, Instant};
use std::sync::{Arc, Barrier};
use std::thread;

/// Test direktnega dodajanja metrik brez uporabe guard-a
#[test]
fn test_manual_metrics_addition() {
    let metrics = Metrics::new();
    
    // Testiramo direktno dodajanje
    metrics.add_success(Duration::from_millis(50));
    metrics.add_error(Duration::from_millis(100));
    
    assert_eq!(metrics.success_count(), 1);
    assert_eq!(metrics.error_count(), 1);
    
    // Preverimo tudi povprečja
    assert_eq!(metrics.avg_success_time().as_millis(), 50);
    assert_eq!(metrics.avg_error_time().as_millis(), 100);
}

/// Test za prevelike vrednosti trajanja
#[test]
fn test_duration_overflow() {
    let metrics = Metrics::new();
    
    // Dodamo enormne vrednosti, ki bi lahko povzročile overflow
    for _ in 0..10 {
        metrics.add_success(Duration::from_secs(u64::MAX / 20));
    }
    
    // Preverimo, da ni prišlo do overflow-a
    assert!(metrics.avg_success_time() > Duration::from_secs(u64::MAX / 30));
    assert_eq!(metrics.success_count(), 10);
}

/// Test za kombinacije success/error dodajanj in skupni čas
#[test]
fn test_combined_metrics() {
    let metrics = Metrics::new();
    
    // Testiramo mešane operacije
    for i in 0..5 {
        // Izmenjujemo success in error
        if i % 2 == 0 {
            metrics.add_success(Duration::from_millis(100));
        } else {
            metrics.add_error(Duration::from_millis(200));
        }
    }
    
    // Preverimo rezultate
    assert_eq!(metrics.success_count(), 3);
    assert_eq!(metrics.error_count(), 2);
    assert_eq!(metrics.total_count(), 5);
    
    // Preverimo povprečja
    assert_eq!(metrics.avg_success_time().as_millis(), 100);
    assert_eq!(metrics.avg_error_time().as_millis(), 200);
    
    // Preverimo razmerje uspešnosti
    assert_eq!(metrics.success_ratio(), 0.6); // 3/5 = 0.6
    
    // Preverimo skupni čas
    assert_eq!(metrics.total_time(), 
              Duration::from_millis(3 * 100 + 2 * 200)); // 3*100 + 2*200 = 700
}

/// Test za obravnavanje MetricsGuard
#[test]
fn test_metrics_guard_lifecycle() {
    let metrics = Arc::new(Metrics::new());
    
    // Uporabimo metrics guard za avtomatsko beleženje
    {
        let guard = MetricsGuard::new(Arc::clone(&metrics));
        
        // Simuliramo neko operacijo
        std::thread::sleep(Duration::from_millis(10));
        
        // Označimo kot uspešno
        guard.success();
    } // guard se sprosti tukaj in zabeleži trajanje
    
    assert_eq!(metrics.success_count(), 1);
    assert_eq!(metrics.error_count(), 0);
    assert!(metrics.avg_success_time().as_millis() >= 10);
    
    // Testiramo še error primer
    {
        let guard = MetricsGuard::new(Arc::clone(&metrics));
        
        // Simuliramo neko operacijo
        std::thread::sleep(Duration::from_millis(15));
        
        // Označimo kot napako
        guard.error();
    } // guard se sprosti tukaj in zabeleži trajanje
    
    assert_eq!(metrics.success_count(), 1);
    assert_eq!(metrics.error_count(), 1);
    assert!(metrics.avg_error_time().as_millis() >= 15);
}

/// Test za sočasno branje in pisanje različnih metrik
#[test]
fn test_mixed_concurrent_operations() {
    let metrics = Arc::new(Metrics::new());
    let threads_count = 8;
    let operations_per_thread = 1000;
    
    // Uporabimo barrier za sinhronizacijo začetka vseh niti
    let barrier = Arc::new(Barrier::new(threads_count));
    
    let mut handles = Vec::new();
    
    // Zaženemo več niti, ki bodo pisale in brale metrike
    for i in 0..threads_count {
        let metrics_clone = Arc::clone(&metrics);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            // Počakamo, da so vse niti pripravljene
            barrier_clone.wait();
            
            for j in 0..operations_per_thread {
                let value = i * operations_per_thread + j;
                
                // Glede na vrednost izvedemo različne operacije
                match value % 4 {
                    0 => {
                        // Dodamo success
                        metrics_clone.add_success(Duration::from_nanos(value as u64));
                    },
                    1 => {
                        // Dodamo error
                        metrics_clone.add_error(Duration::from_nanos(value as u64));
                    },
                    2 => {
                        // Preberemo success statistics
                        let _ = metrics_clone.success_count();
                        let _ = metrics_clone.avg_success_time();
                    },
                    3 => {
                        // Preberemo error statistics
                        let _ = metrics_clone.error_count();
                        let _ = metrics_clone.avg_error_time();
                    },
                    _ => unreachable!(),
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Počakamo, da se vse niti zaključijo
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Preverimo, da se število operacij ujema
    let expected_success = threads_count * operations_per_thread / 4;
    let expected_error = threads_count * operations_per_thread / 4;
    
    assert_eq!(metrics.success_count(), expected_success);
    assert_eq!(metrics.error_count(), expected_error);
    assert_eq!(metrics.total_count(), expected_success + expected_error);
}

/// Test za izračun povprečja z mešanimi operacijami
#[test]
fn test_average_with_mixed_operations() {
    let metrics = Metrics::new();
    
    // Dodamo nekaj operacij z znanimi časi
    metrics.add_success(Duration::from_millis(10));
    metrics.add_success(Duration::from_millis(20));
    metrics.add_success(Duration::from_millis(30));
    
    metrics.add_error(Duration::from_millis(100));
    metrics.add_error(Duration::from_millis(200));
    
    // Preverimo povprečja
    assert_eq!(metrics.avg_success_time().as_millis(), 20); // (10+20+30)/3 = 20
    assert_eq!(metrics.avg_error_time().as_millis(), 150);  // (100+200)/2 = 150
}
