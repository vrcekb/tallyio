// Benchmark testi za komponente z nizko latenco
//
// Ta benchmark meri latenco kritičnih operacij v core modulu za zagotavljanje
// sub-milisekundne izvedbe, kar je ključno za učinkovito MEV delovanje.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use core::{Arena, Queue};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// Benchmark Arena alokacijske performanse
fn bench_arena_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("Arena Allocation");
    
    // Benchmark posamezne alokacije
    group.bench_function("single_alloc", |b| {
        let arena = Arena::new();
        b.iter(|| {
            black_box(arena.alloc(black_box(42)));
        });
    });
    
    // Benchmark več alokacij
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("multiple_allocs", size), size, |b, &size| {
            let arena = Arena::new();
            b.iter(|| {
                for i in 0..size {
                    black_box(arena.alloc(black_box(i)));
                }
            });
        });
    }
    
    group.finish();
}

// Benchmark performanse Queue
fn bench_queue_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Queue Operations");
    
    // Benchmark push operacije
    group.bench_function("push", |b| {
        let queue = Queue::new();
        b.iter(|| {
            queue.push(black_box(42));
        });
    });
    
    // Benchmark pop operacije
    group.bench_function("pop", |b| {
        let queue = Queue::new();
        // Najprej dodaj nekaj elementov
        for i in 0..100 {
            queue.push(i);
        }
        b.iter(|| {
            black_box(queue.pop());
        });
    });
    
    // Benchmark kombinirane operacije (push-pop cikel)
    group.bench_function("push_pop_cycle", |b| {
        let queue = Queue::new();
        b.iter(|| {
            queue.push(black_box(42));
            black_box(queue.pop());
        });
    });
    
    // Benchmark konkurenčnih operacij
    group.bench_function("concurrent_operations", |b| {
        b.iter(|| {
            let queue = Arc::new(Queue::new());
            
            // Ustvari več niti, ki opravljajo push/pop operacije
            let mut handles = Vec::new();
            
            for i in 0..4 {
                let queue_clone = Arc::clone(&queue);
                let handle = thread::spawn(move || {
                    for j in 0..100 {
                        queue_clone.push(i * 1000 + j);
                        let _ = queue_clone.pop();
                    }
                });
                handles.push(handle);
            }
            
            // Počakaj, da se vse niti zaključijo
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// Preveri, da kritične operacije izpolnjujejo zahtevo po latenci (<1ms)
fn verify_latency_requirements() {
    println!("Preverjanje latence za kritične MEV operacije...");
    
    // Test Arena alokacije
    {
        let arena = Arena::new();
        let start = Instant::now();
        for _ in 0..1000 {
            arena.alloc(42);
        }
        let duration = start.elapsed();
        println!("Arena 1000 alokacij: {:?} (povpr.: {:?} na alokacijo)", 
                 duration, duration / 1000);
        assert!(duration / 1000 < Duration::from_micros(10), 
                "Arena alokacija presega zahtevo po latenci");
    }
    
    // Test Queue operacij
    {
        let queue = Queue::new();
        
        // Push latenca
        let start = Instant::now();
        for i in 0..1000 {
            queue.push(i);
        }
        let duration = start.elapsed();
        println!("Queue 1000 push operacij: {:?} (povpr.: {:?} na operacijo)", 
                 duration, duration / 1000);
        assert!(duration / 1000 < Duration::from_micros(10), 
                "Queue push presega zahtevo po latenci");
        
        // Pop latenca
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = queue.pop();
        }
        let duration = start.elapsed();
        println!("Queue 1000 pop operacij: {:?} (povpr.: {:?} na operacijo)", 
                 duration, duration / 1000);
        assert!(duration / 1000 < Duration::from_micros(10), 
                "Queue pop presega zahtevo po latenci");
    }
    
    println!("Vsi testi latence uspešno zaključeni!");
}

criterion_group!(benches, bench_arena_alloc, bench_queue_operations);
criterion_main!(benches);

// Zaženi to funkcijo za preverjanje zahtev po latenci
#[test]
fn test_latency_requirements() {
    verify_latency_requirements();
}
