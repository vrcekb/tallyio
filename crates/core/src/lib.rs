//! Jedro sistema `TallyIO`
//!
//! Core modul implementira križniščne komponente sistema:
//! - Arena alokator za učinkovito upravljanje s spominom
//! - Lock-free podatkovne strukture za visoko prepustnost
//! - Error handling tipe in trait-e
//! - Metrike in sledenje

use bumpalo::Bump;
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicU64, Ordering};

mod error;
mod metrics;
mod types;

pub use error::CoreError;
pub use metrics::Metrics;
pub use types::*;

#[cfg(test)]
mod metrics_coverage_tests {
    use crate::metrics::Metrics;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Test direktnega dodajanja metrik brez uporabe guard-a
    #[test]
    fn test_manual_metrics_addition() {
        let metrics = Metrics::new();

        // Testiramo direktno dodajanje
        metrics.add_success(Duration::from_millis(50));
        metrics.add_error(Duration::from_millis(100));

        assert_eq!(metrics.success_count(), 1);
        assert_eq!(metrics.error_count(), 1);
        assert_eq!(metrics.total_time(), Duration::from_micros(150_000));
        assert_eq!(metrics.average_time(), Duration::from_micros(75_000));
    }

    /// Test za prevelike vrednosti trajanja
    #[test]
    fn test_duration_overflow() {
        let metrics = Metrics::new();

        // Testiramo dodajanje z zelo veliko vrednostjo, ki bi povzročila overflow pri pretvorbi v u64
        let huge_duration = Duration::from_secs(u64::MAX);
        metrics.add_success(huge_duration);

        // Preverimo, da je vrednost bila pravilno omejena na u64::MAX
        assert_eq!(metrics.total_time(), Duration::from_micros(u64::MAX));
    }

    /// Test za kombinacije success/error dodajanj in skupni čas
    #[test]
    fn test_combined_metrics() {
        let metrics = Metrics::new();

        // Dodamo uspešno operacijo z ročnim klicem
        metrics.add_success(Duration::from_micros(100));

        // Dodamo uspešno operacijo z guard
        let guard = metrics.start_operation();
        thread::sleep(Duration::from_millis(1));
        guard.success();

        // Dodamo napako z ročnim klicem
        metrics.add_error(Duration::from_micros(300));

        // Dodamo napako z guard
        let guard = metrics.start_operation();
        thread::sleep(Duration::from_millis(1));
        guard.error();

        // Preverimo vrednosti
        assert_eq!(metrics.success_count(), 2);
        assert_eq!(metrics.error_count(), 2);

        // Skupni čas mora biti vsaj 402 mikrosekund (100 + ~1ms + 300 + ~1ms)
        let total_time = metrics.total_time();
        assert!(
            total_time.as_micros() >= 402,
            "Skupni čas mora biti vsaj 402 µs, dobili smo: {}µs",
            total_time.as_micros()
        );
    }

    /// Test za obravnavanje `MetricsGuard`
    #[test]
    fn test_metrics_guard_lifecycle() {
        let metrics = Metrics::new();

        // Testiramo, da guard pravilno meri čas
        let _start = Instant::now();
        let guard = metrics.start_operation();

        // Počakamo malo
        thread::sleep(Duration::from_millis(5));

        // Končamo operacijo
        guard.success();

        // Preverimo, da je časovna razlika smiselna (vsaj 5ms)
        assert!(
            metrics.total_time().as_millis() >= 5,
            "Metrics guard mora izmeriti vsaj 5ms, dobili smo: {}ms",
            metrics.total_time().as_millis()
        );
    }

    /// Test za sočasno branje in pisanje različnih metrik
    #[test]
    fn test_mixed_concurrent_operations() {
        let metrics = Arc::new(Metrics::new());
        let barrier = Arc::new(Barrier::new(3));

        // Nit, ki dodaja uspešne operacije
        let metrics_clone = Arc::clone(&metrics);
        let barrier_clone = Arc::clone(&barrier);
        let success_handle = thread::spawn(move || {
            barrier_clone.wait();
            for i in 0..50 {
                // Alterniramo med direktnim dodajanjem in guard
                if i % 2 == 0 {
                    metrics_clone.add_success(Duration::from_micros(10));
                } else {
                    let guard = metrics_clone.start_operation();
                    thread::sleep(Duration::from_micros(10));
                    guard.success();
                }
            }
        });

        // Nit, ki dodaja napake
        let metrics_clone = Arc::clone(&metrics);
        let barrier_clone = Arc::clone(&barrier);
        let error_handle = thread::spawn(move || {
            barrier_clone.wait();
            for i in 0..30 {
                // Alterniramo med direktnim dodajanjem in guard
                if i % 2 == 0 {
                    metrics_clone.add_error(Duration::from_micros(20));
                } else {
                    let guard = metrics_clone.start_operation();
                    thread::sleep(Duration::from_micros(20));
                    guard.error();
                }
            }
        });

        // Nit, ki bere metrike med dodajanjem
        let metrics_clone = Arc::clone(&metrics);
        let barrier_clone = Arc::clone(&barrier);
        let read_handle = thread::spawn(move || {
            barrier_clone.wait();
            for _ in 0..100 {
                let _ = metrics_clone.success_count();
                let _ = metrics_clone.error_count();
                let _ = metrics_clone.average_time();
                let _ = metrics_clone.total_time();
                thread::sleep(Duration::from_micros(5));
            }
        });

        // Počakamo, da vse niti končajo
        success_handle.join().expect("Success thread panicked");
        error_handle.join().expect("Error thread panicked");
        read_handle.join().expect("Read thread panicked");

        // Preverimo končne vrednosti
        assert_eq!(metrics.success_count(), 50);
        assert_eq!(metrics.error_count(), 30);

        // Skupni čas mora biti pozitiven
        assert!(
            metrics.total_time().as_micros() > 0,
            "Skupni čas mora biti večji od 0, dobili smo: {}µs",
            metrics.total_time().as_micros()
        );
    }

    /// Test za izračun povprečja z mešanimi operacijami
    #[test]
    fn test_average_with_mixed_operations() {
        let metrics = Metrics::new();

        // Dodamo različne operacije z različnimi trajanji
        metrics.add_success(Duration::from_micros(100));
        metrics.add_success(Duration::from_micros(200));
        metrics.add_error(Duration::from_micros(300));
        metrics.add_error(Duration::from_micros(400));

        // Preverimo povprečje (100+200+300+400)/4 = 250µs
        assert_eq!(metrics.average_time(), Duration::from_micros(250));
    }
}

/// Arena alokator za učinkovito alokacijo spomina
///
/// Arena omogoča učinkovito alokacijo spomina brez potrebe po eksplicitnem sproščanju.
/// Vsi objekti, alocirani v areni, so veljavni dokler arena obstaja.
///
/// # Primer uporabe
///
/// ```
/// use core::Arena;
///
/// // Ustvari novo areno
/// let arena = Arena::new();
///
/// // Alociraj vrednosti v areni
/// let value1 = arena.alloc(42);
/// let value2 = arena.alloc("Hello");
///
/// // Vrednosti so dostopne dokler arena obstaja
/// assert_eq!(*value1, 42);
/// assert_eq!(*value2, "Hello");
///
/// // Preveri število alokacij
/// assert_eq!(arena.allocation_count(), 2);
/// ```
#[derive(Debug)]
pub struct Arena {
    /// Bumpalo arena za zero-copy alokacije
    inner: Bump,
    /// Števec alokacij
    allocs: AtomicU64,
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl Arena {
    /// Ustvari novo areno
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self { inner: Bump::new(), allocs: AtomicU64::new(0) }
    }

    /// Alocira nov objekt v areni
    ///
    /// # Performance
    /// - Zero-allocation design
    /// - O(1) časovna kompleksnost
    /// - Thread-safe
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &T {
        self.allocs.fetch_add(1, Ordering::Relaxed);
        self.inner.alloc(value)
    }

    /// Vrne število alokacij
    #[inline]
    pub fn allocation_count(&self) -> u64 {
        self.allocs.load(Ordering::Relaxed)
    }
}

/// Lock-free vrsta za visoko prepustnost
///
/// Implementira lock-free FIFO vrsto, ki je varna za uporabo v večnitnem okolju.
/// Omogoča visoko prepustnost in nizko latenco pri sočasnem dostopu.
///
/// # Primer uporabe
///
/// ```
/// use core::Queue;
/// use std::sync::Arc;
/// use std::thread;
///
/// // Ustvari novo vrsto
/// let queue = Arc::new(Queue::<i32>::new());
/// let queue_clone = Arc::clone(&queue);
///
/// // Ustvari nit, ki dodaja elemente v vrsto
/// let producer = thread::spawn(move || {
///     for i in 0..5 {
///         queue_clone.push(i);
///     }
/// });
///
/// // Počakaj, da producer konča
/// producer.join().unwrap();
///
/// // Preberi elemente iz vrste
/// for i in 0..5 {
///     assert_eq!(queue.pop(), Some(i));
/// }
///
/// // Vrsta je zdaj prazna
/// assert_eq!(queue.pop(), None);
/// ```
#[derive(Debug, Default)]
pub struct Queue<T> {
    inner: SegQueue<T>,
}

impl<T> Queue<T> {
    /// Ustvari novo vrsto
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { inner: SegQueue::new() }
    }

    /// Doda element v vrsto
    ///
    /// # Performance
    /// - Lock-free implementacija
    /// - O(1) časovna kompleksnost
    #[inline]
    pub fn push(&self, value: T) {
        self.inner.push(value);
    }

    /// Vzame element iz vrste
    ///
    /// # Performance
    /// - Lock-free implementacija
    /// - O(1) časovna kompleksnost
    #[inline]
    pub fn pop(&self) -> Option<T> {
        self.inner.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Instant;

    // Unit tests
    #[test]
    fn test_arena_alloc() {
        let arena = Arena::new();
        let start = Instant::now();
        let value = arena.alloc(42);
        let elapsed = start.elapsed();

        assert_eq!(*value, 42);
        assert_eq!(arena.allocation_count(), 1);
        assert!(elapsed.as_micros() < 500, "Previsoka latenca: {elapsed:?}");
    }

    #[test]
    fn test_arena_multiple_allocs() {
        let arena = Arena::new();
        let values: Vec<_> = (0..1000).map(|i| arena.alloc(i)).collect();

        for (i, &value) in values.iter().enumerate() {
            assert_eq!(*value, i);
        }
        assert_eq!(arena.allocation_count(), 1000);
    }

    #[test]
    fn test_arena_default() {
        // Test za Default implementacijo za Arena
        let arena = Arena::default();
        let value = arena.alloc(42);

        assert_eq!(*value, 42);
        assert_eq!(arena.allocation_count(), 1);
    }

    #[test]
    fn test_queue_operations() {
        let queue = Queue::new();
        let start = Instant::now();
        queue.push(42);
        let value = queue.pop();
        let elapsed = start.elapsed();

        assert_eq!(value, Some(42));
        assert!(elapsed.as_micros() < 500, "Previsoka latenca: {elapsed:?}");
    }

    #[test]
    fn test_queue_fifo() {
        let queue = Queue::new();
        for i in 0..100 {
            queue.push(i);
        }

        for i in 0..100 {
            assert_eq!(queue.pop(), Some(i));
        }
        assert_eq!(queue.pop(), None);
    }

    // Panic test example
    #[test]
    #[should_panic(expected = "Poskus dostopa do neveljavnega indeksa")]
    fn test_panic_on_invalid_access() {
        struct UnsafeAccess {
            data: Vec<i32>,
        }

        impl UnsafeAccess {
            fn get(&self, index: usize) -> i32 {
                assert!(index < self.data.len(), "Poskus dostopa do neveljavnega indeksa");
                self.data[index]
            }
        }

        let access = UnsafeAccess { data: vec![1, 2, 3] };
        access.get(5); // This should panic
    }

    // Stress test example
    #[test]
    #[ignore]
    fn stress_queue_high_load() {
        use std::sync::Arc;
        use std::thread;

        let queue = Arc::new(Queue::new());
        let mut handles = vec![];

        // Create 100 producer threads
        for i in 0..100 {
            let queue = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for j in 0..1000 {
                    queue.push(i * 1000 + j);
                }
            }));
        }

        // Create 100 consumer threads
        let received = Arc::new(AtomicU64::new(0));
        for _ in 0..100 {
            let queue = Arc::clone(&queue);
            let received = Arc::clone(&received);
            handles.push(thread::spawn(move || {
                let mut count = 0;
                while count < 1000 {
                    if queue.pop().is_some() {
                        count += 1;
                    }
                }
                received.fetch_add(count, Ordering::Relaxed);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(received.load(Ordering::Relaxed), 100_000);
    }

    // Security test example
    #[test]
    #[ignore]
    fn security_test_memory_safety() {
        // Test that our Arena doesn't allow use-after-free
        let arena = Arena::new();
        let value_ref = arena.alloc(42);

        // Verify that the value is still accessible and correct
        assert_eq!(*value_ref, 42);

        // In an unsafe implementation, this might cause a use-after-free
        // But our implementation should be safe
        let _new_arena = Arena::new();

        // This should still be valid
        assert_eq!(*value_ref, 42);
    }

    #[test]
    fn test_queue_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let queue = Arc::new(Queue::new());
        let mut handles = vec![];

        // Producer threads
        for i in 0..10 {
            let queue = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    queue.push(i * 100 + j);
                }
            }));
        }

        // Consumer threads
        let received = Arc::new(AtomicU64::new(0));
        for _ in 0..10 {
            let queue = Arc::clone(&queue);
            let received = Arc::clone(&received);
            handles.push(thread::spawn(move || {
                while received.load(Ordering::Relaxed) < 1000 {
                    if queue.pop().is_some() {
                        received.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(received.load(Ordering::Relaxed), 1000);
    }
}

// Property tests
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn arena_allocations_are_correct(values in proptest::collection::vec(0..100i32, 1..100)) {
            let arena = Arena::new();
            let allocated: Vec<_> = values.iter().map(|&v| arena.alloc(v)).collect();

            // Check that all allocated values match the original values
            for (i, &value_ref) in allocated.iter().enumerate() {
                prop_assert_eq!(*value_ref, values[i]);
            }

            // Check that the allocation count is correct
            // Pretvorba iz u64 v usize je varna na 64-bitnih sistemih, na 32-bitnih pa
            // bi lahko prišlo do izgube podatkov, če bi imeli več kot 2^32-1 alokacij
            // Ker pa je to testni primer z majhnim številom alokacij, je pretvorba varna
            let count: usize = arena.allocation_count().try_into()
                .expect("Število alokacij presega kapaciteto usize");
            prop_assert_eq!(count, values.len());
        }

        #[test]
        fn queue_preserves_fifo_order(values in proptest::collection::vec(0..100i32, 1..100)) {
            let queue = Queue::new();

            // Push all values
            for &value in &values {
                queue.push(value);
            }

            // Pop all values and check they come out in the same order
            for &expected in &values {
                let actual = queue.pop();
                prop_assert_eq!(actual, Some(expected));
            }

            // Queue should be empty now
            prop_assert_eq!(queue.pop(), None);
        }
    }
}
