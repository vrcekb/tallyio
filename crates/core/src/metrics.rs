//! Metrike in sledenje za core modul

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Metrike za sledenje performanci
#[derive(Debug, Default)]
pub struct Metrics {
    /// Število uspešnih operacij
    success_count: AtomicU64,
    /// Število napak
    error_count: AtomicU64,
    /// Skupni čas izvajanja (v mikrosekundah)
    total_time_us: AtomicU64,
}

impl Metrics {
    /// Ustvari nove metrike
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Začni merjenje operacije
    #[inline]
    pub fn start_operation(&self) -> MetricsGuard {
        MetricsGuard { metrics: self, start: Instant::now() }
    }

    /// Ročno dodaj uspešno operacijo
    #[inline]
    pub fn add_success(&self, duration: Duration) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.total_time_us
            .fetch_add(u64::try_from(duration.as_micros()).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    /// Ročno dodaj neuspešno operacijo
    #[inline]
    pub fn add_error(&self, duration: Duration) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
        self.total_time_us
            .fetch_add(u64::try_from(duration.as_micros()).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    /// Vrni število uspešnih operacij
    #[inline]
    pub fn success_count(&self) -> u64 {
        self.success_count.load(Ordering::Relaxed)
    }

    /// Vrni število napak
    #[inline]
    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Vrni skupni čas izvajanja
    #[inline]
    pub fn total_time(&self) -> Duration {
        Duration::from_micros(self.total_time_us.load(Ordering::Relaxed))
    }

    /// Vrni povprečni čas izvajanja
    #[inline]
    pub fn average_time(&self) -> Duration {
        let total = self.success_count() + self.error_count();
        if total == 0 {
            Duration::from_micros(0)
        } else {
            Duration::from_micros(self.total_time_us.load(Ordering::Relaxed) / total)
        }
    }
}

/// RAII guard za avtomatsko merjenje časa operacije
#[derive(Debug)]
pub struct MetricsGuard<'a> {
    metrics: &'a Metrics,
    start: Instant,
}

impl MetricsGuard<'_> {
    /// Označi operacijo kot uspešno
    #[inline]
    pub fn success(self) {
        let duration = self.start.elapsed();
        self.metrics.add_success(duration);
    }

    /// Označi operacijo kot neuspešno
    #[inline]
    pub fn error(self) {
        let duration = self.start.elapsed();
        self.metrics.add_error(duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_metrics_success() {
        let metrics = Metrics::new();
        assert_eq!(metrics.success_count(), 0);
        assert_eq!(metrics.error_count(), 0);

        metrics.start_operation().success();
        assert_eq!(metrics.success_count(), 1);
        assert_eq!(metrics.error_count(), 0);
    }

    #[test]
    fn test_metrics_error() {
        let metrics = Metrics::new();
        assert_eq!(metrics.success_count(), 0);
        assert_eq!(metrics.error_count(), 0);

        metrics.start_operation().error();
        assert_eq!(metrics.success_count(), 0);
        assert_eq!(metrics.error_count(), 1);
    }

    #[test]
    fn test_metrics_average() {
        let metrics = Metrics::new();
        let duration1 = Duration::from_micros(100);
        let duration2 = Duration::from_micros(300);

        // Dodamo dve uspešni operaciji
        metrics.add_success(duration1);
        metrics.add_success(duration2);

        // Preverimo povprečje (200µs)
        assert_eq!(metrics.success_count(), 2);
        assert_eq!(metrics.total_time(), Duration::from_micros(400));
        assert_eq!(metrics.average_time(), Duration::from_micros(200));
    }

    #[test]
    fn test_metrics_average_zero_operations() {
        // Test za primer, ko je skupno število operacij 0
        let metrics = Metrics::new();

        // Preverimo, da je povprečni čas 0, ko ni operacij
        assert_eq!(metrics.success_count(), 0);
        assert_eq!(metrics.error_count(), 0);
        assert_eq!(metrics.average_time(), Duration::from_micros(0));
    }

    #[test]
    fn test_metrics_concurrent() {
        let metrics = Arc::new(Metrics::new());
        let mut handles = vec![];

        // Ustvarimo 10 niti, vsaka izvede 100 uspešnih operacij
        for _ in 0..10 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    metrics_clone.add_success(Duration::from_micros(10));
                }
            });
            handles.push(handle);
        }

        // Ustvarimo še 5 niti, vsaka izvede 20 napak
        for _ in 0..5 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..20 {
                    metrics_clone.add_error(Duration::from_micros(50));
                }
            });
            handles.push(handle);
        }

        // Počakamo, da vse niti končajo
        for handle in handles {
            handle.join().expect("Thread failed");
        }

        // Preverimo rezultate
        assert_eq!(metrics.success_count(), 1000); // 10 niti * 100 operacij
        assert_eq!(metrics.error_count(), 100); // 5 niti * 20 operacij

        // Preverimo, da je skupni čas večji od 0
        let total_time = metrics.total_time();
        assert!(
            total_time.as_millis() > 0,
            "Skupni čas mora biti večji od 0, dobili smo: {total_time:?}"
        );
    }

    // Stresni testi - označeni z #[ignore]

    /// Stresni test za veliko število operacij
    #[test]
    #[ignore]
    fn stress_high_volume_operations() {
        let metrics = Arc::new(Metrics::new());
        let mut handles = vec![];

        // Veliko število niti in operacij za obremenitveno testiranje
        // 50 niti, vsaka s 10,000 operacijami
        for i in 0..50 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for j in 0..10_000 {
                    if (i + j) % 5 == 0 {
                        metrics_clone.add_error(Duration::from_nanos(50));
                    } else {
                        metrics_clone.add_success(Duration::from_nanos(20));
                    }
                }
            });
            handles.push(handle);
        }

        // Počakamo, da vse niti končajo
        for handle in handles {
            if let Err(e) = handle.join() {
                panic!("Thread panicked: {e:?}");
            }
        }

        // Skupno število operacij bi moralo biti 50 * 10,000 = 500,000
        let total_ops = metrics.success_count() + metrics.error_count();
        assert_eq!(total_ops, 500_000);

        // Razmerje med uspehom in napakami bi moralo biti približno 4:1
        // (ker vsaka peta operacija je napaka)
        let expected_error_ratio = 0.2;
        // Za u64 -> f64 pretvorbo moramo uporabiti 'as', ker From/Into ni implementiran
        // To je varno, ker f64 lahko predstavi vse vrednosti u64 do 2^53, kar je dovolj za naš primer
        #[allow(clippy::cast_precision_loss)]
        let actual_error_ratio = metrics.error_count() as f64 / total_ops as f64;
        assert!((actual_error_ratio - expected_error_ratio).abs() < 0.01);
    }

    /// Stresni test za preverjanje obnašanja pri dolgotrajnih operacijah
    #[test]
    #[ignore]
    fn stress_long_duration_operations() {
        let metrics = Metrics::new();

        // Dolge operacije (simuliramo operacije z dolgimi časi izvajanja)
        let very_long_duration = Duration::from_secs(3600); // 1 ura

        for _ in 0..10 {
            metrics.add_success(very_long_duration);
        }

        // Skupni čas bi moral biti 10 ur = 36_000 sekund = 36_000_000_000 mikrosekund
        let expected_micros = 36_000_000_000;
        let total_micros = metrics.total_time().as_micros();

        // Preverimo, da se je pravilno beležil čas (ob predpostavki, da ni prekoračitve)
        if u128::from(u64::MAX) >= expected_micros {
            assert_eq!(total_micros, expected_micros);
        } else {
            // V primeru prekoračitve bi morala biti vrednost u64::MAX
            assert_eq!(metrics.total_time().as_micros(), u128::from(u64::MAX));
        }
    }

    /// Stresni test za sočasno dodajanje in branje metrik
    #[test]
    #[ignore]
    fn stress_concurrent_read_write() {
        let metrics = Arc::new(Metrics::new());
        let mut handles = vec![];

        // Niti za pisanje (20 niti)
        for _ in 0..20 {
            let m = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..5000 {
                    m.add_success(Duration::from_micros(1));
                }
            });
            handles.push(handle);
        }

        // Niti za branje (10 niti)
        for _ in 0..10 {
            let m = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    let _ = m.success_count();
                    let _ = m.error_count();
                    let _ = m.total_time();
                    let _ = m.average_time();
                    // Kratka pavza za povečanje možnosti prepletanja operacij
                    thread::yield_now();
                }
            });
            handles.push(handle);
        }

        // Počakamo, da vse niti končajo
        for handle in handles {
            if let Err(e) = handle.join() {
                panic!("Thread panicked: {e:?}");
            }
        }

        // Preverimo končno število uspešnih operacij (20 niti * 5000 operacij)
        assert_eq!(metrics.success_count(), 100_000);
    }
}
