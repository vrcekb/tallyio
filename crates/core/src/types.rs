//! Osnovni tipi za core modul

/// Rezultat operacije z metriko latence
#[derive(Debug)]
pub struct TimedResult<T> {
    /// Rezultat operacije
    pub result: T,
    /// Latenca operacije
    pub latency: std::time::Duration,
}

impl<T> TimedResult<T> {
    /// Ustvari nov časovno merjen rezultat
    #[inline]
    #[must_use]
    pub const fn new(result: T, latency: std::time::Duration) -> Self {
        Self { result, latency }
    }

    /// Preslika vrednost rezultata
    #[inline]
    pub fn map<U, F>(self, f: F) -> TimedResult<U>
    where
        F: FnOnce(T) -> U,
    {
        TimedResult { result: f(self.result), latency: self.latency }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_timed_result() {
        let tr = TimedResult::new(42, Duration::from_micros(100));
        assert_eq!(tr.result, 42);
        assert_eq!(tr.latency, Duration::from_micros(100));
    }

    #[test]
    fn test_timed_result_map() {
        let tr = TimedResult::new(42, Duration::from_micros(100));
        let tr2 = tr.map(|x| x.to_string());
        assert_eq!(tr2.result, "42");
        assert_eq!(tr2.latency, Duration::from_micros(100));
    }
}
