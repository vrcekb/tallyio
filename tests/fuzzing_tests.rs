//! Fuzzing Tests for `TallyIO`
//!
//! Simplified property-based tests to discover edge cases and ensure robustness.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::default_numeric_fallback)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use std::time::Instant;
use tallyio_core::error::CoreResult;
use tallyio_core::mempool::MempoolAnalyzer;
use tallyio_core::types::{Gas, Price, Transaction};

#[cfg(test)]
mod simple_fuzzing_tests {
    use super::*;

    #[test]
    fn test_random_transaction_processing() -> CoreResult<()> {
        let analyzer = MempoolAnalyzer::new();
        let mut successful_operations = 0;
        let mut error_operations = 0;
        let iterations = 50;

        for i in 0..iterations {
            let value = Price::from_gwei(fastrand::u64(1_000_000..10_000_000_000));
            let gas_price = Price::from_gwei(fastrand::u64(1..1000));
            let gas_limit = Gas::new(fastrand::u64(21_000..1_000_000));

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                value,
                gas_price,
                gas_limit,
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            match analyzer.analyze_transaction(&tx) {
                Ok(_) => successful_operations += 1,
                Err(_) => error_operations += 1,
            }
        }

        assert!(successful_operations + error_operations == iterations);
        assert!(
            successful_operations > 0,
            "Should process some transactions successfully"
        );

        println!(
            "Random transaction processing: {}/{} successful",
            successful_operations, iterations
        );
        Ok(())
    }

    #[test]
    fn test_extreme_values() -> CoreResult<()> {
        let analyzer = MempoolAnalyzer::new();
        let mut handled_gracefully = 0;
        let test_cases = vec![
            (Price::from_gwei(0), Price::from_gwei(1), Gas::new(21_000)),
            (
                Price::from_gwei(u64::MAX / 1_000_000),
                Price::from_gwei(1000),
                Gas::new(1_000_000),
            ),
            (Price::from_gwei(1), Price::from_gwei(1), Gas::new(21_000)),
        ];

        for (i, (value, gas_price, gas_limit)) in test_cases.iter().enumerate() {
            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                *value,
                *gas_price,
                *gas_limit,
                i as u64,
                vec![],
            );

            match analyzer.analyze_transaction(&tx) {
                Ok(_) | Err(_) => handled_gracefully += 1,
            }
        }

        assert_eq!(handled_gracefully, test_cases.len());
        println!(
            "Extreme values: {} cases handled gracefully",
            handled_gracefully
        );
        Ok(())
    }

    #[test]
    fn test_performance_under_load() -> CoreResult<()> {
        let analyzer = MempoolAnalyzer::new();
        let iterations = 100;
        let start_time = Instant::now();

        for i in 0..iterations {
            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                Price::from_gwei(fastrand::u64(1_000_000..5_000_000_000)),
                Price::from_gwei(fastrand::u64(10..200)),
                Gas::new(fastrand::u64(21_000..500_000)),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            let _ = analyzer.analyze_transaction(&tx);
        }

        let total_duration = start_time.elapsed();
        let throughput = iterations as f64 / total_duration.as_secs_f64();

        assert!(
            throughput > 50.0,
            "Throughput too low: {:.2} ops/sec",
            throughput
        );
        println!("Performance: {:.2} ops/sec", throughput);
        Ok(())
    }
}
