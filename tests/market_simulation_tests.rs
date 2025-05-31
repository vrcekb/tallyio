//! Market Condition Simulation Tests for TallyIO
//!
//! Tests system behavior under various market conditions including:
//! - High volatility periods
//! - Flash crashes
//! - Bull/bear markets
//! - Low liquidity scenarios
//! - MEV competition

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::expect_used)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::default_numeric_fallback)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::suboptimal_flops)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tallyio_core::error::CoreResult;
use tallyio_core::mempool::MempoolAnalyzer;
use tallyio_core::types::{Gas, Price, Transaction};

/// Market condition types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketCondition {
    Bull,           // Rising prices
    Bear,           // Falling prices
    Sideways,       // Stable prices
    HighVolatility, // Rapid price changes
    FlashCrash,     // Sudden price drop
    LowLiquidity,   // Reduced trading volume
}

/// Market simulator for testing different conditions
#[derive(Debug)]
pub struct MarketSimulator {
    base_price: Price,
    current_price: Price,
    volatility: f64,
    condition: MarketCondition,
    tick_count: AtomicU64,
}

impl MarketSimulator {
    pub fn new(base_price: Price, condition: MarketCondition) -> Self {
        let volatility = match condition {
            MarketCondition::Bull => 0.02,           // 2% volatility
            MarketCondition::Bear => 0.03,           // 3% volatility
            MarketCondition::Sideways => 0.005,      // 0.5% volatility
            MarketCondition::HighVolatility => 0.15, // 15% volatility
            MarketCondition::FlashCrash => 0.50,     // 50% volatility
            MarketCondition::LowLiquidity => 0.08,   // 8% volatility
        };

        Self {
            base_price,
            current_price: base_price,
            volatility,
            condition,
            tick_count: AtomicU64::new(0),
        }
    }

    pub fn tick(&mut self) -> Price {
        let tick = self.tick_count.fetch_add(1, Ordering::Relaxed);

        match self.condition {
            MarketCondition::Bull => {
                // Gradual upward trend with volatility
                let trend = 1.0 + (tick as f64 * 0.001); // 0.1% per tick
                let noise = 1.0 + (fastrand::f64() - 0.5) * self.volatility;
                self.current_price =
                    Price::from_gwei((self.base_price.as_gwei() as f64 * trend * noise) as u64);
            }
            MarketCondition::Bear => {
                // Gradual downward trend with volatility
                let trend = 1.0 - (tick as f64 * 0.001); // -0.1% per tick
                let noise = 1.0 + (fastrand::f64() - 0.5) * self.volatility;
                self.current_price =
                    Price::from_gwei((self.base_price.as_gwei() as f64 * trend * noise) as u64);
            }
            MarketCondition::Sideways => {
                // Small random movements around base price
                let noise = 1.0 + (fastrand::f64() - 0.5) * self.volatility;
                self.current_price =
                    Price::from_gwei((self.base_price.as_gwei() as f64 * noise) as u64);
            }
            MarketCondition::HighVolatility => {
                // Large random movements
                let noise = 1.0 + (fastrand::f64() - 0.5) * self.volatility;
                self.current_price =
                    Price::from_gwei((self.current_price.as_gwei() as f64 * noise) as u64);
            }
            MarketCondition::FlashCrash => {
                // Sudden drop followed by recovery
                if tick < 10 {
                    // Crash phase
                    let crash_factor = 1.0 - (tick as f64 * 0.05); // 5% drop per tick
                    self.current_price =
                        Price::from_gwei((self.base_price.as_gwei() as f64 * crash_factor) as u64);
                } else {
                    // Recovery phase
                    let recovery_factor = 0.5 + ((tick - 10) as f64 * 0.02); // 2% recovery per tick
                    self.current_price = Price::from_gwei(
                        (self.base_price.as_gwei() as f64 * recovery_factor) as u64,
                    );
                }
            }
            MarketCondition::LowLiquidity => {
                // Wider spreads and more volatile movements
                let noise = 1.0 + (fastrand::f64() - 0.5) * self.volatility;
                self.current_price =
                    Price::from_gwei((self.current_price.as_gwei() as f64 * noise) as u64);
            }
        }

        self.current_price
    }

    pub fn current_price(&self) -> Price {
        self.current_price
    }

    pub fn condition(&self) -> MarketCondition {
        self.condition
    }
}

/// Market condition simulation tests
#[cfg(test)]
mod market_condition_tests {
    use super::*;

    #[test]
    fn test_bull_market_performance() -> CoreResult<()> {
        let mut simulator = MarketSimulator::new(
            Price::from_gwei(2000_000_000_000), // 2000 ETH base price
            MarketCondition::Bull,
        );

        let analyzer = MempoolAnalyzer::new();
        let mut profitable_opportunities = 0;
        let mut total_transactions = 0;
        let start_price = simulator.current_price();
        let mut price_samples = Vec::new();

        for i in 0..100 {
            let current_price = simulator.tick();
            price_samples.push(current_price.as_gwei());

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                current_price,
                Price::from_gwei(50 + i as u64), // Increasing gas prices
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            if let Ok(analysis) = analyzer.analyze_transaction(&tx) {
                total_transactions += 1;
                if analysis.has_mev_opportunity {
                    profitable_opportunities += 1;
                }
            }
        }

        let end_price = simulator.current_price();

        // Calculate trend using linear regression or simple average comparison
        let mid_point = price_samples.len() / 2;
        let first_half_avg: u64 = price_samples[..mid_point].iter().sum::<u64>() / mid_point as u64;
        let second_half_avg: u64 = price_samples[mid_point..].iter().sum::<u64>()
            / (price_samples.len() - mid_point) as u64;

        // In bull market, second half should be at least as high as first half (allowing for volatility)
        let trend_positive = second_half_avg >= first_half_avg.saturating_sub(first_half_avg / 10); // Allow 10% tolerance

        assert!(
            trend_positive,
            "Price trend should be positive in bull market"
        );
        assert!(
            total_transactions > 80,
            "Should process most transactions in bull market"
        );

        println!("Bull market test results:");
        println!("  Start price: {} gwei", start_price.as_gwei());
        println!("  End price: {} gwei", end_price.as_gwei());
        println!("  Total transactions: {total_transactions}");
        println!("  Profitable opportunities: {profitable_opportunities}");

        Ok(())
    }

    #[test]
    fn test_flash_crash_resilience() -> CoreResult<()> {
        let mut simulator = MarketSimulator::new(
            Price::from_gwei(2000_000_000_000), // 2000 ETH base price
            MarketCondition::FlashCrash,
        );

        let analyzer = MempoolAnalyzer::new();
        let mut crash_phase_transactions = 0;
        let mut recovery_phase_transactions = 0;
        let mut prices = Vec::new();

        for i in 0..50 {
            let current_price = simulator.tick();
            prices.push(current_price.as_gwei());

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                current_price,
                Price::from_gwei(100), // High gas price for urgency
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            if analyzer.analyze_transaction(&tx).is_ok() {
                if i < 10 {
                    crash_phase_transactions += 1;
                } else {
                    recovery_phase_transactions += 1;
                }
            }
        }

        // System should handle both crash and recovery phases
        assert!(
            crash_phase_transactions > 5,
            "Should handle some transactions during crash"
        );
        assert!(
            recovery_phase_transactions > 20,
            "Should handle more transactions during recovery"
        );

        // Price should show crash pattern (drop then recovery)
        let min_price = prices
            .iter()
            .min()
            .ok_or_else(|| tallyio_core::error::CoreError::mempool("No prices recorded"))?;
        let max_price = prices
            .iter()
            .max()
            .ok_or_else(|| tallyio_core::error::CoreError::mempool("No prices recorded"))?;
        let price_volatility = (*max_price as f64 - *min_price as f64) / *max_price as f64;

        assert!(
            price_volatility > 0.3,
            "Flash crash should show high volatility"
        );

        println!("Flash crash test results:");
        println!("  Crash phase transactions: {crash_phase_transactions}");
        println!("  Recovery phase transactions: {recovery_phase_transactions}");
        println!("  Price volatility: {:.2}%", price_volatility * 100.0);
        println!("  Min price: {min_price} gwei");
        println!("  Max price: {max_price} gwei");

        Ok(())
    }

    #[test]
    fn test_high_volatility_adaptation() -> CoreResult<()> {
        let mut simulator = MarketSimulator::new(
            Price::from_gwei(2000_000_000_000),
            MarketCondition::HighVolatility,
        );

        let analyzer = MempoolAnalyzer::new();
        let mut successful_adaptations = 0;
        let mut price_changes = Vec::new();
        let mut previous_price = simulator.current_price();

        for i in 0..100 {
            let current_price = simulator.tick();
            let price_change = if previous_price.as_gwei() > 0 {
                (current_price.as_gwei() as f64 - previous_price.as_gwei() as f64)
                    / previous_price.as_gwei() as f64
            } else {
                0.0
            };
            price_changes.push(price_change.abs());

            // Adjust gas price based on volatility
            let adaptive_gas_price = if price_change.abs() > 0.05 {
                Price::from_gwei(100) // High gas for volatile periods
            } else {
                Price::from_gwei(50) // Normal gas for stable periods
            };

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                current_price,
                adaptive_gas_price,
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            if analyzer.analyze_transaction(&tx).is_ok() {
                successful_adaptations += 1;
            }

            previous_price = current_price;
        }

        let avg_volatility = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
        let adaptation_rate = successful_adaptations as f64 / 100.0;

        // System should adapt well to high volatility
        assert!(
            adaptation_rate > 0.7,
            "Adaptation rate too low: {:.2}%",
            adaptation_rate * 100.0
        );
        assert!(
            avg_volatility > 0.01,
            "Volatility should be measurable in this test: {:.4}",
            avg_volatility
        );

        println!("High volatility adaptation test results:");
        println!("  Successful adaptations: {}", successful_adaptations);
        println!("  Adaptation rate: {:.2}%", adaptation_rate * 100.0);
        println!("  Average volatility: {:.4}", avg_volatility);

        Ok(())
    }

    #[test]
    fn test_low_liquidity_handling() -> CoreResult<()> {
        let mut simulator = MarketSimulator::new(
            Price::from_gwei(2000_000_000_000),
            MarketCondition::LowLiquidity,
        );

        let analyzer = MempoolAnalyzer::new();
        let mut large_transactions = 0;
        let mut small_transactions = 0;
        let mut slippage_events = 0;

        for i in 0..100 {
            let current_price = simulator.tick();

            // Simulate different transaction sizes
            let is_large_tx = i % 10 == 0;
            let tx_value = if is_large_tx {
                Price::from_gwei(10_000_000_000) // 10 ETH - large transaction
            } else {
                Price::from_gwei(1_000_000_000) // 1 ETH - small transaction
            };

            let tx = Transaction::new(
                [i as u8; 20],
                Some([(i + 1) as u8; 20]),
                tx_value,
                Price::from_gwei(75), // Higher gas for low liquidity
                Gas::new(150_000),
                i as u64,
                vec![0xa9, 0x05, 0x9c, 0xbb],
            );

            if let Ok(analysis) = analyzer.analyze_transaction(&tx) {
                if is_large_tx {
                    large_transactions += 1;
                    // Large transactions in low liquidity might cause slippage
                    if analysis.analysis_time_ns > 500_000 {
                        // > 0.5ms indicates complexity
                        slippage_events += 1;
                    }
                } else {
                    small_transactions += 1;
                }
            }
        }

        // Small transactions should be processed more easily than large ones
        assert!(
            small_transactions > large_transactions,
            "Small transactions should be more successful"
        );
        assert!(
            large_transactions > 5,
            "Some large transactions should still succeed"
        );

        println!("Low liquidity handling test results:");
        println!("  Large transactions processed: {}", large_transactions);
        println!("  Small transactions processed: {}", small_transactions);
        println!("  Slippage events detected: {}", slippage_events);

        Ok(())
    }

    #[test]
    fn test_mev_competition_simulation() -> CoreResult<()> {
        let mut simulator = MarketSimulator::new(
            Price::from_gwei(2000_000_000_000),
            MarketCondition::HighVolatility,
        );

        let analyzer = Arc::new(MempoolAnalyzer::new());
        let mev_opportunities = Arc::new(AtomicU64::new(0));
        let successful_extractions = Arc::new(AtomicU64::new(0));

        let mut handles = vec![];
        let num_competitors = 3;
        let transactions_per_competitor = 20;

        // Simulate multiple MEV bots competing
        for competitor_id in 0..num_competitors {
            let analyzer = Arc::clone(&analyzer);
            let mev_opportunities = Arc::clone(&mev_opportunities);
            let successful_extractions = Arc::clone(&successful_extractions);

            let handle = thread::spawn(move || {
                for i in 0..transactions_per_competitor {
                    // Each competitor uses different gas prices (competition)
                    let competitive_gas_price = Price::from_gwei(50 + (competitor_id * 20) as u64);

                    let tx = Transaction::new(
                        [(competitor_id * 100 + i) as u8; 20],
                        Some([((competitor_id * 100 + i) + 1) as u8; 20]),
                        Price::from_gwei(5_000_000_000), // 5 ETH - profitable transaction
                        competitive_gas_price,
                        Gas::new(150_000),
                        (competitor_id * 100 + i) as u64,
                        vec![0xa9, 0x05, 0x9c, 0xbb],
                    );

                    if let Ok(analysis) = analyzer.analyze_transaction(&tx) {
                        if analysis.has_mev_opportunity {
                            mev_opportunities.fetch_add(1, Ordering::Relaxed);

                            // Simulate successful extraction based on gas price
                            if competitive_gas_price.as_gwei() > 70 {
                                successful_extractions.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }

                    // Small delay to simulate real competition timing
                    thread::sleep(Duration::from_millis(1));
                }
            });

            handles.push(handle);
        }

        // Wait for all competitors
        for handle in handles {
            handle.join().expect("Competitor thread panicked");
        }

        let total_opportunities = mev_opportunities.load(Ordering::Relaxed);
        let total_extractions = successful_extractions.load(Ordering::Relaxed);
        let extraction_rate = if total_opportunities > 0 {
            total_extractions as f64 / total_opportunities as f64
        } else {
            0.0
        };

        // MEV competition should create opportunities and successful extractions
        assert!(
            total_opportunities > 10,
            "Should find MEV opportunities in competition"
        );
        assert!(
            extraction_rate > 0.3,
            "Extraction rate too low: {:.2}%",
            extraction_rate * 100.0
        );

        println!("MEV competition simulation results:");
        println!("  Total MEV opportunities: {}", total_opportunities);
        println!("  Successful extractions: {}", total_extractions);
        println!("  Extraction rate: {:.2}%", extraction_rate * 100.0);
        println!("  Competitors: {}", num_competitors);

        Ok(())
    }

    #[test]
    fn test_market_regime_transitions() -> CoreResult<()> {
        let analyzer = MempoolAnalyzer::new();
        let mut regime_performance = HashMap::new();

        // Test different market regimes
        let regimes = vec![
            MarketCondition::Bull,
            MarketCondition::Bear,
            MarketCondition::Sideways,
            MarketCondition::HighVolatility,
        ];

        for regime in regimes {
            let mut simulator = MarketSimulator::new(Price::from_gwei(2000_000_000_000), regime);

            let mut successful_transactions = 0;
            let start_time = Instant::now();

            for i in 0..50 {
                let current_price = simulator.tick();

                let tx = Transaction::new(
                    [i as u8; 20],
                    Some([(i + 1) as u8; 20]),
                    current_price,
                    Price::from_gwei(60),
                    Gas::new(150_000),
                    i as u64,
                    vec![0xa9, 0x05, 0x9c, 0xbb],
                );

                if analyzer.analyze_transaction(&tx).is_ok() {
                    successful_transactions += 1;
                }
            }

            let duration = start_time.elapsed();
            regime_performance.insert(regime, (successful_transactions, duration));
        }

        // System should perform reasonably well across all market regimes
        for (regime, (success_count, duration)) in &regime_performance {
            assert!(
                *success_count > 30,
                "Poor performance in {:?} regime: {}",
                regime,
                success_count
            );
            assert!(
                *duration < Duration::from_secs(1),
                "Too slow in {:?} regime",
                regime
            );
        }

        println!("Market regime transition test results:");
        for (regime, (success_count, duration)) in regime_performance {
            println!(
                "  {:?}: {} successful, {:?} duration",
                regime, success_count, duration
            );
        }

        Ok(())
    }
}
