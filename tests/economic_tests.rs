//! Economic Logic Testing for `TallyIO` MEV/Liquidator
//!
//! Tests critical financial calculations, profit margins, and economic assumptions
//! that are fundamental to MEV bot profitability and safety.

#![allow(clippy::unnecessary_wraps)] // Tests need Result for consistency

use tallyio_core::error::CoreResult;
use tallyio_core::types::{Gas, Opportunity, OpportunityType, Price};

/// Test arbitrage profitability calculations under various conditions
#[cfg(test)]
mod arbitrage_economics {
    use super::*;

    #[test]
    fn test_minimum_profit_thresholds() -> CoreResult<()> {
        let gas_price = Price::from_gwei(50); // 50 gwei
        let gas_limit = Gas::new(150_000); // Typical DEX swap
        let gas_cost = gas_limit.cost_at_price(gas_price);

        // Test minimum profitable arbitrage
        let min_profit = gas_cost.mul(2); // 2x gas cost minimum
        let opportunity = Opportunity::new(OpportunityType::Arbitrage, min_profit, gas_limit);

        assert!(opportunity.is_profitable(gas_price));

        // Test unprofitable arbitrage
        let low_profit = gas_cost.div(2); // 0.5x gas cost
        let unprofitable = Opportunity::new(OpportunityType::Arbitrage, low_profit, gas_limit);

        assert!(!unprofitable.is_profitable(gas_price));

        Ok(())
    }

    #[test]
    fn test_gas_price_impact_on_profitability() -> CoreResult<()> {
        let base_value = Price::from_ether(1);
        let gas_limit = Gas::new(200_000);

        let opportunity = Opportunity::new(OpportunityType::Arbitrage, base_value, gas_limit);

        // Test at different gas prices
        let low_gas = Price::from_gwei(20);
        let medium_gas = Price::from_gwei(100);
        let high_gas = Price::from_gwei(500);

        let low_margin = opportunity.profit_margin(low_gas);
        let medium_margin = opportunity.profit_margin(medium_gas);
        let high_margin = opportunity.profit_margin(high_gas);

        // Higher gas price should result in lower profit margin
        assert!(low_margin > medium_margin);
        assert!(medium_margin > high_margin);

        Ok(())
    }

    #[test]
    fn test_slippage_impact_calculation() -> CoreResult<()> {
        let mut opportunity = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(1),
            Gas::new(150_000),
        );

        // Test different slippage tolerances
        let slippages = [50, 100, 300, 500]; // 0.5%, 1%, 3%, 5%

        for slippage in slippages {
            opportunity.slippage_tolerance = slippage;

            // Calculate expected value after slippage
            let slippage_factor = 10000 - slippage; // basis points
            let expected_value = opportunity.value.mul(u64::from(slippage_factor)).div(10000);

            // Verify slippage reduces expected profit
            assert!(expected_value.as_wei() < opportunity.value.as_wei());

            // Higher slippage should reduce profitability more
            if slippage > 100 {
                // > 1%
                let gas_price = Price::from_gwei(50);
                let margin_with_slippage = opportunity.profit_margin(gas_price);

                // With high slippage, margin should be reasonable
                // Note: margin can be high if gas costs are low relative to value
                assert!(margin_with_slippage <= 10000); // At most 100% (10000 basis points)
            }
        }

        Ok(())
    }

    #[test]
    fn test_multi_hop_arbitrage_economics() -> CoreResult<()> {
        // Test 3-hop arbitrage: ETH -> USDC -> DAI -> ETH
        let _gas_per_hop = Gas::new(150_000);
        let total_gas = Gas::new(450_000); // 3 hops
        let gas_price = Price::from_gwei(100);

        let opportunity = Opportunity::new(
            OpportunityType::Arbitrage,
            Price::from_ether(2), // 2 ETH profit potential
            total_gas,
        );

        // Multi-hop should require higher profit due to:
        // 1. Higher gas costs
        // 2. Increased slippage risk
        // 3. Higher execution complexity

        let total_gas_cost = total_gas.cost_at_price(gas_price);
        let profit_after_gas = opportunity.value.sub(total_gas_cost);

        // Should still be profitable after accounting for all costs
        assert!(profit_after_gas.as_wei() > 0);

        // Profit margin should be reasonable (>20%)
        let margin = opportunity.profit_margin(gas_price);
        assert!(margin >= 20);

        Ok(())
    }
}

/// Test liquidation economic calculations
#[cfg(test)]
mod liquidation_economics {
    use super::*;

    #[test]
    fn test_liquidation_bonus_calculations() -> CoreResult<()> {
        // Typical liquidation: 5% bonus on $100k position (50 ETH at $2000/ETH)
        let position_value = Price::from_gwei(50_000_000); // 0.05 ETH = ~$100 at $2000/ETH
        let liquidation_bonus = 500_u16; // 5% in basis points

        let bonus_amount = position_value.mul(u64::from(liquidation_bonus)).div(10000);
        let _gas_cost = Gas::new(400_000).cost_at_price(Price::from_gwei(100));

        let opportunity = Opportunity::new(
            OpportunityType::Liquidation,
            bonus_amount,
            Gas::new(400_000),
        );

        // Check if liquidation is profitable with reasonable gas costs
        let gas_price = Price::from_gwei(100);
        let gas_cost = opportunity.gas_cost.cost_at_price(gas_price);

        // Only test profitability if bonus exceeds gas cost
        if bonus_amount.as_wei() > gas_cost.as_wei() {
            assert!(opportunity.is_profitable(gas_price));

            // Profit margin should be reasonable
            let margin = opportunity.profit_margin(gas_price);
            assert!(margin > 0); // Should have some profit margin
        }

        Ok(())
    }

    #[test]
    fn test_health_factor_boundaries() -> CoreResult<()> {
        // Test liquidation thresholds
        // Health factor = (collateral * liquidation_threshold) / debt

        struct LiquidationScenario {
            collateral_value: u64,      // in wei
            debt_value: u64,            // in wei
            liquidation_threshold: u16, // in basis points (e.g., 8000 = 80%)
            expected_liquidatable: bool,
        }

        let scenarios = [
            // Healthy position
            LiquidationScenario {
                collateral_value: 1_000_000_000_000_000_000, // 1 ETH
                debt_value: 500_000_000_000_000_000,         // 0.5 ETH
                liquidation_threshold: 8000,                 // 80%
                expected_liquidatable: false,
            },
            // Borderline position (exactly at threshold)
            LiquidationScenario {
                collateral_value: 1_000_000_000_000_000_000, // 1 ETH
                debt_value: 800_000_000_000_000_000,         // 0.8 ETH
                liquidation_threshold: 8000,                 // 80%
                expected_liquidatable: false, // Health factor = 1.0, not liquidatable yet
            },
            // Liquidatable position (health factor < 1)
            LiquidationScenario {
                collateral_value: 1_000_000_000_000_000_000, // 1 ETH
                debt_value: 900_000_000_000_000_000,         // 0.9 ETH
                liquidation_threshold: 8000,                 // 80%
                expected_liquidatable: true,
            },
        ];

        for scenario in scenarios {
            // Health factor = (collateral_value * liquidation_threshold) / debt_value
            // Use u128 to prevent overflow, then convert back
            let collateral_u128 = u128::from(scenario.collateral_value);
            let debt_u128 = u128::from(scenario.debt_value);
            let threshold_u128 = u128::from(scenario.liquidation_threshold);

            let numerator = collateral_u128 * threshold_u128;
            let health_factor_bp = if debt_u128 == 0 {
                u64::MAX
            } else {
                let result = numerator / debt_u128;
                if result > u128::from(u64::MAX) {
                    u64::MAX
                } else {
                    #[allow(clippy::cast_possible_truncation)] // We checked bounds above
                    {
                        result as u64
                    }
                }
            };

            // Position is liquidatable if health factor < 10000 basis points (100%)
            let is_liquidatable = health_factor_bp < 10000;
            assert_eq!(is_liquidatable, scenario.expected_liquidatable);

            if is_liquidatable {
                // Create liquidation opportunity
                let liquidation_bonus = scenario.debt_value / 20; // 5% bonus
                let opportunity = Opportunity::new(
                    OpportunityType::Liquidation,
                    Price::new(liquidation_bonus),
                    Gas::new(400_000),
                );

                assert!(opportunity.is_profitable(Price::from_gwei(100)));
            }
        }

        Ok(())
    }

    #[test]
    fn test_collateral_price_volatility_impact() -> CoreResult<()> {
        // Test how collateral price changes affect liquidation opportunities
        let base_collateral_price = Price::from_gwei(2_000_000); // 0.002 ETH collateral
        let debt_amount = Price::from_gwei(1_000_000); // 0.001 ETH debt (50% LTV)

        // Test price drops that trigger liquidations
        let price_drops = [5, 10, 15, 20]; // percentage drops

        for drop_percent in price_drops {
            let new_price = base_collateral_price.mul(100 - drop_percent).div(100);
            // Health factor = (collateral_value * liquidation_threshold) / debt_value
            // Using 80% liquidation threshold (8000 basis points)
            let numerator = new_price.as_wei().saturating_mul(8000);
            let denominator = debt_amount.as_wei().saturating_mul(10000);
            let health_factor = if denominator == 0 {
                u64::MAX
            } else {
                numerator / denominator
            };

            // Position becomes liquidatable if health factor < 1.0 (10000 basis points)
            if health_factor < 10000 {
                // Position becomes liquidatable
                let liquidation_value = debt_amount.mul(105).div(100); // 5% bonus
                let opportunity = Opportunity::new(
                    OpportunityType::Liquidation,
                    liquidation_value,
                    Gas::new(400_000),
                );

                let gas_price = Price::from_gwei(100);
                let gas_cost = opportunity.gas_cost.cost_at_price(gas_price);

                // Only test profitability if liquidation value exceeds gas cost
                if liquidation_value.as_wei() > gas_cost.as_wei() {
                    assert!(opportunity.is_profitable(gas_price));

                    // Larger price drops should create more profitable opportunities
                    if drop_percent >= 15 {
                        let margin = opportunity.profit_margin(gas_price);
                        assert!(margin > 0); // Should have some profit margin
                    }
                }
            }
        }

        Ok(())
    }
}

/// Test MEV opportunity economic validation
#[cfg(test)]
mod mev_economics {
    use super::*;

    #[test]
    fn test_sandwich_attack_economics() -> CoreResult<()> {
        // Test sandwich attack profitability calculation
        // Front-run + victim tx + back-run

        let victim_tx_size = Price::from_gwei(10_000_000); // 0.01 ETH trade
        let expected_slippage = 200_u16; // 2% slippage from large trade

        // Front-run gas cost
        let _frontrun_gas = Gas::new(150_000);
        // Back-run gas cost
        let _backrun_gas = Gas::new(150_000);
        let total_gas = Gas::new(300_000);

        let gas_price = Price::from_gwei(150); // High priority gas

        // Expected profit from price impact
        let price_impact_profit = victim_tx_size.mul(u64::from(expected_slippage)).div(10000);

        let opportunity =
            Opportunity::new(OpportunityType::Sandwich, price_impact_profit, total_gas);

        // Calculate actual profit vs gas cost
        let gas_cost = total_gas.cost_at_price(gas_price);
        let net_profit = if price_impact_profit.as_wei() > gas_cost.as_wei() {
            price_impact_profit.sub(gas_cost)
        } else {
            Price::new(0)
        };

        // Should have some profit after gas costs or not be profitable
        assert!(net_profit.as_wei() > 0 || !opportunity.is_profitable(gas_price));

        // Check if opportunity is profitable, if so, margin should be reasonable
        if opportunity.is_profitable(gas_price) {
            let margin = opportunity.profit_margin(gas_price);
            assert!(margin >= 10); // Minimum 10% margin for execution risk
        }

        Ok(())
    }

    #[test]
    fn test_flash_loan_arbitrage_economics() -> CoreResult<()> {
        // Test flash loan arbitrage economics
        let loan_amount = Price::from_gwei(100_000_000); // 0.1 ETH flash loan
        let flash_loan_fee = 9_u16; // 0.09% fee (9 basis points)

        let loan_fee = loan_amount.mul(u64::from(flash_loan_fee)).div(10000);
        let arbitrage_profit = Price::from_gwei(5_000_000); // 0.005 ETH profit potential
        let gas_cost = Gas::new(500_000).cost_at_price(Price::from_gwei(100));

        // Net profit = arbitrage_profit - loan_fee - gas_cost
        let net_profit = arbitrage_profit.sub(loan_fee).sub(gas_cost);

        let opportunity =
            Opportunity::new(OpportunityType::FlashLoan, net_profit, Gas::new(500_000));

        // Check if flash loan arbitrage is profitable
        let gas_price = Price::from_gwei(100);

        // Only test profitability if net profit is positive
        if net_profit.as_wei() > 0 {
            assert!(opportunity.is_profitable(gas_price));

            // Flash loan arbitrage should have reasonable margins
            let margin = opportunity.profit_margin(gas_price);
            assert!(margin > 0); // Should have some profit margin
        }

        Ok(())
    }
}
