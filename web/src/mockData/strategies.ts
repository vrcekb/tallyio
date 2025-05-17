export interface Strategy {
  id: string;
  name: string;
  type: 'arbitrage' | 'liquidation' | 'mev' | 'flash';
  status: 'active' | 'paused' | 'stopped';
  profit24h: number;
  profitTotal: number;
  successRate: number;
  executionCount: number;
  avgExecutionTime: number;
  lastExecution: string;
}

export interface StrategyMetrics {
  title: string;
  value: number;
  unit?: string;
  change: number;
}

export interface StrategyPerformance {
  timestamp: string;
  profit: number;
  executions: number;
  gasUsed: number;
}

export const strategies: Strategy[] = [
  {
    id: 'strat-1',
    name: 'Cross-Exchange Arbitrage',
    type: 'arbitrage',
    status: 'active',
    profit24h: 2.45,
    profitTotal: 158.67,
    successRate: 92.5,
    executionCount: 1245,
    avgExecutionTime: 125,
    lastExecution: new Date().toISOString()
  },
  {
    id: 'strat-2',
    name: 'MEV Sandwich',
    type: 'mev',
    status: 'active',
    profit24h: 3.12,
    profitTotal: 245.89,
    successRate: 88.2,
    executionCount: 2456,
    avgExecutionTime: 85,
    lastExecution: new Date().toISOString()
  },
  {
    id: 'strat-3',
    name: 'Liquidation Hunter',
    type: 'liquidation',
    status: 'active',
    profit24h: 1.85,
    profitTotal: 125.45,
    successRate: 95.6,
    executionCount: 856,
    avgExecutionTime: 156,
    lastExecution: new Date().toISOString()
  },
  {
    id: 'strat-4',
    name: 'Flash Loan Arbitrage',
    type: 'flash',
    status: 'paused',
    profit24h: 0,
    profitTotal: 89.34,
    successRate: 82.4,
    executionCount: 1567,
    avgExecutionTime: 245,
    lastExecution: new Date(Date.now() - 86400000).toISOString()
  }
];

export const strategyMetrics: StrategyMetrics[] = [
  {
    title: 'Total Active Strategies',
    value: 3,
    change: 0
  },
  {
    title: '24h Profit',
    value: 7.42,
    unit: 'ETH',
    change: 12.5
  },
  {
    title: 'Success Rate',
    value: 92.1,
    unit: '%',
    change: 1.2
  },
  {
    title: 'Avg Execution Time',
    value: 152,
    unit: 'ms',
    change: -8.5
  }
];

export const generateStrategyPerformance = (hours: number): StrategyPerformance[] => {
  const data: StrategyPerformance[] = [];
  const now = new Date();

  for (let i = hours; i >= 0; i--) {
    const timestamp = new Date(now);
    timestamp.setHours(timestamp.getHours() - i);

    data.push({
      timestamp: timestamp.toISOString(),
      profit: Math.random() * 0.5 + 0.1,
      executions: Math.floor(Math.random() * 50 + 20),
      gasUsed: Math.floor(Math.random() * 1000000 + 500000)
    });
  }

  return data;
};