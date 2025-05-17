export interface RiskMetric {
  title: string;
  value: number;
  unit: string;
  threshold: number;
  status: 'low' | 'medium' | 'high';
}

export interface ExposureData {
  asset: string;
  amount: number;
  value: number;
  risk: number;
}

export interface RiskEvent {
  id: string;
  type: 'warning' | 'breach' | 'info';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface CircuitBreaker {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  currentValue: number;
  status: 'active' | 'triggered' | 'disabled';
  lastTriggered?: string;
}

export const riskMetrics: RiskMetric[] = [
  {
    title: 'Portfolio Risk',
    value: 35.2,
    unit: '%',
    threshold: 50,
    status: 'low'
  },
  {
    title: 'Exposure Risk',
    value: 42.8,
    unit: '%',
    threshold: 60,
    status: 'medium'
  },
  {
    title: 'Volatility Risk',
    value: 28.4,
    unit: '%',
    threshold: 45,
    status: 'low'
  },
  {
    title: 'Liquidity Risk',
    value: 15.6,
    unit: '%',
    threshold: 40,
    status: 'low'
  }
];

export const exposureData: ExposureData[] = [
  {
    asset: 'ETH',
    amount: 124.5,
    value: 245678,
    risk: 32.4
  },
  {
    asset: 'USDC',
    amount: 156789,
    value: 156789,
    risk: 12.8
  },
  {
    asset: 'WBTC',
    amount: 4.2,
    value: 167890,
    risk: 28.6
  },
  {
    asset: 'ARB',
    amount: 12456,
    value: 45678,
    risk: 42.1
  }
];

export const riskEvents: RiskEvent[] = [
  {
    id: 'event-1',
    type: 'warning',
    message: 'High exposure detected in ETH position',
    timestamp: new Date().toISOString(),
    acknowledged: false
  },
  {
    id: 'event-2',
    type: 'breach',
    message: 'Circuit breaker triggered for Flash Loan strategy',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    acknowledged: true
  },
  {
    id: 'event-3',
    type: 'info',
    message: 'Risk parameters updated for MEV strategy',
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    acknowledged: true
  }
];

export const circuitBreakers: CircuitBreaker[] = [
  {
    id: 'cb-1',
    name: 'Max Daily Loss',
    condition: 'Daily loss exceeds threshold',
    threshold: 5,
    currentValue: 2.4,
    status: 'active'
  },
  {
    id: 'cb-2',
    name: 'Strategy Failure Rate',
    condition: 'Success rate below threshold',
    threshold: 80,
    currentValue: 92.5,
    status: 'active'
  },
  {
    id: 'cb-3',
    name: 'Flash Loan Exposure',
    condition: 'Total borrowed amount exceeds threshold',
    threshold: 1000000,
    currentValue: 1250000,
    status: 'triggered',
    lastTriggered: new Date(Date.now() - 3600000).toISOString()
  }
];