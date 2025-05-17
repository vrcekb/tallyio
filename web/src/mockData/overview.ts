import { format } from 'date-fns';
import {
  ActivityData,
  Alert,
  ChainStatus,
  KeyMetric,
  RpcLocation,
  Strategy,
  SystemHealthStatus,
  TimeRange,
  Transaction
} from '../types';

// Generate timestamps for the last 24 hours
const generateTimestamps = (count: number, range: TimeRange): string[] => {
  const now = new Date();
  const timestamps: string[] = [];

  let hoursToSubtract = 1;
  switch(range) {
    case '1h':
      hoursToSubtract = 1;
      break;
    case '24h':
      hoursToSubtract = 24;
      break;
    case '7d':
      hoursToSubtract = 24 * 7;
      break;
    case '30d':
      hoursToSubtract = 24 * 30;
      break;
  }

  const intervalMs = (hoursToSubtract * 60 * 60 * 1000) / count;

  for (let i = 0; i < count; i++) {
    const time = new Date(now.getTime() - (i * intervalMs));
    timestamps.unshift(format(time, 'yyyy-MM-dd HH:mm:ss'));
  }

  return timestamps;
};

// Generate activity data
export const generateActivityData = (count: number, range: TimeRange): ActivityData[] => {
  const timestamps = generateTimestamps(count, range);
  return timestamps.map((timestamp) => ({
    timestamp,
    transactions: Math.floor(Math.random() * 50) + 10,
    profit: parseFloat((Math.random() * 0.5 + 0.1).toFixed(4)),
  }));
};

// Generate system health status
export const systemHealthStatus: SystemHealthStatus[] = [
  {
    component: 'Core Engine',
    status: 'healthy',
    message: 'All systems operational',
  },
  {
    component: 'Transaction Executor',
    status: 'healthy',
    message: 'Processing transactions normally',
  },
  {
    component: 'RPC Connections',
    status: 'warning',
    message: 'Latency issues with 2 providers',
  },
  {
    component: 'Strategy Evaluator',
    status: 'healthy',
    message: 'All strategies running optimally',
  },
  {
    component: 'Risk Management',
    status: 'healthy',
    message: 'All parameters within limits',
  },
];

// Generate key metrics
export const keyMetrics: KeyMetric[] = [
  {
    title: 'Total Profit',
    value: '14.38',
    change: 12.5,
    unit: 'ETH',
    icon: 'trending-up',
  },
  {
    title: 'Active Strategies',
    value: 8,
    change: 0,
    icon: 'zap',
  },
  {
    title: 'Success Rate',
    value: 98.2,
    change: -0.5,
    unit: '%',
    icon: 'check-circle',
  },
  {
    title: 'Avg. Latency',
    value: 0.42,
    change: -15.3,
    unit: 'ms',
    icon: 'clock',
  },
];

// Generate strategies
export const strategies: Strategy[] = [
  {
    id: '1',
    name: 'MEV Sandwich',
    profit: 5.23,
    change: 12.8,
    enabled: true,
  },
  {
    id: '2',
    name: 'Liquidation Hunter',
    profit: 3.85,
    change: 8.3,
    enabled: true,
  },
  {
    id: '3',
    name: 'Arbitrage Detector',
    profit: 2.91,
    change: -4.2,
    enabled: true,
  },
  {
    id: '4',
    name: 'JIT Liquidity',
    profit: 1.73,
    change: 21.5,
    enabled: true,
  },
  {
    id: '5',
    name: 'Flash Loan Strategy',
    profit: 0.66,
    change: -2.1,
    enabled: false,
  },
];

// Generate transactions - optimizirana verzija
export const generateRecentTransactions = (count: number): Transaction[] => {
  const status = ['success', 'pending', 'failed'];
  const networks = ['Ethereum', 'Arbitrum', 'Optimism', 'Base', 'Polygon'];
  const strategyNames = strategies.map(s => s.name);

  // Uporabimo en timestamp za vse transakcije, da zmanjšamo število klicev Date.now()
  const baseTimestamp = Date.now();
  const result: Transaction[] = [];

  for (let i = 0; i < count; i++) {
    // Predračunamo naključne indekse
    const strategyIndex = Math.floor(Math.random() * strategyNames.length);
    const networkIndex = Math.floor(Math.random() * networks.length);
    const statusIndex = Math.floor(Math.random() * (status.length - 0.7));

    // Predračunamo profit/loss
    const profitLoss = parseFloat((Math.random() * 0.5 - Math.random() * 0.1).toFixed(4));

    // Ustvarimo transakcijo
    result.push({
      id: `tx-${baseTimestamp}-${i}`,
      timestamp: format(new Date(baseTimestamp - Math.floor(Math.random() * 3600000)), 'yyyy-MM-dd HH:mm:ss'),
      strategy: strategyNames[strategyIndex],
      network: networks[networkIndex],
      status: status[statusIndex] as 'success' | 'pending' | 'failed',
      profitLoss: profitLoss,
    });
  }

  return result;
};

// Generate chain status
export const chainStatus: ChainStatus[] = [
  {
    id: 'ethereum',
    name: 'Ethereum',
    status: 'online',
    gasPrice: 32,
    gasTrend: 'up',
    connections: 5,
    mempoolSize: 1248,
  },
  {
    id: 'arbitrum',
    name: 'Arbitrum',
    status: 'online',
    gasPrice: 0.25,
    gasTrend: 'stable',
    connections: 3,
    mempoolSize: 458,
  },
  {
    id: 'optimism',
    name: 'Optimism',
    status: 'degraded',
    gasPrice: 0.18,
    gasTrend: 'down',
    connections: 3,
    mempoolSize: 312,
  },
  {
    id: 'base',
    name: 'Base',
    status: 'online',
    gasPrice: 0.12,
    gasTrend: 'stable',
    connections: 2,
    mempoolSize: 206,
  },
  {
    id: 'polygon',
    name: 'Polygon',
    status: 'online',
    gasPrice: 105,
    gasTrend: 'down',
    connections: 4,
    mempoolSize: 875,
  },
];

// Generate alerts
export const alerts: Alert[] = [
  {
    id: 'alert-1',
    severity: 'warning',
    message: 'RPC provider latency exceeding threshold',
    timestamp: format(new Date(Date.now() - 1200000), 'yyyy-MM-dd HH:mm:ss'),
    acknowledged: false,
  },
  {
    id: 'alert-2',
    severity: 'info',
    message: 'New strategy version available',
    timestamp: format(new Date(Date.now() - 3600000), 'yyyy-MM-dd HH:mm:ss'),
    acknowledged: true,
  },
  {
    id: 'alert-3',
    severity: 'critical',
    message: 'Circuit breaker activated for Flash Loan Strategy',
    timestamp: format(new Date(Date.now() - 7200000), 'yyyy-MM-dd HH:mm:ss'),
    acknowledged: true,
  },
];

// Generate RPC locations
export const rpcLocations: RpcLocation[] = [
  { name: 'New York', coordinates: [-74.006, 40.7128], latency: 12, status: 'online' },
  { name: 'San Francisco', coordinates: [-122.4194, 37.7749], latency: 18, status: 'online' },
  { name: 'London', coordinates: [-0.1278, 51.5074], latency: 42, status: 'online' },
  { name: 'Tokyo', coordinates: [139.6503, 35.6762], latency: 78, status: 'online' },
  { name: 'Singapore', coordinates: [103.8198, 1.3521], latency: 65, status: 'online' },
  { name: 'Frankfurt', coordinates: [8.6821, 50.1109], latency: 38, status: 'degraded' },
  { name: 'Sydney', coordinates: [151.2093, -33.8688], latency: 95, status: 'online' },
  { name: 'São Paulo', coordinates: [-46.6333, -23.5505], latency: 110, status: 'degraded' },
  { name: 'Mumbai', coordinates: [72.8777, 19.0760], latency: 145, status: 'offline' },
];

// Utility functions
export const getStatusColor = (status: 'healthy' | 'warning' | 'critical' | 'online' | 'degraded' | 'offline'): string => {
  switch (status) {
    case 'healthy':
    case 'online':
      return 'success';
    case 'warning':
    case 'degraded':
      return 'warning';
    case 'critical':
    case 'offline':
      return 'error';
    default:
      return 'primary';
  }
};

export const getChangeColor = (change: number): string => {
  if (change > 0) return 'text-success-500 dark:text-success-400';
  if (change < 0) return 'text-error-500 dark:text-error-400';
  return 'text-gray-500 dark:text-gray-400';
};

export const getProfitLossColor = (value: number): string => {
  if (value > 0) return 'text-success-500 dark:text-success-400';
  if (value < 0) return 'text-error-500 dark:text-error-400';
  return 'text-gray-500 dark:text-gray-400';
};