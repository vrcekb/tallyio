import { generateActivityData, generateRecentTransactions } from '../mockData';
import { TimeRange } from '../types';

// Utility function to fetch mock data with a delay to simulate API calls
// Zmanjšamo privzeto zakasnitev na 100ms namesto 500ms za boljšo odzivnost
export const fetchMockData = <T>(data: T, delay: number = 100): Promise<T> => {
  // V produkcijskem načinu bi lahko zakasnitev popolnoma odstranili
  if (import.meta.env.PROD) {
    return Promise.resolve(data);
  }

  return new Promise((resolve) => {
    setTimeout(() => resolve(data), delay);
  });
};

// Cache za podatke o aktivnosti
const activityDataCache: Record<TimeRange, { data: unknown; timestamp: number }> = {
  '1h': { data: null, timestamp: 0 },
  '24h': { data: null, timestamp: 0 },
  '7d': { data: null, timestamp: 0 },
  '30d': { data: null, timestamp: 0 }
};

// Fetch activity data for the dashboard - z uporabo predpomnjenja
export const fetchActivityData = async (range: TimeRange): Promise<unknown> => {
  // Preverimo, če imamo veljavne predpomnjene podatke (veljavni 10 sekund)
  const now = Date.now();
  const cacheEntry = activityDataCache[range];

  if (cacheEntry.data && now - cacheEntry.timestamp < 10000) {
    return Promise.resolve(cacheEntry.data);
  }

  // Če nimamo veljavnih podatkov, jih generiramo
  const countMap: Record<TimeRange, number> = {
    '1h': 12,
    '24h': 24,
    '7d': 14,
    '30d': 30
  };

  const count = countMap[range];
  const data = generateActivityData(count, range);

  // Shranimo v predpomnilnik
  activityDataCache[range] = { data, timestamp: now };

  return fetchMockData(data);
};

// Cache za transakcije
let transactionsCache: { data: unknown; timestamp: number } = { data: null, timestamp: 0 };

// Fetch recent transactions - z uporabo predpomnjenja
export const fetchRecentTransactions = async (count: number = 10): Promise<unknown> => {
  // Preverimo, če imamo veljavne predpomnjene podatke (veljavni 10 sekund)
  const now = Date.now();

  if (transactionsCache.data && now - transactionsCache.timestamp < 10000) {
    return Promise.resolve(transactionsCache.data);
  }

  // Če nimamo veljavnih podatkov, jih generiramo
  const transactions = generateRecentTransactions(count);

  // Shranimo v predpomnilnik
  transactionsCache = { data: transactions, timestamp: now };

  return fetchMockData(transactions);
};

// Format currency values
export const formatCurrency = (value: number, precision: number = 4): string => {
  return value.toFixed(precision);
};

// Format percentage values
export const formatPercentage = (value: number): string => {
  return `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
};

// Format latency values
export const formatLatency = (value: number): string => {
  if (value < 1) {
    return `${(value * 1000).toFixed(0)}μs`;
  }
  return `${value.toFixed(0)}ms`;
};

// Format timestamp
export const formatTimestamp = (timestamp: string): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// Get color for status
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

// Get color for change value
export const getChangeColor = (change: number): string => {
  if (change > 0) return 'success';
  if (change < 0) return 'error';
  return 'text-gray-500 dark:text-gray-400';
};

// Get color for profit/loss value
export const getProfitLossColor = (value: number): string => {
  if (value > 0) return 'text-success-500';
  if (value < 0) return 'text-error-500';
  return 'text-gray-500 dark:text-gray-400';
};