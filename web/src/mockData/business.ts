import { format } from 'date-fns';

export interface RevenueData {
  date: string;
  revenue: number;
  expenses: number;
  profit: number;
}

export interface CustomerMetric {
  title: string;
  value: number;
  change: number;
  trend: 'up' | 'down' | 'stable';
}

export interface MarketShare {
  network: string;
  share: number;
  volume: number;
  change: number;
}

export const generateRevenueData = (days: number): RevenueData[] => {
  const data: RevenueData[] = [];
  const now = new Date();

  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    
    const revenue = Math.random() * 100 + 50;
    const expenses = Math.random() * 40 + 20;
    
    data.push({
      date: format(date, 'yyyy-MM-dd'),
      revenue: parseFloat(revenue.toFixed(2)),
      expenses: parseFloat(expenses.toFixed(2)),
      profit: parseFloat((revenue - expenses).toFixed(2))
    });
  }

  return data;
};

export const customerMetrics: CustomerMetric[] = [
  {
    title: 'Total Users',
    value: 12458,
    change: 12.5,
    trend: 'up'
  },
  {
    title: 'Active Users',
    value: 8234,
    change: 8.2,
    trend: 'up'
  },
  {
    title: 'New Users (24h)',
    value: 342,
    change: -2.1,
    trend: 'down'
  },
  {
    title: 'User Retention',
    value: 92.4,
    change: 1.5,
    trend: 'up'
  }
];

export const marketShareData: MarketShare[] = [
  {
    network: 'Ethereum',
    share: 45.2,
    volume: 1245789,
    change: 5.2
  },
  {
    network: 'Arbitrum',
    share: 22.8,
    volume: 628456,
    change: 12.4
  },
  {
    network: 'Optimism',
    share: 15.6,
    volume: 429873,
    change: 8.7
  },
  {
    network: 'Base',
    share: 9.8,
    volume: 269745,
    change: 15.2
  },
  {
    network: 'Polygon',
    share: 6.6,
    volume: 181234,
    change: -2.3
  }
];