import { format } from 'date-fns';

export interface PerformanceMetric {
  title: string;
  value: number;
  unit: string;
  change: number;
  threshold: number;
  status: 'good' | 'warning' | 'critical';
}

export interface ResourceUsage {
  timestamp: string;
  cpu: number;
  memory: number;
  network: number;
  disk: number;
}

export interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  uptime: number;
  responseTime: number;
  errorRate: number;
}

export const performanceMetrics: PerformanceMetric[] = [
  {
    title: 'Average Response Time',
    value: 124,
    unit: 'ms',
    change: -12.5,
    threshold: 200,
    status: 'good'
  },
  {
    title: 'Error Rate',
    value: 0.12,
    unit: '%',
    change: 0.05,
    threshold: 1,
    status: 'good'
  },
  {
    title: 'CPU Usage',
    value: 82,
    unit: '%',
    change: 15.3,
    threshold: 85,
    status: 'warning'
  },
  {
    title: 'Memory Usage',
    value: 76,
    unit: '%',
    change: 8.2,
    threshold: 90,
    status: 'good'
  }
];

export const generateResourceUsage = (hours: number): ResourceUsage[] => {
  const data: ResourceUsage[] = [];
  const now = new Date();

  for (let i = hours; i >= 0; i--) {
    const timestamp = new Date(now);
    timestamp.setHours(timestamp.getHours() - i);

    data.push({
      timestamp: format(timestamp, 'HH:mm'),
      cpu: Math.random() * 30 + 60,
      memory: Math.random() * 20 + 65,
      network: Math.random() * 40 + 30,
      disk: Math.random() * 15 + 70
    });
  }

  return data;
};

export const serviceHealth: ServiceHealth[] = [
  {
    name: 'Transaction Processor',
    status: 'healthy',
    uptime: 99.998,
    responseTime: 45,
    errorRate: 0.001
  },
  {
    name: 'Strategy Engine',
    status: 'healthy',
    uptime: 99.995,
    responseTime: 78,
    errorRate: 0.003
  },
  {
    name: 'Data Aggregator',
    status: 'degraded',
    uptime: 99.942,
    responseTime: 156,
    errorRate: 0.012
  },
  {
    name: 'Risk Manager',
    status: 'healthy',
    uptime: 99.999,
    responseTime: 32,
    errorRate: 0.001
  },
  {
    name: 'ML Pipeline',
    status: 'healthy',
    uptime: 99.997,
    responseTime: 89,
    errorRate: 0.002
  }
];