import React, { useMemo, memo } from 'react';
import { format } from 'date-fns';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import Card from '../ui/Card';
import { ActivityData, TimeRange } from '../../types';

interface ActivityChartProps {
  data: ActivityData[];
  timeRange: TimeRange;
}

// Memorizirane pomožne funkcije
const getFormatXAxis = (timeRange: TimeRange) => {
  return (timestamp: string) => {
    const date = new Date(timestamp);
    switch (timeRange) {
      case '1h':
        return format(date, 'HH:mm');
      case '24h':
        return format(date, 'HH:mm');
      case '7d':
        return format(date, 'MMM dd');
      case '30d':
        return format(date, 'MMM dd');
      default:
        return format(date, 'HH:mm');
    }
  };
};

const getFormatTooltipTimestamp = (timeRange: TimeRange) => {
  return (timestamp: string) => {
    const date = new Date(timestamp);
    switch (timeRange) {
      case '1h':
      case '24h':
        return format(date, 'HH:mm:ss');
      case '7d':
      case '30d':
        return format(date, 'MMM dd, HH:mm');
      default:
        return format(date, 'MMM dd, HH:mm:ss');
    }
  };
};

// Memorizirana komponenta za tooltip
interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
  formatTimestamp: (timestamp: string) => string;
}

const CustomTooltip = memo(({ active, payload, label, formatTimestamp }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white dark:bg-dark-card shadow-lg p-3 rounded-lg border border-gray-200 dark:border-dark-border">
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
          {formatTimestamp(label)}
        </p>
        <div className="space-y-1">
          <p className="text-sm">
            <span className="font-medium text-primary-600 dark:text-primary-400">Transactions: </span>
            <span className="text-gray-900 dark:text-white">{payload[0].value}</span>
          </p>
          <p className="text-sm">
            <span className="font-medium text-secondary-600 dark:text-secondary-400">Profit: </span>
            <span className="text-gray-900 dark:text-white">{payload[1].value.toFixed(4)} ETH</span>
          </p>
        </div>
      </div>
    );
  }
  return null;
});

const ActivityChart: React.FC<ActivityChartProps> = ({ data, timeRange }) => {
  // Memorizirane funkcije za formatiranje
  const formatXAxis = useMemo(() => getFormatXAxis(timeRange), [timeRange]);
  const formatTooltipTimestamp = useMemo(() => getFormatTooltipTimestamp(timeRange), [timeRange]);

  // Memorizirana legenda
  const Legend = useMemo(() => (
    <div className="flex space-x-4">
      <div className="flex items-center">
        <div className="w-3 h-3 rounded-full bg-primary-400"></div>
        <span className="ml-2 text-sm text-gray-600 dark:text-gray-300">Transactions</span>
      </div>
      <div className="flex items-center">
        <div className="w-3 h-3 rounded-full bg-secondary-400"></div>
        <span className="ml-2 text-sm text-gray-600 dark:text-gray-300">Profit</span>
      </div>
    </div>
  ), []);

  // Memorizirane definicije gradientov
  const ChartDefs = useMemo(() => (
    <defs>
      <linearGradient id="colorTransactions" x1="0" y1="0" x2="0" y2="1">
        <stop offset="5%" stopColor="#6D94ED" stopOpacity={0.3} />
        <stop offset="95%" stopColor="#6D94ED" stopOpacity={0} />
      </linearGradient>
      <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
        <stop offset="5%" stopColor="#4FD1C5" stopOpacity={0.3} />
        <stop offset="95%" stopColor="#4FD1C5" stopOpacity={0} />
      </linearGradient>
    </defs>
  ), []);

  return (
    <Card title="Real-time Activity">
      <div className="flex justify-between items-center mb-4">
        {Legend}
      </div>

      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
          >
            {ChartDefs}
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatXAxis}
              tick={{ fontSize: 12, fill: '#6B7280' }}
              stroke="#9CA3AF"
            />
            <YAxis
              yAxisId="left"
              orientation="left"
              stroke="#6D94ED"
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => value.toFixed(0)}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#4FD1C5"
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip content={<CustomTooltip formatTimestamp={formatTooltipTimestamp} />} />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="transactions"
              stroke="#6D94ED"
              fillOpacity={1}
              fill="url(#colorTransactions)"
              strokeWidth={2}
            />
            <Area
              yAxisId="right"
              type="monotone"
              dataKey="profit"
              stroke="#4FD1C5"
              fillOpacity={1}
              fill="url(#colorProfit)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};

export default memo(ActivityChart);