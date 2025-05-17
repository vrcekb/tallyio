import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Strategy } from '../../types';
import Card from '../ui/Card';
// import { Twitch as Switch } from 'lucide-react';

interface TopStrategiesChartProps {
  data: Strategy[];
}

const TopStrategiesChart: React.FC<TopStrategiesChartProps> = ({ data }) => {
  // Sort strategies by profit in descending order and take top 5
  const sortedData = [...data].sort((a, b) => b.profit - a.profit).slice(0, 5);

  // Custom tooltip component
  interface CustomTooltipProps {
    active?: boolean;
    payload?: Array<{ payload: Strategy }>;
    // label parameter is not used
  }

  const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
    if (active && payload && payload.length) {
      const strategy = payload[0].payload;
      return (
        <div className="bg-white dark:bg-dark-card shadow-lg p-3 rounded-lg border border-gray-200 dark:border-dark-border">
          <p className="text-sm font-medium text-gray-900 dark:text-white mb-1">{strategy.name}</p>
          <div className="space-y-1">
            <p className="text-xs">
              <span className="text-gray-500 dark:text-gray-400">Profit: </span>
              <span className="font-medium text-gray-900 dark:text-white">{strategy.profit.toFixed(3)} ETH</span>
            </p>
            <p className="text-xs">
              <span className="text-gray-500 dark:text-gray-400">Change: </span>
              <span className={`font-medium ${strategy.change >= 0 ? 'text-success-500' : 'text-error-500'}`}>
                {strategy.change > 0 && '+'}
                {strategy.change.toFixed(1)}%
              </span>
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <Card title="Top Performing Strategies">
      <div className="h-[180px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sortedData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
            <XAxis
              type="number"
              domain={[0, Math.max(...sortedData.map(item => item.profit)) * 1.2]}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              dataKey="name"
              type="category"
              width={120}
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="profit" radius={[0, 4, 4, 0]}>
              {sortedData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.change >= 0 ? '#4ADE80' : '#F87171'}
                  fillOpacity={0.8}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 overflow-auto max-h-[170px]">
        {sortedData.map((strategy) => (
          <div
            key={strategy.id}
            className="flex items-center justify-between py-3 border-b border-gray-100 dark:border-dark-border last:border-0 hover:bg-primary-50/30 dark:hover:bg-dark-background/30 transition-all duration-300 hover:translate-x-1"
          >
            <div className="flex items-center">
              <span className="w-3 h-3 rounded-full mr-2 bg-primary-500"></span>
              <span className="text-sm font-medium text-gray-800 dark:text-gray-200">{strategy.name}</span>
            </div>
            <div className="flex items-center">
              <span className={`text-xs font-medium mr-3 ${strategy.change >= 0 ? 'text-success-500' : 'text-error-500'}`}>
                {strategy.change > 0 && '+'}
                {strategy.change}%
              </span>
              <div className="relative inline-block w-10 h-4 align-middle select-none transition duration-200 ease-in">
                <input
                  type="checkbox"
                  name={`strategy-${strategy.id}`}
                  id={`strategy-${strategy.id}`}
                  className="toggle-checkbox absolute block w-4 h-4 rounded-full bg-white border-2 border-gray-300 appearance-none cursor-pointer"
                  defaultChecked={strategy.enabled}
                />
                <label
                  htmlFor={`strategy-${strategy.id}`}
                  className={`toggle-label block overflow-hidden h-4 rounded-full cursor-pointer ${
                    strategy.enabled ? 'bg-primary-500' : 'bg-gray-300'
                  }`}
                ></label>
              </div>
            </div>
          </div>
        ))}
      </div>


    </Card>
  );
};

export default TopStrategiesChart;