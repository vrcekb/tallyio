import React from 'react';
import { TrendingUp, TrendingDown, DivideIcon as LucideIcon } from 'lucide-react';
import * as LucideIcons from 'lucide-react';
import { KeyMetric } from '../../types';
import { getChangeColor } from '../../utils/mockData';

interface MetricCardProps {
  data: KeyMetric;
}

const MetricCard: React.FC<MetricCardProps> = ({ data }) => {
  const IconComponent = (LucideIcons as Record<string, LucideIcon>)[
    data.icon.charAt(0).toUpperCase() + data.icon.slice(1)
  ] || TrendingUp;

  return (
    <div className="relative h-[160px] bg-white dark:bg-dark-card rounded-lg border border-primary-100 dark:border-dark-border p-6 transition-all duration-300 group flex flex-col shadow-sm hover:shadow-md">
      <div className="flex items-center justify-between mb-4">
        <div className="w-12 h-12 rounded-lg bg-primary-50 dark:bg-primary-900/20 flex items-center justify-center transition-transform duration-300 group-hover:scale-110">
          <IconComponent size={24} className="text-primary-600 dark:text-primary-400" />
        </div>
        <div className={`flex items-center ${getChangeColor(data.change)} transition-transform duration-300 group-hover:scale-105`}>
          {data.change > 0 ? (
            <TrendingUp size={16} className="mr-1" />
          ) : data.change < 0 ? (
            <TrendingDown size={16} className="mr-1" />
          ) : null}
          <span className="text-sm font-medium">
            {data.change > 0 && '+'}
            {data.change}%
          </span>
        </div>
      </div>
      <h3 className="text-sm font-medium text-primary-600 dark:text-primary-400 mb-2">{data.title}</h3>
      <div className="flex items-baseline mt-auto">
        <span className="text-2xl font-bold text-primary-900 dark:text-primary-50 transition-all duration-300 group-hover:text-primary-700 dark:group-hover:text-primary-300">{data.value}</span>
        {data.unit && (
          <span className="ml-1 text-sm text-primary-500 dark:text-primary-400">{data.unit}</span>
        )}
      </div>
      <div className="absolute inset-x-0 bottom-0 h-0.5 bg-gradient-to-r from-primary-500/0 via-primary-500/10 to-primary-500/0 dark:from-primary-400/0 dark:via-primary-400/10 dark:to-primary-400/0 transition-opacity duration-300 opacity-0 group-hover:opacity-100"></div>
    </div>
  );
};

export default MetricCard;