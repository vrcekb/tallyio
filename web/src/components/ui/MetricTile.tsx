import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface MetricTileProps {
  title: string;
  value: string | number;
  unit?: string;
  change?: number;
  icon?: React.ReactNode;
  color?: string;
  className?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz metrike
 */
const MetricTile: React.FC<MetricTileProps> = ({
  title,
  value,
  unit,
  change,
  icon,
  color = 'bg-primary-500',
  className = '',
  animate = true
}) => {
  const getChangeColor = (change: number | undefined) => {
    if (change === undefined) return '';
    return change > 0 
      ? 'text-success-500 dark:text-success-400' 
      : change < 0 
        ? 'text-error-500 dark:text-error-400' 
        : 'text-gray-500 dark:text-gray-400';
  };

  const tileContent = (
    <div className={`
      relative p-4 bg-white dark:bg-dark-card rounded-lg border border-primary-100/50 dark:border-dark-border
      transition-all duration-300 hover:shadow-md group
      ${className}
    `}>
      <div className="flex items-center mb-2">
        {icon && (
          <div className={`
            w-8 h-8 rounded-lg ${color} flex items-center justify-center mr-3
          `}>
            {icon}
          </div>
        )}
        <h3 className="text-sm font-medium text-primary-600 dark:text-primary-400">{title}</h3>
      </div>
      
      <div className="flex items-baseline">
        <span className="text-xl font-bold text-primary-900 dark:text-primary-50">
          {value}
        </span>
        {unit && (
          <span className="ml-1 text-sm text-primary-500 dark:text-primary-400">{unit}</span>
        )}
      </div>
      
      {change !== undefined && (
        <div className={`
          flex items-center mt-2 text-xs font-medium ${getChangeColor(change)}
        `}>
          {change > 0 ? (
            <TrendingUp size={14} className="mr-1" />
          ) : change < 0 ? (
            <TrendingDown size={14} className="mr-1" />
          ) : null}
          <span>
            {change > 0 && '+'}
            {change}%
          </span>
        </div>
      )}
    </div>
  );

  if (!animate) {
    return tileContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
    >
      {tileContent}
    </motion.div>
  );
};

export default MetricTile;
