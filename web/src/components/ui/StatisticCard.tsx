import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StatisticCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon?: React.ReactNode;
  className?: string;
  valueClassName?: string;
  iconClassName?: string;
  changeClassName?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz statističnih podatkov
 */
const StatisticCard: React.FC<StatisticCardProps> = ({
  title,
  value,
  change,
  icon,
  className = '',
  valueClassName = '',
  iconClassName = '',
  changeClassName = '',
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

  const cardContent = (
    <div className={`
      relative p-6 bg-white dark:bg-dark-card rounded-lg border border-primary-100/50 dark:border-dark-border
      transition-all duration-300 hover:shadow-md group
      ${className}
    `}>
      <div className="flex items-center justify-between mb-4">
        {icon && (
          <div className={`
            w-12 h-12 rounded-lg bg-primary-50 dark:bg-primary-900/20 
            flex items-center justify-center transition-transform duration-300 
            group-hover:scale-110 ${iconClassName}
          `}>
            {icon}
          </div>
        )}
        {change !== undefined && (
          <div className={`
            flex items-center ${getChangeColor(change)} 
            transition-transform duration-300 group-hover:scale-105
            ${changeClassName}
          `}>
            {change > 0 ? (
              <TrendingUp size={16} className="mr-1" />
            ) : change < 0 ? (
              <TrendingDown size={16} className="mr-1" />
            ) : null}
            <span className="text-sm font-medium">
              {change > 0 && '+'}
              {change}%
            </span>
          </div>
        )}
      </div>
      <h3 className="text-sm font-medium text-primary-600 dark:text-primary-400 mb-2">{title}</h3>
      <div className="flex items-baseline">
        <span className={`
          text-2xl font-bold text-primary-900 dark:text-primary-50 
          transition-all duration-300 group-hover:text-primary-700 dark:group-hover:text-primary-300
          ${valueClassName}
        `}>
          {value}
        </span>
      </div>
      <div className="absolute inset-x-0 bottom-0 h-0.5 bg-gradient-to-r from-primary-500/0 via-primary-500/10 to-primary-500/0 dark:from-primary-400/0 dark:via-primary-400/10 dark:to-primary-400/0 transition-opacity duration-300 opacity-0 group-hover:opacity-100"></div>
    </div>
  );

  if (!animate) {
    return cardContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
    >
      {cardContent}
    </motion.div>
  );
};

export default StatisticCard;
