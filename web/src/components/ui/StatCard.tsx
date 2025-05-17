import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';
import Card from './Card';

interface StatCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  chart?: React.ReactNode;
  icon?: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
  iconClassName?: string;
  valueClassName?: string;
  chartClassName?: string;
  footerClassName?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz kartice s statistiko
 */
const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  change,
  changeLabel,
  chart,
  icon,
  footer,
  className = '',
  iconClassName = '',
  valueClassName = '',
  chartClassName = '',
  footerClassName = '',
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
    <Card className={`${className} group`}>
      <div className="flex flex-col h-full">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-medium text-primary-600 dark:text-primary-400">{title}</h3>
            <div className="flex items-baseline mt-1">
              <div className={`text-2xl font-bold text-primary-900 dark:text-primary-50 ${valueClassName}`}>
                {value}
              </div>
              
              {change !== undefined && (
                <div className={`
                  ml-2 flex items-center text-sm font-medium ${getChangeColor(change)}
                `}>
                  {change > 0 ? (
                    <TrendingUp size={16} className="mr-1" />
                  ) : change < 0 ? (
                    <TrendingDown size={16} className="mr-1" />
                  ) : null}
                  <span>
                    {change > 0 && '+'}
                    {change}%
                  </span>
                  {changeLabel && (
                    <span className="ml-1 text-gray-500 dark:text-gray-400">
                      {changeLabel}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
          
          {icon && (
            <div className={`
              w-10 h-10 rounded-lg bg-primary-50 dark:bg-primary-900/20 
              flex items-center justify-center text-primary-500 dark:text-primary-400
              transition-transform duration-300 group-hover:scale-110
              ${iconClassName}
            `}>
              {icon}
            </div>
          )}
        </div>
        
        {chart && (
          <div className={`mt-2 ${chartClassName}`}>
            {chart}
          </div>
        )}
        
        {footer && (
          <div className={`
            mt-auto pt-4 border-t border-primary-100/50 dark:border-dark-border
            text-sm text-gray-500 dark:text-gray-400 ${footerClassName}
          `}>
            {footer}
          </div>
        )}
      </div>
    </Card>
  );

  if (!animate) {
    return cardContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
    >
      {cardContent}
    </motion.div>
  );
};

export default StatCard;
