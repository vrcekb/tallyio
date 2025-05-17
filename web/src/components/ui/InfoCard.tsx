import React from 'react';
import { motion } from 'framer-motion';
import Card from './Card';

interface InfoCardProps {
  title: string;
  value: string | number | React.ReactNode;
  description?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    label?: string;
    icon?: React.ReactNode;
  };
  footer?: React.ReactNode;
  className?: string;
  iconClassName?: string;
  valueClassName?: string;
  descriptionClassName?: string;
  trendClassName?: string;
  footerClassName?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz kartice z informacijami
 */
const InfoCard: React.FC<InfoCardProps> = ({
  title,
  value,
  description,
  icon,
  trend,
  footer,
  className = '',
  iconClassName = '',
  valueClassName = '',
  descriptionClassName = '',
  trendClassName = '',
  footerClassName = '',
  animate = true
}) => {
  const getTrendColor = (value: number) => {
    if (value > 0) return 'text-success-500 dark:text-success-400';
    if (value < 0) return 'text-error-500 dark:text-error-400';
    return 'text-gray-500 dark:text-gray-400';
  };

  const cardContent = (
    <Card className={className}>
      <div className="flex flex-col h-full">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-medium text-primary-600 dark:text-primary-400">{title}</h3>
          </div>
          {icon && (
            <div className={`
              w-10 h-10 rounded-lg bg-primary-50 dark:bg-primary-900/20 
              flex items-center justify-center text-primary-500 dark:text-primary-400
              ${iconClassName}
            `}>
              {icon}
            </div>
          )}
        </div>
        
        <div className="flex items-baseline">
          <div className={`text-2xl font-bold text-primary-900 dark:text-primary-50 ${valueClassName}`}>
            {value}
          </div>
          
          {trend && (
            <div className={`
              ml-2 flex items-center text-sm font-medium ${getTrendColor(trend.value)} ${trendClassName}
            `}>
              {trend.icon}
              <span>
                {trend.value > 0 && '+'}
                {trend.value}%
              </span>
              {trend.label && (
                <span className="ml-1 text-gray-500 dark:text-gray-400">
                  {trend.label}
                </span>
              )}
            </div>
          )}
        </div>
        
        {description && (
          <p className={`mt-2 text-sm text-gray-500 dark:text-gray-400 ${descriptionClassName}`}>
            {description}
          </p>
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

export default InfoCard;
