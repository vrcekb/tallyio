import React from 'react';
import { motion } from 'framer-motion';

interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showValue?: boolean;
  valueFormat?: (value: number, max: number) => string;
  color?: 'primary' | 'success' | 'warning' | 'error';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz napredka
 */
const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  label,
  showValue = true,
  valueFormat,
  color = 'primary',
  size = 'md',
  className = '',
  animate = true
}) => {
  const percentage = Math.min(Math.max(0, (value / max) * 100), 100);
  
  const getColorClass = () => {
    switch (color) {
      case 'success':
        return 'bg-success-500 dark:bg-success-600';
      case 'warning':
        return 'bg-warning-500 dark:bg-warning-600';
      case 'error':
        return 'bg-error-500 dark:bg-error-600';
      default:
        return 'bg-primary-500 dark:bg-primary-600';
    }
  };
  
  const getSizeClass = () => {
    switch (size) {
      case 'sm':
        return 'h-1.5';
      case 'lg':
        return 'h-3';
      default:
        return 'h-2';
    }
  };
  
  const formatValue = () => {
    if (valueFormat) {
      return valueFormat(value, max);
    }
    return `${Math.round(percentage)}%`;
  };

  const progressContent = (
    <div className={`w-full ${className}`}>
      {(label || showValue) && (
        <div className="flex justify-between items-center mb-1">
          {label && (
            <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
              {label}
            </span>
          )}
          {showValue && (
            <span className="text-xs font-medium text-primary-600 dark:text-primary-400">
              {formatValue()}
            </span>
          )}
        </div>
      )}
      <div className={`w-full bg-gray-200 dark:bg-dark-background rounded-full overflow-hidden ${getSizeClass()}`}>
        {animate ? (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
            className={`${getSizeClass()} ${getColorClass()} rounded-full`}
          />
        ) : (
          <div
            className={`${getSizeClass()} ${getColorClass()} rounded-full`}
            style={{ width: `${percentage}%` }}
          />
        )}
      </div>
    </div>
  );

  return progressContent;
};

export default ProgressBar;
