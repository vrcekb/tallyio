import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';
import Badge from './Badge';
import Switch from './Switch';

type StrategyStatus = 'active' | 'paused' | 'disabled' | 'error';

interface StrategyItemProps {
  name: string;
  description?: string;
  status: StrategyStatus;
  profit?: number;
  change?: number;
  transactions?: number;
  networks?: string[];
  enabled?: boolean;
  onToggle?: (enabled: boolean) => void;
  onClick?: () => void;
  className?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz elementa strategije
 */
const StrategyItem: React.FC<StrategyItemProps> = ({
  name,
  description,
  status,
  profit,
  change,
  transactions,
  networks = [],
  enabled = false,
  onToggle,
  onClick,
  className = '',
  animate = true
}) => {
  const getStatusBadge = () => {
    switch (status) {
      case 'active':
        return <Badge color="success" variant="subtle">Active</Badge>;
      case 'paused':
        return <Badge color="warning" variant="subtle">Paused</Badge>;
      case 'error':
        return <Badge color="error" variant="subtle">Error</Badge>;
      default:
        return <Badge color="gray" variant="subtle">Disabled</Badge>;
    }
  };

  const getChangeColor = (change: number | undefined) => {
    if (change === undefined) return '';
    return change > 0 
      ? 'text-success-500 dark:text-success-400' 
      : change < 0 
        ? 'text-error-500 dark:text-error-400' 
        : 'text-gray-500 dark:text-gray-400';
  };

  const handleToggle = (checked: boolean) => {
    if (onToggle) {
      onToggle(checked);
    }
  };

  const strategyContent = (
    <div 
      className={`
        p-4 border border-gray-200 dark:border-dark-border rounded-lg
        ${onClick ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-dark-background' : ''}
        transition-colors duration-200
        ${className}
      `}
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="font-medium text-primary-900 dark:text-primary-100">
            {name}
          </h3>
          {description && (
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {description}
            </p>
          )}
        </div>
        <div className="flex items-center space-x-3">
          {getStatusBadge()}
          {onToggle && (
            <div onClick={(e) => e.stopPropagation()}>
              <Switch
                checked={enabled}
                onChange={handleToggle}
                size="sm"
                color="primary"
              />
            </div>
          )}
        </div>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
        {profit !== undefined && (
          <div>
            <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">
              Profit
            </span>
            <div className="flex items-center">
              <span className="font-medium text-primary-900 dark:text-primary-100">
                {profit.toFixed(4)} ETH
              </span>
              {change !== undefined && (
                <div className={`ml-2 flex items-center text-xs ${getChangeColor(change)}`}>
                  {change > 0 ? (
                    <TrendingUp size={12} className="mr-1" />
                  ) : change < 0 ? (
                    <TrendingDown size={12} className="mr-1" />
                  ) : null}
                  <span>
                    {change > 0 && '+'}
                    {change}%
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
        
        {transactions !== undefined && (
          <div>
            <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">
              Transactions
            </span>
            <span className="font-medium text-primary-900 dark:text-primary-100">
              {transactions}
            </span>
          </div>
        )}
        
        {networks.length > 0 && (
          <div>
            <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">
              Networks
            </span>
            <div className="flex flex-wrap gap-1">
              {networks.map((network, index) => (
                <Badge key={index} size="sm" color="primary" variant="subtle">
                  {network}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  if (!animate) {
    return strategyContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {strategyContent}
    </motion.div>
  );
};

export default StrategyItem;
