import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';
import Badge from './Badge';
import ProgressBar from './ProgressBar';

type NetworkStatus = 'online' | 'degraded' | 'offline';

interface NetworkItemProps {
  name: string;
  status: NetworkStatus;
  icon?: React.ReactNode;
  gasPrice?: number;
  gasTrend?: number;
  blockHeight?: number;
  connections?: number;
  latency?: number;
  mempoolSize?: number;
  onClick?: () => void;
  className?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz elementa omrežja
 */
const NetworkItem: React.FC<NetworkItemProps> = ({
  name,
  status,
  icon,
  gasPrice,
  gasTrend,
  blockHeight,
  connections,
  latency,
  mempoolSize,
  onClick,
  className = '',
  animate = true
}) => {
  const getStatusBadge = () => {
    switch (status) {
      case 'online':
        return <Badge color="success" variant="subtle">Online</Badge>;
      case 'degraded':
        return <Badge color="warning" variant="subtle">Degraded</Badge>;
      default:
        return <Badge color="error" variant="subtle">Offline</Badge>;
    }
  };

  const formatGasPrice = (price: number) => {
    if (price < 1) {
      return `${(price * 1000).toFixed(0)} gwei`;
    }
    return `${price.toFixed(2)} gwei`;
  };

  const getLatencyColor = (latency: number) => {
    if (latency < 100) return 'success';
    if (latency < 500) return 'warning';
    return 'error';
  };

  const getConnectionsPercentage = (connections: number) => {
    // Assuming 10 connections is the maximum for good health
    return Math.min(connections / 10 * 100, 100);
  };

  const networkContent = (
    <div 
      className={`
        p-4 border border-gray-200 dark:border-dark-border rounded-lg
        ${onClick ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-dark-background' : ''}
        transition-colors duration-200
        ${className}
      `}
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          {icon && (
            <div className="w-8 h-8 mr-3 flex-shrink-0">
              {icon}
            </div>
          )}
          <div>
            <h3 className="font-medium text-primary-900 dark:text-primary-100">
              {name}
            </h3>
          </div>
        </div>
        <div>
          {getStatusBadge()}
        </div>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 mt-3">
        {gasPrice !== undefined && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Gas Price
              </span>
              <div className="flex items-center">
                {gasTrend !== undefined && (
                  <div className={`
                    flex items-center text-xs mr-1
                    ${gasTrend > 0 ? 'text-error-500' : gasTrend < 0 ? 'text-success-500' : 'text-gray-500'}
                  `}>
                    {gasTrend > 0 ? (
                      <TrendingUp size={12} className="mr-0.5" />
                    ) : gasTrend < 0 ? (
                      <TrendingDown size={12} className="mr-0.5" />
                    ) : null}
                    <span>
                      {gasTrend > 0 && '+'}
                      {gasTrend}%
                    </span>
                  </div>
                )}
                <span className="text-xs font-medium text-primary-900 dark:text-primary-100">
                  {formatGasPrice(gasPrice)}
                </span>
              </div>
            </div>
          </div>
        )}
        
        {blockHeight !== undefined && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Block Height
              </span>
              <span className="text-xs font-medium text-primary-900 dark:text-primary-100">
                {blockHeight.toLocaleString()}
              </span>
            </div>
          </div>
        )}
        
        {connections !== undefined && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Connections
              </span>
              <span className="text-xs font-medium text-primary-900 dark:text-primary-100">
                {connections}
              </span>
            </div>
            <ProgressBar
              value={connections}
              max={10}
              size="sm"
              color={connections >= 5 ? 'success' : connections >= 2 ? 'warning' : 'error'}
              showValue={false}
            />
          </div>
        )}
        
        {latency !== undefined && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Latency
              </span>
              <span className="text-xs font-medium text-primary-900 dark:text-primary-100">
                {latency} ms
              </span>
            </div>
            <ProgressBar
              value={100 - Math.min(latency / 10, 100)}
              size="sm"
              color={getLatencyColor(latency)}
              showValue={false}
            />
          </div>
        )}
        
        {mempoolSize !== undefined && (
          <div className="sm:col-span-2 mt-1">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Mempool Size
              </span>
              <span className="text-xs font-medium text-primary-900 dark:text-primary-100">
                {mempoolSize.toLocaleString()} txs
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  if (!animate) {
    return networkContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {networkContent}
    </motion.div>
  );
};

export default NetworkItem;
