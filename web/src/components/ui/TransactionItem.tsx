import React from 'react';
import { motion } from 'framer-motion';
import { ExternalLink } from 'lucide-react';
import Badge from './Badge';

type TransactionStatus = 'pending' | 'success' | 'failed' | 'cancelled';

interface TransactionItemProps {
  hash: string;
  status: TransactionStatus;
  timestamp: string | Date;
  network: string;
  from: string;
  to: string;
  value?: string | number;
  gas?: string | number;
  explorerUrl?: string;
  onClick?: () => void;
  className?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz elementa transakcije
 */
const TransactionItem: React.FC<TransactionItemProps> = ({
  hash,
  status,
  timestamp,
  network,
  from,
  to,
  value,
  gas,
  explorerUrl,
  onClick,
  className = '',
  animate = true
}) => {
  const getStatusBadge = () => {
    switch (status) {
      case 'success':
        return <Badge color="success" variant="subtle">Success</Badge>;
      case 'failed':
        return <Badge color="error" variant="subtle">Failed</Badge>;
      case 'cancelled':
        return <Badge color="gray" variant="subtle">Cancelled</Badge>;
      default:
        return <Badge color="warning" variant="subtle">Pending</Badge>;
    }
  };

  const formatTimestamp = (timestamp: string | Date) => {
    if (!timestamp) return '';
    
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    
    return date.toLocaleString([], {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const truncateAddress = (address: string) => {
    if (!address) return '';
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
  };

  const truncateHash = (hash: string) => {
    if (!hash) return '';
    return `${hash.substring(0, 10)}...${hash.substring(hash.length - 6)}`;
  };

  const transactionContent = (
    <div 
      className={`
        p-4 border border-gray-200 dark:border-dark-border rounded-lg
        ${onClick ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-dark-background' : ''}
        transition-colors duration-200
        ${className}
      `}
      onClick={onClick}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-2">
        <div className="flex items-center mb-2 sm:mb-0">
          <span className="font-medium text-primary-900 dark:text-primary-100 mr-2">
            {truncateHash(hash)}
          </span>
          {explorerUrl && (
            <a
              href={explorerUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-500 hover:text-primary-600 dark:text-primary-400 dark:hover:text-primary-300"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLink size={14} />
            </a>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <Badge color="primary" variant="subtle">{network}</Badge>
          {getStatusBadge()}
        </div>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-gray-500 dark:text-gray-400">From: </span>
          <span className="font-medium text-primary-900 dark:text-primary-100">
            {truncateAddress(from)}
          </span>
        </div>
        <div>
          <span className="text-gray-500 dark:text-gray-400">To: </span>
          <span className="font-medium text-primary-900 dark:text-primary-100">
            {truncateAddress(to)}
          </span>
        </div>
        {value !== undefined && (
          <div>
            <span className="text-gray-500 dark:text-gray-400">Value: </span>
            <span className="font-medium text-primary-900 dark:text-primary-100">
              {value}
            </span>
          </div>
        )}
        {gas !== undefined && (
          <div>
            <span className="text-gray-500 dark:text-gray-400">Gas: </span>
            <span className="font-medium text-primary-900 dark:text-primary-100">
              {gas}
            </span>
          </div>
        )}
      </div>
      
      <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
        {formatTimestamp(timestamp)}
      </div>
    </div>
  );

  if (!animate) {
    return transactionContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {transactionContent}
    </motion.div>
  );
};

export default TransactionItem;
