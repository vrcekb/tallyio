import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import TransactionItem from './TransactionItem';
import Card from './Card';
import EmptyState from './EmptyState';
import { FileText } from 'lucide-react';

type TransactionStatus = 'pending' | 'success' | 'failed' | 'cancelled';

interface Transaction {
  id: string | number;
  hash: string;
  status: TransactionStatus;
  timestamp: string | Date;
  network: string;
  from: string;
  to: string;
  value?: string | number;
  gas?: string | number;
  explorerUrl?: string;
}

interface TransactionListProps {
  transactions: Transaction[];
  title?: string;
  emptyState?: React.ReactNode;
  onTransactionClick?: (transaction: Transaction) => void;
  className?: string;
  itemClassName?: string;
  animate?: boolean;
  showCard?: boolean;
  maxHeight?: number | string;
}

/**
 * Komponenta za prikaz seznama transakcij
 */
const TransactionList: React.FC<TransactionListProps> = ({
  transactions,
  title = 'Recent Transactions',
  emptyState,
  onTransactionClick,
  className = '',
  itemClassName = '',
  animate = true,
  showCard = true,
  maxHeight = 600
}) => {
  const defaultEmptyState = (
    <EmptyState
      title="No transactions"
      description="There are no transactions to display at this time."
      icon={<FileText size={24} />}
    />
  );

  const content = (
    <div className={className}>
      {title && (
        <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100 mb-4">
          {title}
        </h3>
      )}
      
      {transactions.length === 0 ? (
        emptyState || defaultEmptyState
      ) : (
        <div 
          className="space-y-3 overflow-y-auto pr-1"
          style={{ maxHeight }}
        >
          <AnimatePresence>
            {transactions.map((transaction) => (
              <TransactionItem
                key={transaction.id}
                hash={transaction.hash}
                status={transaction.status}
                timestamp={transaction.timestamp}
                network={transaction.network}
                from={transaction.from}
                to={transaction.to}
                value={transaction.value}
                gas={transaction.gas}
                explorerUrl={transaction.explorerUrl}
                onClick={onTransactionClick ? () => onTransactionClick(transaction) : undefined}
                className={itemClassName}
                animate={animate}
              />
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );

  if (!showCard) {
    return content;
  }

  return (
    <Card>
      {content}
    </Card>
  );
};

export default TransactionList;
