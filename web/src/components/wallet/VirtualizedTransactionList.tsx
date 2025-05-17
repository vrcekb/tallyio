import React, { useCallback, useMemo } from 'react';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { Send, Download, ExternalLink } from 'lucide-react';
import Button from '../ui/Button';
import Badge from '../ui/Badge';
import Tooltip from '../ui/Tooltip';

export interface Transaction {
  id: string;
  type: 'send' | 'receive';
  amount: string;
  token: string;
  address: string;
  timestamp: string;
  status: 'completed' | 'pending' | 'failed';
}

interface VirtualizedTransactionListProps {
  transactions: Transaction[];
  onViewDetails: (txId: string) => void;
  className?: string;
  emptyMessage?: string;
}

/**
 * Virtualiziran seznam transakcij za izboljšano zmogljivost pri velikem številu transakcij
 */
const VirtualizedTransactionList: React.FC<VirtualizedTransactionListProps> = ({
  transactions,
  onViewDetails,
  className = '',
  emptyMessage = 'Ni transakcij za prikaz'
}) => {
  // Memoizacija funkcije za izris posamezne transakcije
  const renderTransaction = useCallback(({ index, style }: { index: number; style: React.CSSProperties }) => {
    const tx = transactions[index];
    return (
      <div 
        key={tx.id}
        style={style}
        className="px-2"
      >
        <div className="flex items-center justify-between p-4 bg-primary-50 dark:bg-dark-background rounded-lg transition-all duration-300 hover:shadow-md mb-2">
          <div className="flex items-center">
            {tx.type === 'send' ? (
              <Send size={18} className="text-primary-500" />
            ) : (
              <Download size={18} className="text-primary-500" />
            )}
            <div className="ml-3">
              <p className="text-sm font-medium text-primary-900 dark:text-primary-100">
                {tx.type === 'send' ? 'Sent' : 'Received'} {tx.amount} {tx.token}
              </p>
              <p className="text-xs text-primary-500 dark:text-primary-400">
                {new Date(tx.timestamp).toLocaleString()}
              </p>
            </div>
          </div>
          <div className="flex items-center">
            <Badge
              variant="subtle"
              color={
                tx.status === 'completed'
                  ? 'success'
                  : tx.status === 'pending'
                  ? 'warning'
                  : 'error'
              }
              size="sm"
            >
              {tx.status.charAt(0).toUpperCase() + tx.status.slice(1)}
            </Badge>
            <Tooltip content="View transaction details">
              <Button
                variant="ghost"
                size="xs"
                onClick={() => onViewDetails(tx.id)}
              >
                <ExternalLink size={16} className="text-primary-400" />
              </Button>
            </Tooltip>
          </div>
        </div>
      </div>
    );
  }, [transactions, onViewDetails]);

  // Memoizacija višine posamezne vrstice
  const itemSize = useMemo(() => 80, []);

  // Prikaz sporočila, če ni transakcij
  if (transactions.length === 0) {
    return (
      <div className="p-4 text-center text-primary-500 dark:text-primary-400">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className={`w-full ${className}`} style={{ height: '400px' }}>
      <AutoSizer>
        {({ height, width }) => (
          <List
            className="scrollbar-thin scrollbar-thumb-primary-200 dark:scrollbar-thumb-primary-800 scrollbar-track-transparent"
            height={height}
            width={width}
            itemCount={transactions.length}
            itemSize={itemSize}
          >
            {renderTransaction}
          </List>
        )}
      </AutoSizer>
    </div>
  );
};

export default VirtualizedTransactionList;
