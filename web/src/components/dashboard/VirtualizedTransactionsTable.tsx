import React, { useState, useMemo, useCallback, memo } from 'react';
import { CheckCircle, AlertCircle, Clock } from 'lucide-react';
import VirtualizedList from '../ui/VirtualizedList';
import Card from '../ui/Card';
import { Transaction } from '../../types';

interface VirtualizedTransactionsTableProps {
  data: Transaction[];
  title?: string;
  maxHeight?: number | string;
  itemHeight?: number;
  className?: string;
}

/**
 * Virtualizirana tabela transakcij
 * Uporablja VirtualizedList za učinkovit prikaz velikega števila transakcij
 */
const VirtualizedTransactionsTable: React.FC<VirtualizedTransactionsTableProps> = ({
  data,
  title = 'Recent Transactions',
  maxHeight = 400,
  itemHeight = 60,
  className = '',
}) => {
  const [sortField, setSortField] = useState<keyof Transaction>('timestamp');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Funkcija za razvrščanje podatkov
  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => {
      if (sortField === 'timestamp') {
        return sortDirection === 'asc'
          ? new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
          : new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
      } else if (sortField === 'profitLoss') {
        return sortDirection === 'asc'
          ? a.profitLoss - b.profitLoss
          : b.profitLoss - a.profitLoss;
      } else {
        const aValue = a[sortField];
        const bValue = b[sortField];
        if (typeof aValue === 'string' && typeof bValue === 'string') {
          return sortDirection === 'asc'
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue);
        }
        return 0;
      }
    });
  }, [data, sortField, sortDirection]);

  // Funkcija za spremembo polja razvrščanja
  const handleSort = useCallback((field: keyof Transaction) => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  }, [sortField, sortDirection]);

  // Funkcija za prikaz ikone stanja
  const renderStatusIcon = useCallback((status: 'success' | 'pending' | 'failed') => {
    switch (status) {
      case 'success':
        return <CheckCircle size={16} className="text-success-500" />;
      case 'pending':
        return <Clock size={16} className="text-warning-500" />;
      case 'failed':
        return <AlertCircle size={16} className="text-error-500" />;
    }
  }, []);

  // Funkcija za prikaz elementa
  const renderTransaction = useCallback((transaction: Transaction, _index: number, _style: React.CSSProperties) => (
    <div className="flex items-center px-4 py-3 w-full">
      <div className="flex-1 min-w-0">
        <div className="flex items-center">
          {renderStatusIcon(transaction.status)}
          <span className="ml-2 font-medium text-gray-900 dark:text-white truncate">
            {transaction.strategy}
          </span>
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          {new Date(transaction.timestamp).toLocaleString()}
        </div>
      </div>
      <div className="flex-1 min-w-0 hidden md:block">
        <span className="text-sm text-gray-700 dark:text-gray-300">
          {transaction.network}
        </span>
      </div>
      <div className="flex-1 min-w-0 text-right">
        <span
          className={`text-sm font-medium ${
            transaction.profitLoss >= 0
              ? 'text-success-500'
              : 'text-error-500'
          }`}
        >
          {transaction.profitLoss >= 0 ? '+' : ''}
          {transaction.profitLoss.toFixed(4)} ETH
        </span>
      </div>
    </div>
  ), [renderStatusIcon]);

  return (
    <Card title={title || "Recent Transactions"} className={className}>
      <div className="flex items-center px-4 py-3 border-b border-gray-200 dark:border-dark-border text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider bg-primary-50/50 dark:bg-dark-background sticky top-0 z-10">
        <div
          className="flex-1 min-w-0 cursor-pointer hover:text-primary-600 dark:hover:text-primary-400 transition-colors duration-200"
          onClick={() => handleSort('strategy')}
        >
          Strategy
          {sortField === 'strategy' && (
            <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
          )}
        </div>
        <div
          className="flex-1 min-w-0 hidden md:block cursor-pointer hover:text-primary-600 dark:hover:text-primary-400 transition-colors duration-200"
          onClick={() => handleSort('network')}
        >
          Network
          {sortField === 'network' && (
            <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
          )}
        </div>
        <div
          className="flex-1 min-w-0 text-right cursor-pointer hover:text-primary-600 dark:hover:text-primary-400 transition-colors duration-200"
          onClick={() => handleSort('profitLoss')}
        >
          Profit/Loss
          {sortField === 'profitLoss' && (
            <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
          )}
        </div>
      </div>

      {sortedData.length === 0 ? (
        <div className="flex justify-center items-center h-32 text-gray-500 dark:text-gray-400">
          No transactions found
        </div>
      ) : (
        <VirtualizedList
          data={sortedData}
          height={maxHeight || 350}
          itemHeight={itemHeight}
          renderItem={renderTransaction}
          className="overflow-y-auto"
        />
      )}
    </Card>
  );
};

// Memorizirana verzija komponente za boljšo učinkovitost
export default memo(VirtualizedTransactionsTable);
