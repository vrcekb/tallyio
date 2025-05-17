import React from 'react';
import { format } from 'date-fns';
import Card from '../ui/Card';
import { Transaction } from '../../types';
import { Check, Clock, AlertTriangle, ArrowRight } from 'lucide-react';
import { getProfitLossColor } from '../../utils/mockData';

interface TransactionsTableProps {
  data: Transaction[];
}

const TransactionsTable: React.FC<TransactionsTableProps> = ({ data }) => {
  const getStatusIcon = (status: 'success' | 'pending' | 'failed') => {
    switch (status) {
      case 'success':
        return <Check size={16} className="text-success-500" />;
      case 'pending':
        return <Clock size={16} className="text-warning-500" />;
      case 'failed':
        return <AlertTriangle size={16} className="text-error-500" />;
    }
  };

  return (
    <Card title="Recent Transactions">
      <div className="overflow-x-auto -mx-4">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-dark-border">
          <thead>
            <tr>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Time</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Strategy</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Network</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Profit/Loss</th>
              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"></th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-dark-card divide-y divide-gray-100 dark:divide-dark-border">
            {data.map((transaction) => (
              <tr key={transaction.id} className="hover:bg-gray-50 dark:hover:bg-dark-background">
                <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 whitespace-nowrap">
                  {format(new Date(transaction.timestamp), 'HH:mm:ss')}
                </td>
                <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 whitespace-nowrap">
                  {transaction.strategy}
                </td>
                <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 whitespace-nowrap">
                  {transaction.network}
                </td>
                <td className="px-4 py-3 text-sm whitespace-nowrap">
                  <div className="flex items-center">
                    {getStatusIcon(transaction.status)}
                    <span 
                      className={`ml-1.5 ${
                        transaction.status === 'success' 
                          ? 'text-success-600 dark:text-success-400' 
                          : transaction.status === 'pending' 
                            ? 'text-warning-600 dark:text-warning-400' 
                            : 'text-error-600 dark:text-error-400'
                      }`}
                    >
                      {transaction.status.charAt(0).toUpperCase() + transaction.status.slice(1)}
                    </span>
                  </div>
                </td>
                <td className={`px-4 py-3 text-sm font-medium whitespace-nowrap text-right ${getProfitLossColor(transaction.profitLoss)}`}>
                  {transaction.profitLoss > 0 ? '+' : ''}
                  {transaction.profitLoss.toFixed(4)} ETH
                </td>
                <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 whitespace-nowrap text-right">
                  <button className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-dark-background">
                    <ArrowRight size={16} className="text-gray-500 dark:text-gray-400" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex justify-center mt-4">
        <button className="px-4 py-2 text-sm font-medium text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-md transition-colors">
          View All Transactions
        </button>
      </div>
    </Card>
  );
};

export default TransactionsTable;