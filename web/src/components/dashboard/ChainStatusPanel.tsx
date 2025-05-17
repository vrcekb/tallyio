import React from 'react';
import Card from '../ui/Card';
import { ChainStatus } from '../../types';
import { TrendingUp, TrendingDown, Minus, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

interface ChainStatusPanelProps {
  data: ChainStatus[];
}

export const ChainStatusPanel: React.FC<ChainStatusPanelProps> = ({ data }) => {
  const getStatusIcon = (status: 'online' | 'degraded' | 'offline') => {
    switch (status) {
      case 'online':
        return <CheckCircle size={16} className="text-success-500" />;
      case 'degraded':
        return <AlertTriangle size={16} className="text-warning-500" />;
      case 'offline':
        return <XCircle size={16} className="text-error-500" />;
    }
  };

  const getGasTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up':
        return <TrendingUp size={14} className="text-error-500" />;
      case 'down':
        return <TrendingDown size={14} className="text-success-500" />;
      case 'stable':
        return <Minus size={14} className="text-primary-500 dark:text-primary-400" />;
    }
  };

  const formatGasPrice = (network: string, price: number): string => {
    if (network === 'Ethereum') return `${price} gwei`;
    return `${price} gwei`;
  };

  return (
    <Card title="Chain Status">
      <div className="grid grid-cols-1 gap-3 max-h-[350px] overflow-auto pr-1">
        {data.map((chain) => (
          <div
            key={chain.id}
            className="p-3 rounded bg-primary-50/50 dark:bg-dark-background flex flex-col sm:flex-row sm:items-center justify-between gap-2 shadow-sm hover:shadow-md transition-all duration-300 hover:translate-y-[-2px]"
          >
            <div className="flex items-center">
              {getStatusIcon(chain.status)}
              <span className="ml-2 font-medium text-primary-900 dark:text-primary-100">
                {chain.name}
              </span>
              <span
                className={`ml-2 px-1.5 py-0.5 text-xs rounded ${
                  chain.status === 'online'
                    ? 'bg-success-100/80 text-success-700 dark:bg-success-900/30 dark:text-success-400'
                    : chain.status === 'degraded'
                      ? 'bg-warning-100/80 text-warning-700 dark:bg-warning-900/30 dark:text-warning-400'
                      : 'bg-error-100/80 text-error-700 dark:bg-error-900/30 dark:text-error-400'
                }`}
              >
                {chain.status}
              </span>
            </div>

            <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
              <div className="flex items-center">
                <span className="text-primary-500 dark:text-primary-400 mr-1">Gas:</span>
                <span className="font-medium text-primary-900 dark:text-primary-100 mr-1">
                  {formatGasPrice(chain.name, chain.gasPrice)}
                </span>
                {getGasTrendIcon(chain.gasTrend)}
              </div>

              <div>
                <span className="text-primary-500 dark:text-primary-400 mr-1">Conn:</span>
                <span className="font-medium text-primary-900 dark:text-primary-100">
                  {chain.connections}
                </span>
              </div>

              <div>
                <span className="text-primary-500 dark:text-primary-400 mr-1">Mempool:</span>
                <span className="font-medium text-primary-900 dark:text-primary-100">
                  {chain.mempoolSize.toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

export default ChainStatusPanel;