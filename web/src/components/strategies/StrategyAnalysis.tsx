import React, { useState, useEffect, useCallback, memo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import Card from '../ui/Card';
import useWorker, { MessageType } from '../../hooks/useWorker';
import {
  AnalysisStrategy as Strategy,
  StrategyAnalysisResult as AnalysisData
} from '../../types/strategy';

interface StrategyAnalysisProps {
  strategy: Strategy;
  className?: string;
}

/**
 * Komponenta za prikaz analize strategije
 * Uporablja WebWorker za zahtevne izračune
 */
const StrategyAnalysis: React.FC<StrategyAnalysisProps> = ({ strategy, className = '' }) => {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);

  // Uporabi WebWorker za analizo strategije
  const { loading, error, execute } = useWorker<AnalysisData, Strategy>(
    new URL('../../workers/dataProcessor.worker.ts', import.meta.url).href,
    MessageType.ANALYZE_STRATEGY
  );

  // Učinek za analizo strategije
  useEffect(() => {
    if (strategy) {
      execute(strategy)
        .then(result => {
          setAnalysisData(result);
        })
        .catch(err => {
          console.error('Error analyzing strategy:', err);
        });
    }
  }, [strategy, execute]);

  // Funkcija za formatiranje datuma
  const formatDate = useCallback((timestamp: string) => {
    const date = new Date(timestamp);
    return `${date.getDate()}.${date.getMonth() + 1}.${date.getFullYear()}`;
  }, []);

  // Funkcija za formatiranje vrednosti
  const formatValue = useCallback((value: number) => {
    return value.toFixed(4);
  }, []);

  return (
    <Card
      title={`Strategy Analysis: ${strategy?.name || 'Unknown'}`}
      className={className}
    >
      {loading && (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
        </div>
      )}

      {error && (
        <div className="flex justify-center items-center h-64 text-error-500">
          Error: {error}
        </div>
      )}

      {!loading && !error && analysisData && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-dark-card p-4 rounded-lg shadow-sm">
              <div className="text-sm text-gray-500 dark:text-gray-400">Profit/Loss</div>
              <div className={`text-xl font-semibold ${
                analysisData.profitLoss >= 0 ? 'text-success-500' : 'text-error-500'
              }`}>
                {analysisData.profitLoss >= 0 ? '+' : ''}
                {formatValue(analysisData.profitLoss)} ETH
              </div>
            </div>

            <div className="bg-white dark:bg-dark-card p-4 rounded-lg shadow-sm">
              <div className="text-sm text-gray-500 dark:text-gray-400">Success Rate</div>
              <div className="text-xl font-semibold text-primary-500">
                {analysisData.successRate.toFixed(2)}%
              </div>
            </div>

            <div className="bg-white dark:bg-dark-card p-4 rounded-lg shadow-sm">
              <div className="text-sm text-gray-500 dark:text-gray-400">Transactions</div>
              <div className="text-xl font-semibold text-gray-700 dark:text-gray-300">
                {analysisData.transactionCount}
              </div>
            </div>
          </div>

          <div className="h-64">
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Profit/Loss Trend
            </div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={analysisData.timeSeriesData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatDate}
                  stroke="#6B7280"
                />
                <YAxis
                  tickFormatter={formatValue}
                  stroke="#6B7280"
                />
                <Tooltip
                  formatter={(value: number) => [formatValue(value), 'Profit/Loss']}
                  labelFormatter={(label: string) => formatDate(label)}
                />
                <Line
                  type="monotone"
                  dataKey="profitLoss"
                  stroke="#0EA5E9"
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="h-64">
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Moving Average (5 transactions)
            </div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={analysisData.movingAverage}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatDate}
                  stroke="#6B7280"
                />
                <YAxis
                  tickFormatter={formatValue}
                  stroke="#6B7280"
                />
                <Tooltip
                  formatter={(value: number) => [formatValue(value), 'Moving Avg']}
                  labelFormatter={(label: string) => formatDate(label)}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#10B981"
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </Card>
  );
};

// Memorizirana verzija komponente za boljšo učinkovitost
export default memo(StrategyAnalysis);
