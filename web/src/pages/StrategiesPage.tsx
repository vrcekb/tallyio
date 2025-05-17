import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/Card';
import { LineChart, Activity, TrendingUp, RefreshCw } from 'lucide-react';
import Layout from '../components/layout/Layout';
import StrategyAnalysis from '../components/strategies/StrategyAnalysis';
import { strategies } from '../mockData';

// Definiramo tip za strategijo
interface Strategy {
  id: string;
  name: string;
  type: string;
  status: string;
  profit: number;
  transactions: number;
  successRate: number;
}

const StrategiesPage = () => {
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Učinek za nastavitev privzete strategije
  useEffect(() => {
    if (strategies.length > 0 && !selectedStrategy) {
      setSelectedStrategy(strategies[0]);
    }
  }, [selectedStrategy]);

  // Funkcija za izbiro strategije
  const handleSelectStrategy = (strategy: Strategy) => {
    setIsLoading(true);
    // Simulacija nalaganja podatkov
    setTimeout(() => {
      setSelectedStrategy(strategy);
      setIsLoading(false);
    }, 300);
  };

  // Funkcija za osvežitev podatkov
  const handleRefresh = () => {
    if (selectedStrategy) {
      setIsLoading(true);
      // Simulacija nalaganja podatkov
      setTimeout(() => {
        setIsLoading(false);
      }, 500);
    }
  };

  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Trading Strategies</h1>
        <button
          onClick={handleRefresh}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-background"
          disabled={isLoading}
        >
          <RefreshCw size={20} className={`text-gray-700 dark:text-gray-300 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <div className="h-[150px]">
          <Card className="p-6 cursor-pointer hover:shadow-md transition-shadow duration-200 h-full" onClick={() => {}}>
            <div className="flex flex-col h-full">
              <div className="flex items-center gap-4 mb-4">
                <LineChart className="w-6 h-6 text-primary-500" />
                <h2 className="text-xl font-semibold">Performance Metrics</h2>
              </div>
              <p className="text-primary-600 dark:text-primary-400">View detailed performance analytics for all active trading strategies.</p>
            </div>
          </Card>
        </div>

        <div className="h-[150px]">
          <Card className="p-6 cursor-pointer hover:shadow-md transition-shadow duration-200 h-full" onClick={() => {}}>
            <div className="flex flex-col h-full">
              <div className="flex items-center gap-4 mb-4">
                <Activity className="w-6 h-6 text-success-500" />
                <h2 className="text-xl font-semibold">Active Strategies</h2>
              </div>
              <p className="text-primary-600 dark:text-primary-400">Monitor and manage currently running automated trading strategies.</p>
            </div>
          </Card>
        </div>

        <div className="h-[150px]">
          <Card className="p-6 cursor-pointer hover:shadow-md transition-shadow duration-200 h-full" onClick={() => {}}>
            <div className="flex flex-col h-full">
              <div className="flex items-center gap-4 mb-4">
                <TrendingUp className="w-6 h-6 text-accent-500" />
                <h2 className="text-xl font-semibold">Strategy Builder</h2>
              </div>
              <p className="text-primary-600 dark:text-primary-400">Create and customize new trading strategies using our visual builder.</p>
            </div>
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1">
          <div className="h-[500px]">
            <Card className="p-4 h-full">
              <div className="flex flex-col h-full">
                <h2 className="text-lg font-semibold mb-4">Available Strategies</h2>
                <div className="space-y-2 flex-1 overflow-auto">
                  {strategies.map((strategy) => (
                    <div
                      key={strategy.id}
                      className={`p-3 rounded-lg cursor-pointer transition-colors duration-200 ${
                        selectedStrategy?.id === strategy.id
                          ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                          : 'hover:bg-gray-100 dark:hover:bg-dark-background'
                      }`}
                      onClick={() => handleSelectStrategy(strategy)}
                    >
                      <div className="font-medium">{strategy.name}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {strategy.type}
                      </div>
                      <div className={`text-sm font-medium mt-1 ${
                        strategy.profit >= 0 ? 'text-success-500' : 'text-error-500'
                      }`}>
                        {strategy.profit >= 0 ? '+' : ''}
                        {strategy.profit.toFixed(4)} ETH
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          </div>
        </div>

        <div className="lg:col-span-3">
          <div className="h-[500px]">
            {selectedStrategy ? (
              <StrategyAnalysis
                strategy={{
                  ...selectedStrategy,
                  transactions: Array.from({ length: 20 }, (_, i) => ({
                    id: `tx-${i}`,
                    timestamp: new Date(Date.now() - i * 86400000).toISOString(),
                    status: Math.random() > 0.2 ? 'success' : 'failed',
                    profitLoss: (Math.random() * 0.2 - 0.05) * (selectedStrategy.profit > 0 ? 1 : -1),
                    executionTime: Math.random() * 1000
                  }))
                }}
              />
            ) : (
              <Card className="p-6 flex justify-center items-center h-full">
                <p className="text-gray-500 dark:text-gray-400">Select a strategy to view analysis</p>
              </Card>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default StrategiesPage;