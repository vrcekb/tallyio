import React, { useState } from 'react';
import { Plus, Edit2, Trash2, TrendingUp, ArrowUpRight } from 'lucide-react';
import Layout from '../components/layout/Layout';
import Card from '../components/ui/Card';

interface DexData {
  id: string;
  name: string;
  tvl: number;
  volume24h: number;
  transactions24h: number;
  change24h: number;
  chain: string;
}

// Mock data - In production, this would come from an API
const mockDexData: DexData[] = [
  {
    id: '1',
    name: 'Uniswap V3',
    tvl: 5234567890,
    volume24h: 423456789,
    transactions24h: 45678,
    change24h: 12.5,
    chain: 'Ethereum'
  },
  {
    id: '2',
    name: 'PancakeSwap',
    tvl: 3234567890,
    volume24h: 323456789,
    transactions24h: 34567,
    change24h: -5.2,
    chain: 'BSC'
  },
  // Add more mock data as needed
];

const DexManagementPage: React.FC = () => {
  // Uporabljeno v onClick handleru
  const setShowAddModal = useState(false)[1];
  // Uporabljeno v onClick handleru
  const setSelectedDex = useState<DexData | null>(null)[1];

  const formatNumber = (num: number): string => {
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
    return `$${num.toFixed(2)}`;
  };

  return (
    <Layout>
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">DEX Management</h1>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600"
        >
          <Plus size={20} className="mr-2" />
          Add New DEX
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="h-[120px]">
          <Card>
            <div className="flex flex-col h-full justify-center">
              <span className="text-sm text-gray-500 dark:text-gray-400">Total TVL</span>
              <div className="flex items-baseline mt-2">
                <span className="text-2xl font-bold text-gray-900 dark:text-white">
                  $8.47B
                </span>
                <span className="ml-2 text-sm text-success-500">+7.2%</span>
              </div>
            </div>
          </Card>
        </div>
        <div className="h-[120px]">
          <Card>
            <div className="flex flex-col h-full justify-center">
              <span className="text-sm text-gray-500 dark:text-gray-400">24h Volume</span>
              <div className="flex items-baseline mt-2">
                <span className="text-2xl font-bold text-gray-900 dark:text-white">
                  $746.9M
                </span>
                <span className="ml-2 text-sm text-success-500">+12.5%</span>
              </div>
            </div>
          </Card>
        </div>
        <div className="h-[120px]">
          <Card>
            <div className="flex flex-col h-full justify-center">
              <span className="text-sm text-gray-500 dark:text-gray-400">Active DEXes</span>
              <div className="flex items-baseline mt-2">
                <span className="text-2xl font-bold text-gray-900 dark:text-white">
                  12
                </span>
                <span className="ml-2 text-sm text-success-500">+1</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <Card title="DEX Overview">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-dark-border">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  DEX Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Chain
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  TVL
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  24h Volume
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  24h Change
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-dark-card divide-y divide-gray-200 dark:divide-dark-border">
              {mockDexData.map((dex) => (
                <tr key={dex.id} className="hover:bg-gray-50 dark:hover:bg-dark-background">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="font-medium text-gray-900 dark:text-white">{dex.name}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-gray-900 dark:text-white">{dex.chain}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-gray-900 dark:text-white">{formatNumber(dex.tvl)}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-gray-900 dark:text-white">{formatNumber(dex.volume24h)}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`flex items-center ${
                      dex.change24h >= 0 ? 'text-success-500' : 'text-error-500'
                    }`}>
                      {dex.change24h >= 0 ? <TrendingUp size={16} className="mr-1" /> : <ArrowUpRight size={16} className="mr-1 transform rotate-90" />}
                      {dex.change24h > 0 ? '+' : ''}{dex.change24h}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <button
                      onClick={() => setSelectedDex(dex)}
                      className="text-primary-600 hover:text-primary-900 dark:text-primary-400 dark:hover:text-primary-300 mr-3"
                    >
                      <Edit2 size={16} />
                    </button>
                    <button className="text-error-600 hover:text-error-900 dark:text-error-400 dark:hover:text-error-300">
                      <Trash2 size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </Layout>
  );
};

export default DexManagementPage;