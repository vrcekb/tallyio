import React, { useState } from 'react';
import { Plus, Edit2, Trash2 } from 'lucide-react';
import Layout from '../components/layout/Layout';
import Card from '../components/ui/Card';

interface ProtocolData {
  id: string;
  name: string;
  tvl: number;
  totalBorrowed: number;
  totalSupplied: number;
  utilizationRate: number;
  chain: string;
  apy: number;
}

// Mock data - In production, this would come from an API
const mockProtocolData: ProtocolData[] = [
  {
    id: '1',
    name: 'Aave V3',
    tvl: 6234567890,
    totalBorrowed: 3234567890,
    totalSupplied: 5234567890,
    utilizationRate: 61.8,
    chain: 'Ethereum',
    apy: 4.2
  },
  {
    id: '2',
    name: 'Compound',
    tvl: 4234567890,
    totalBorrowed: 2234567890,
    totalSupplied: 3234567890,
    utilizationRate: 69.2,
    chain: 'Ethereum',
    apy: 3.8
  },
  // Add more mock data as needed
];

const ProtocolManagementPage: React.FC = () => {
  // Uporabljeno v onClick handleru
  const setShowAddModal = useState(false)[1];
  // Uporabljeno v onClick handleru
  const setSelectedProtocol = useState<ProtocolData | null>(null)[1];

  const formatNumber = (num: number): string => {
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
    return `$${num.toFixed(2)}`;
  };

  return (
    <Layout>
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Protocol Management</h1>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600"
        >
          <Plus size={20} className="mr-2" />
          Add New Protocol
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <div className="flex flex-col">
            <span className="text-sm text-gray-500 dark:text-gray-400">Total TVL</span>
            <div className="flex items-baseline mt-2">
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                $10.47B
              </span>
              <span className="ml-2 text-sm text-success-500">+5.2%</span>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col">
            <span className="text-sm text-gray-500 dark:text-gray-400">Total Borrowed</span>
            <div className="flex items-baseline mt-2">
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                $5.47B
              </span>
              <span className="ml-2 text-sm text-error-500">-2.5%</span>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col">
            <span className="text-sm text-gray-500 dark:text-gray-400">Total Supplied</span>
            <div className="flex items-baseline mt-2">
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                $8.47B
              </span>
              <span className="ml-2 text-sm text-success-500">+3.8%</span>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col">
            <span className="text-sm text-gray-500 dark:text-gray-400">Active Protocols</span>
            <div className="flex items-baseline mt-2">
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                8
              </span>
              <span className="ml-2 text-sm text-success-500">+1</span>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Protocol Overview">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-dark-border">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Protocol Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Chain
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  TVL
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Total Borrowed
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Utilization Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  APY
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-dark-card divide-y divide-gray-200 dark:divide-dark-border">
              {mockProtocolData.map((protocol) => (
                <tr key={protocol.id} className="hover:bg-gray-50 dark:hover:bg-dark-background">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="font-medium text-gray-900 dark:text-white">{protocol.name}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-gray-900 dark:text-white">{protocol.chain}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-gray-900 dark:text-white">{formatNumber(protocol.tvl)}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-gray-900 dark:text-white">{formatNumber(protocol.totalBorrowed)}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-200 dark:bg-dark-background rounded-full h-2 mr-2">
                        <div
                          className="bg-primary-500 h-2 rounded-full"
                          style={{ width: `${protocol.utilizationRate}%` }}
                        />
                      </div>
                      <span className="text-gray-900 dark:text-white">{protocol.utilizationRate}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-success-500">{protocol.apy}%</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <button
                      onClick={() => setSelectedProtocol(protocol)}
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

export default ProtocolManagementPage;