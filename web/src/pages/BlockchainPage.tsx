import React, { lazy, Suspense, useMemo, useState } from 'react';
import { Card } from '../components/ui/Card';
import { ChainStatusPanel } from '../components/dashboard/ChainStatusPanel';
import SystemHealthPanel from '../components/dashboard/SystemHealthPanel';
import Layout from '../components/layout/Layout';
import { chainStatus, systemHealthStatus, generateRecentTransactions, rpcLocations } from '../mockData';

// Nove UI komponente
import ChartCard from '../components/ui/ChartCard';
import Select from '../components/ui/Select';
import Tabs from '../components/ui/Tabs';
import NetworkList from '../components/ui/NetworkList';
import TransactionList from '../components/ui/TransactionList';
import Skeleton from '../components/ui/Skeleton';
import { Globe, Activity, Database } from 'lucide-react';

// Lazy loading za zahtevne komponente
const RpcWorldMap = lazy(() => import('../components/dashboard/RpcWorldMap'));
const TransactionsTable = lazy(() => import('../components/dashboard/TransactionsTable'));

// Komponenta za prikaz nalaganja
const LoadingPlaceholder = () => (
  <div className="w-full h-64 flex items-center justify-center bg-primary-50/50 dark:bg-dark-background rounded-lg">
    <div className="flex flex-col items-center space-y-2">
      <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-500"></div>
      <div className="text-primary-500 dark:text-primary-400 text-sm font-medium">Nalaganje podatkov...</div>
    </div>
  </div>
);

const BlockchainPage = () => {
  // Memoriziramo podatke za transakcije, da se ne generirajo ob vsakem renderiranju
  const transactionData = useMemo(() => generateRecentTransactions(20), []);
  const [activeTab, setActiveTab] = useState('overview');

  // Pretvori podatke o verigi v format za NetworkList komponento
  const networkData = useMemo(() => {
    return chainStatus.map((chain, index) => ({
      id: index,
      name: chain.name,
      status: chain.status,
      icon: <Globe size={16} />,
      gasPrice: chain.gasPrice,
      gasTrend: chain.gasTrend === 'up' ? 5 : chain.gasTrend === 'down' ? -3 : 0,
      connections: chain.connections,
      latency: Math.floor(Math.random() * 500), // Simulirani podatki
      mempoolSize: chain.mempoolSize
    }));
  }, []);

  // Pretvori podatke o transakcijah v format za TransactionList komponento
  const formattedTransactions = useMemo(() => {
    return transactionData.map((tx, index) => ({
      id: index,
      hash: tx.hash || `0x${Math.random().toString(16).substring(2, 10)}...`,
      status: tx.status || (Math.random() > 0.2 ? 'success' : Math.random() > 0.5 ? 'pending' : 'failed'),
      timestamp: tx.timestamp || new Date(Date.now() - Math.random() * 86400000),
      network: tx.network || 'Ethereum',
      from: tx.from || `0x${Math.random().toString(16).substring(2, 10)}...`,
      to: tx.to || `0x${Math.random().toString(16).substring(2, 10)}...`,
      value: tx.value || `${(Math.random() * 10).toFixed(4)} ETH`,
      gas: tx.gas || `${(Math.random() * 0.05).toFixed(6)} ETH`,
      explorerUrl: `https://etherscan.io/tx/${tx.hash}`
    }));
  }, [transactionData]);

  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Blockchain Explorer
        </h1>
        <div className="flex items-center space-x-4">
          <Select
            options={[
              { value: 'all', label: 'All Networks' },
              { value: 'ethereum', label: 'Ethereum' },
              { value: 'arbitrum', label: 'Arbitrum' },
              { value: 'optimism', label: 'Optimism' },
              { value: 'polygon', label: 'Polygon' }
            ]}
            value="all"
            placeholder="Select Network"
          />
        </div>
      </div>

      <Tabs
        tabs={[
          {
            id: 'overview',
            label: 'Overview',
            icon: <Activity size={16} />,
            content: (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <ChartCard
                    title="Chain Status"
                    subtitle="Current status of supported blockchains"
                    chart={<ChainStatusPanel data={chainStatus} />}
                  />
                  <ChartCard
                    title="System Health"
                    subtitle="Health status of blockchain components"
                    chart={<SystemHealthPanel data={systemHealthStatus} />}
                  />
                </div>

                <div className="mb-6">
                  <ChartCard
                    title="RPC Network"
                    subtitle="Global RPC node distribution and status"
                    chart={
                      <Suspense fallback={<LoadingPlaceholder />}>
                        <RpcWorldMap data={rpcLocations} />
                      </Suspense>
                    }
                  />
                </div>

                <div className="mb-6">
                  <TransactionList
                    transactions={formattedTransactions.slice(0, 5)}
                    title="Recent Transactions"
                    onTransactionClick={(tx) => console.log('Transaction clicked:', tx)}
                  />
                </div>
              </>
            )
          },
          {
            id: 'networks',
            label: 'Networks',
            icon: <Globe size={16} />,
            content: (
              <div className="mb-6">
                <NetworkList
                  networks={networkData}
                  title="Supported Networks"
                  onNetworkClick={(network) => console.log('Network clicked:', network)}
                />
              </div>
            )
          },
          {
            id: 'transactions',
            label: 'Transactions',
            icon: <Database size={16} />,
            content: (
              <div className="mb-6">
                <TransactionList
                  transactions={formattedTransactions}
                  title="All Transactions"
                  onTransactionClick={(tx) => console.log('Transaction clicked:', tx)}
                />
              </div>
            )
          }
        ]}
        defaultTab="overview"
        onChange={setActiveTab}
        variant="enclosed"
        color="primary"
      />
    </Layout>
  );
};

export default BlockchainPage;