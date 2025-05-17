import React, { useState, useEffect, lazy, Suspense, useMemo } from 'react';
import Layout from '../layout/Layout';
import SystemHealthPanel from './SystemHealthPanel';
import TimeRangeSelector from '../ui/TimeRangeSelector';
import {
  keyMetrics,
  rpcLocations
} from '../../mockData';
import { fetchActivityData } from '../../utils/mockData';
import { TimeRange } from '../../types';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { useGet } from '../../hooks/useApi';

// Nove UI komponente
import StatCard from '../ui/StatCard';
import ChartCard from '../ui/ChartCard';
import Skeleton from '../ui/Skeleton';
import Select from '../ui/Select';
import { BarChart, Activity, Globe, Zap } from 'lucide-react';

// Lazy loading za zahtevne komponente
const ActivityChart = lazy(() => import('./ActivityChart'));
const TopStrategiesChart = lazy(() => import('./TopStrategiesChart'));
const VirtualizedTransactionsTable = lazy(() => import('./VirtualizedTransactionsTable'));
const ChainStatusPanel = lazy(() => import('./ChainStatusPanel'));
const RpcWorldMap = lazy(() => import('./RpcWorldMap'));

// Komponenta za prikaz nalaganja
const LoadingPlaceholder = () => (
  <div className="w-full h-64 flex items-center justify-center bg-primary-50/50 dark:bg-dark-background rounded-lg">
    <div className="flex flex-col items-center space-y-2">
      <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-500"></div>
      <div className="text-primary-500 dark:text-primary-400 text-sm font-medium">Nalaganje podatkov...</div>
    </div>
  </div>
);

interface OverviewDashboardProps {
  timeRange: TimeRange;
  onTimeRangeChange: (range: TimeRange) => void;
}

const OverviewDashboard: React.FC<OverviewDashboardProps> = ({
  timeRange,
  onTimeRangeChange
}) => {
  const [activityData, setActivityData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Pridobi podatke iz WebSocket konteksta
  const {
    systemStatus,
    chainStatus,
    transactions = [],
    strategies = [],
    rpcStatus
  } = useWebSocket();

  // Pridobi podatke iz API-ja
  const { data: systemHealthData } = useGet<SystemHealthData>(
    'system/health',
    undefined,
    { skip: !!systemStatus }
  );

  const { data: chainStatusData } = useGet<ChainStatusData[]>(
    'blockchain/status',
    undefined,
    { skip: !!chainStatus }
  );

  const { data: strategiesData } = useGet<StrategyData[]>(
    'strategies',
    undefined,
    { skip: strategies.length > 0 }
  );

  const { data: metricsData } = useGet<MetricData[]>(
    'metrics/key',
    undefined,
    { skip: false }
  );

  const { data: rpcLocationsData } = useGet<RpcLocationData[]>(
    'rpc/locations',
    undefined,
    { skip: !!rpcStatus }
  );

  const { data: recentTransactionsData } = useGet<TransactionData[]>(
    'transactions/recent',
    undefined,
    { skip: transactions.length > 0 }
  );

  // Definirajmo tipe za podatke
  interface SystemHealthData {
    components: Array<{
      name: string;
      status: string;
      details?: Record<string, unknown>;
    }>;
  }

  interface ChainStatusData {
    id: string;
    name: string;
    status: string;
    latency: number;
    blockHeight: number;
    lastUpdate: string;
  }

  interface StrategyData {
    id: string;
    name: string;
    type: string;
    status: string;
    profit: number;
    transactions: number;
    successRate: number;
  }

  interface MetricData {
    title: string;
    value: string | number;
    change: number;
    changeType: 'positive' | 'negative' | 'neutral';
    icon: string;
  }

  interface RpcLocationData {
    id: string;
    name: string;
    location: [number, number];
    status: string;
    latency: number;
  }

  interface TransactionData {
    id: string;
    hash: string;
    type: string;
    status: string;
    amount: number;
    profit: number;
    timestamp: string;
    chain: string;
  }

  // Združi podatke iz WebSocket in API
  const combinedSystemStatus = useMemo(() => {
    return systemStatus || systemHealthData || [];
  }, [systemStatus, systemHealthData]);

  const combinedChainStatus = useMemo(() => {
    return chainStatus || chainStatusData || [];
  }, [chainStatus, chainStatusData]);

  const combinedStrategies = useMemo(() => {
    return strategies.length > 0 ? strategies : (strategiesData || []);
  }, [strategies, strategiesData]);

  const combinedRpcStatus = useMemo(() => {
    return rpcStatus || rpcLocationsData || rpcLocations;
  }, [rpcStatus, rpcLocationsData]);

  const combinedTransactions = useMemo(() => {
    return transactions.length > 0 ? transactions : (recentTransactionsData || []);
  }, [transactions, recentTransactionsData]);

  // Memorizirani podatki za metrike, da se ne renderirajo ob vsaki spremembi
  const metricCards = useMemo(() => {
    const metrics = metricsData || keyMetrics;
    return metrics.map((metric, index) => {
      // Določimo ikono glede na tip metrike
      let icon;
      switch (metric.icon) {
        case 'activity':
          icon = <Activity size={20} className="text-white" />;
          break;
        case 'bar-chart':
          icon = <BarChart size={20} className="text-white" />;
          break;
        case 'zap':
          icon = <Zap size={20} className="text-white" />;
          break;
        case 'globe':
          icon = <Globe size={20} className="text-white" />;
          break;
        default:
          icon = <Activity size={20} className="text-white" />;
      }

      return (
        <StatCard
          key={index}
          title={metric.title}
          value={metric.value}
          change={metric.change}
          icon={icon}
          iconClassName={`bg-primary-500 dark:bg-primary-600`}
        />
      );
    });
  }, [metricsData]);

  // Optimizirana funkcija za pridobivanje podatkov
  const fetchData = useMemo(() => {
    return async () => {
      try {
        // Pridobi samo podatke o aktivnosti, ostalo pride preko API ali WebSocket
        const activityResult = await fetchActivityData(timeRange);
        setActivityData(activityResult);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setLoading(false);
      }
    };
  }, [timeRange]);

  // Učinek za nalaganje podatkov
  useEffect(() => {
    setLoading(true);
    fetchData();

    // Interval za osveževanje podatkov - zmanjšamo frekvenco na 60 sekund
    const interval = setInterval(() => {
      fetchData();
    }, 60000);

    return () => clearInterval(interval);
  }, [fetchData]);

  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Dashboard Overview
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
          <TimeRangeSelector
            value={timeRange}
            onChange={onTimeRangeChange}
          />
        </div>
      </div>

      {loading && !activityData.length ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          {[...Array(4)].map((_, index) => (
            <Skeleton key={index} variant="card" height={160} animation="pulse" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          {metricCards}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2">
          <Suspense fallback={<LoadingPlaceholder />}>
            <ChartCard
              title="Activity Overview"
              subtitle="Transactions and profit over time"
              chart={<ActivityChart data={activityData} timeRange={timeRange} />}
              toolbar={
                <div className="flex items-center space-x-2">
                  <Select
                    options={[
                      { value: 'all', label: 'All Activities' },
                      { value: 'transactions', label: 'Transactions' },
                      { value: 'profit', label: 'Profit' }
                    ]}
                    value="all"
                    placeholder="Filter"
                  />
                </div>
              }
            />
          </Suspense>
        </div>
        <div className="lg:col-span-1">
          <SystemHealthPanel data={combinedSystemStatus?.components || combinedSystemStatus || []} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div>
          <Suspense fallback={<LoadingPlaceholder />}>
            <ChartCard
              title="RPC Network Status"
              subtitle="Global RPC node distribution and status"
              chart={<RpcWorldMap data={combinedRpcStatus} />}
            />
          </Suspense>
        </div>
        <div>
          <Suspense fallback={<LoadingPlaceholder />}>
            <ChartCard
              title="Top Performing Strategies"
              subtitle="Strategies ranked by profit"
              chart={<TopStrategiesChart data={combinedStrategies} />}
            />
          </Suspense>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2">
          <Suspense fallback={<LoadingPlaceholder />}>
            <ChartCard
              title="Recent Transactions"
              subtitle="Latest transactions across all networks"
              chart={
                <VirtualizedTransactionsTable
                  data={combinedTransactions}
                  maxHeight={350}
                />
              }
            />
          </Suspense>
        </div>
        <div className="lg:col-span-1">
          <ChartCard
            title="Chain Status"
            subtitle="Current status of supported blockchains"
            chart={
              <Suspense fallback={<LoadingPlaceholder />}>
                <ChainStatusPanel data={combinedChainStatus} />
              </Suspense>
            }
          />
        </div>
      </div>
    </Layout>
  );
};

export default OverviewDashboard;