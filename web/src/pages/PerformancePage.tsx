import React, { useState, useMemo, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, CheckCircle, AlertTriangle, XCircle, Activity, Cpu, HardDrive, Network } from 'lucide-react';
import Layout from '../components/layout/Layout';
import { performanceMetrics, generateResourceUsage, serviceHealth } from '../mockData/performance';

// Nove UI komponente
import ChartCard from '../components/ui/ChartCard';
import StatCard from '../components/ui/StatCard';
import Select from '../components/ui/Select';
import Button from '../components/ui/Button';
import Tabs from '../components/ui/Tabs';
import ProgressBar from '../components/ui/ProgressBar';

const PerformancePage: React.FC = () => {
  // Memoizacija podatkov za izboljšanje zmogljivosti
  const resourceData = useMemo(() => generateResourceUsage(24), []);
  const [timeRange, setTimeRange] = useState('24h');
  const [activeTab, setActiveTab] = useState('overview');

  // Memoizacija funkcije za prikaz ikon statusa
  const getStatusIcon = useCallback((status: 'healthy' | 'degraded' | 'down') => {
    switch (status) {
      case 'healthy':
        return <CheckCircle size={16} className="text-success-500" />;
      case 'degraded':
        return <AlertTriangle size={16} className="text-warning-500" />;
      case 'down':
        return <XCircle size={16} className="text-error-500" />;
    }
  }, []);

  // Memoizacija funkcije za prikaz ikon metrik
  const getMetricIcon = useCallback((metricType: string) => {
    switch (metricType) {
      case 'cpu':
        return <Cpu size={20} className="text-white" />;
      case 'memory':
        return <HardDrive size={20} className="text-white" />;
      case 'network':
        return <Network size={20} className="text-white" />;
      default:
        return <Activity size={20} className="text-white" />;
    }
  }, []);

  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">System Performance</h1>
        <div className="flex items-center space-x-4">
          <Select
            options={[
              { value: '24h', label: 'Last 24 Hours' },
              { value: '7d', label: 'Last 7 Days' },
              { value: '30d', label: 'Last 30 Days' }
            ]}
            value={timeRange}
            onChange={(value) => setTimeRange(value)}
          />
          <Button
            color="primary"
            leftIcon={<Activity size={16} />}
          >
            Export Report
          </Button>
        </div>
      </div>

      <Tabs
        tabs={useMemo(() => [
          {
            id: 'overview',
            label: 'Overview',
            icon: <Activity size={16} />,
            content: (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                  {performanceMetrics.map((metric, index) => (
                    <StatCard
                      key={index}
                      title={metric.title}
                      value={`${metric.value}${metric.unit}`}
                      change={metric.change}
                      icon={getMetricIcon(metric.type || 'default')}
                      iconClassName={`bg-${metric.status === 'good' ? 'success' : metric.status === 'warning' ? 'warning' : 'error'}-500`}
                      footer={
                        <ProgressBar
                          value={metric.value}
                          max={metric.threshold}
                          color={metric.status === 'good' ? 'success' : metric.status === 'warning' ? 'warning' : 'error'}
                          size="sm"
                        />
                      }
                    />
                  ))}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <ChartCard
                    title="Resource Usage"
                    subtitle="CPU, Memory and Network utilization"
                    chart={
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={resourceData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="timestamp" />
                            <YAxis />
                            <Tooltip />
                            <Line type="monotone" dataKey="cpu" stroke="#3575E3" />
                            <Line type="monotone" dataKey="memory" stroke="#4FD1C5" />
                            <Line type="monotone" dataKey="network" stroke="#F59E0B" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    }
                  />

                  <ChartCard
                    title="Service Health"
                    subtitle="Status of critical system services"
                    chart={
                      <div className="space-y-4">
                        {serviceHealth.map((service) => (
                          <div key={service.name} className="flex items-center justify-between p-4 bg-primary-50/50 dark:bg-dark-background rounded-lg transition-all duration-300 hover:translate-y-[-2px]">
                            <div className="flex items-center">
                              {getStatusIcon(service.status)}
                              <span className="ml-2 font-medium text-primary-900 dark:text-primary-100">{service.name}</span>
                            </div>
                            <div className="flex items-center space-x-4">
                              <div className="text-sm">
                                <span className="text-primary-500 dark:text-primary-400">Uptime: </span>
                                <span className="font-medium text-primary-900 dark:text-primary-100">{service.uptime}%</span>
                              </div>
                              <div className="text-sm">
                                <span className="text-primary-500 dark:text-primary-400">Response: </span>
                                <span className="font-medium text-primary-900 dark:text-primary-100">{service.responseTime}ms</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    }
                  />
                </div>
              </>
            )
          },
          {
            id: 'cpu',
            label: 'CPU',
            icon: <Cpu size={16} />,
            content: (
              <ChartCard
                title="CPU Performance"
                subtitle="Detailed CPU metrics and utilization"
                chart={
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={resourceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="cpu" stroke="#3575E3" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                }
              />
            )
          },
          {
            id: 'memory',
            label: 'Memory',
            icon: <HardDrive size={16} />,
            content: (
              <ChartCard
                title="Memory Usage"
                subtitle="RAM and swap memory utilization"
                chart={
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={resourceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="memory" stroke="#4FD1C5" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                }
              />
            )
          },
          {
            id: 'network',
            label: 'Network',
            icon: <Network size={16} />,
            content: (
              <ChartCard
                title="Network Traffic"
                subtitle="Bandwidth and packet metrics"
                chart={
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={resourceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="network" stroke="#F59E0B" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                }
              />
            )
          }
        ], [resourceData, getMetricIcon, getStatusIcon])}
        defaultTab="overview"
        onChange={setActiveTab}
        variant="enclosed"
        color="primary"
      />
    </Layout>
  );
};

export default PerformancePage;