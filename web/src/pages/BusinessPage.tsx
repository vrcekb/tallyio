import React from 'react';
import Layout from '../components/layout/Layout';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import Card from '../components/ui/Card';
import { customerMetrics, generateRevenueData, marketShareData } from '../mockData/business';

const BusinessPage: React.FC = () => {
  const revenueData = generateRevenueData(30);

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up':
        return <TrendingUp size={16} className="text-success-500" />;
      case 'down':
        return <TrendingDown size={16} className="text-error-500" />;
      case 'stable':
        return <Minus size={16} className="text-gray-500" />;
    }
  };

  return (
    <Layout>
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Business Analytics</h1>
        <div className="flex items-center space-x-4">
          <select className="bg-white dark:bg-dark-card border border-gray-200 dark:border-dark-border rounded-lg px-3 py-2 text-sm">
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
            <option value="1y">Last Year</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {customerMetrics.map((metric, index) => (
          <Card key={index}>
            <div className="flex flex-col">
              <span className="text-sm text-gray-500 dark:text-gray-400">{metric.title}</span>
              <div className="flex items-baseline mt-2">
                <span className="text-2xl font-bold text-gray-900 dark:text-white">{metric.value}</span>
                <div className="flex items-center ml-2">
                  {getTrendIcon(metric.trend)}
                  <span className={`text-sm ml-1 ${
                    metric.change > 0 ? 'text-success-500' : metric.change < 0 ? 'text-error-500' : 'text-gray-500'
                  }`}>
                    {metric.change > 0 ? '+' : ''}{metric.change}%
                  </span>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Revenue Overview">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={revenueData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="revenue" fill="#3575E3" />
                <Bar dataKey="profit" fill="#4FD1C5" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Market Share Distribution">
          <div className="space-y-4">
            {marketShareData.map((item) => (
              <div key={item.network} className="flex items-center">
                <div className="flex-1">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{item.network}</span>
                    <span className="text-sm text-gray-500 dark:text-gray-400">{item.share}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-dark-background rounded-full h-2">
                    <div
                      className="bg-primary-500 h-2 rounded-full"
                      style={{ width: `${item.share}%` }}
                    />
                  </div>
                </div>
                <div className="ml-4 flex items-center">
                  <span className={`text-sm ${item.change >= 0 ? 'text-success-500' : 'text-error-500'}`}>
                    {item.change > 0 ? '+' : ''}{item.change}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </Layout>
  );
};

export default BusinessPage;