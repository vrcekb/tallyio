import React from 'react';
import { AlertTriangle, TrendingDown, TrendingUp } from 'lucide-react';

const RiskManagementPage: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Risk Management Dashboard</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-green-500 mr-3" />
              <div>
                <h3 className="text-lg font-semibold">Current Risk Score</h3>
                <p className="text-2xl font-bold text-green-500">Low (2.4)</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <TrendingDown className="h-8 w-8 text-red-500 mr-3" />
              <div>
                <h3 className="text-lg font-semibold">Risk Threshold</h3>
                <p className="text-2xl font-bold text-gray-700">7.5</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <AlertTriangle className="h-8 w-8 text-yellow-500 mr-3" />
              <div>
                <h3 className="text-lg font-semibold">Active Alerts</h3>
                <p className="text-2xl font-bold text-yellow-500">2</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Risk Factors</h2>
          <div className="space-y-4">
            {[
              { name: 'Market Volatility', value: '3.2', status: 'normal' },
              { name: 'Liquidity Risk', value: '2.8', status: 'normal' },
              { name: 'Smart Contract Risk', value: '1.5', status: 'low' },
              { name: 'Protocol Risk', value: '4.2', status: 'elevated' }
            ].map((factor) => (
              <div key={factor.name} className="flex items-center justify-between">
                <span className="text-gray-700">{factor.name}</span>
                <span className={`font-semibold ${
                  factor.status === 'low' ? 'text-green-500' :
                  factor.status === 'normal' ? 'text-blue-500' :
                  'text-yellow-500'
                }`}>{factor.value}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Risk Events</h2>
          <div className="space-y-4">
            {[
              { event: 'Unusual trading volume detected', time: '2 hours ago', severity: 'medium' },
              { event: 'New smart contract deployment', time: '5 hours ago', severity: 'low' },
              { event: 'Price oracle deviation', time: '1 day ago', severity: 'high' }
            ].map((event, index) => (
              <div key={index} className="flex items-center justify-between border-b pb-2">
                <div>
                  <p className="text-gray-800">{event.event}</p>
                  <p className="text-sm text-gray-500">{event.time}</p>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm ${
                  event.severity === 'low' ? 'bg-green-100 text-green-800' :
                  event.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {event.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskManagementPage;