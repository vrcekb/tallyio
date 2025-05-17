import React from 'react';
import Card from '../ui/Card';
import { SystemHealthStatus } from '../../types';
import { getStatusColor } from '../../utils/mockData';
import { CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';

interface SystemHealthPanelProps {
  data: SystemHealthStatus[];
}

const SystemHealthPanel: React.FC<SystemHealthPanelProps> = ({ data }) => {
  const renderStatusIcon = (status: 'healthy' | 'warning' | 'critical') => {
    switch (status) {
      case 'healthy':
        return <CheckCircle size={16} className="text-success-500" />;
      case 'warning':
        return <AlertTriangle size={16} className="text-warning-500" />;
      case 'critical':
        return <AlertCircle size={16} className="text-error-500" />;
    }
  };

  return (
    <Card title="System Health">
      <div className="space-y-3 h-full flex flex-col">
        <div className="flex-1 overflow-auto max-h-[350px]">
          {data.map((item) => (
            <div key={item.component} className="flex items-center justify-between p-3 rounded bg-primary-50/50 dark:bg-dark-background mb-3 shadow-sm hover:shadow-md transition-all duration-300 hover:translate-y-[-2px]">
              <div className="flex items-center">
                {renderStatusIcon(item.status)}
                <span className="ml-2 font-medium text-primary-900 dark:text-primary-100">
                  {item.component}
                </span>
              </div>
              <div className="flex items-center">
                <span
                  className={`px-2 py-0.5 text-xs font-medium rounded ${
                    `bg-${getStatusColor(item.status)}-100/80 text-${getStatusColor(item.status)}-700 dark:bg-${getStatusColor(item.status)}-900/30 dark:text-${getStatusColor(item.status)}-400`
                  }`}
                >
                  {item.status.toUpperCase()}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};

export default SystemHealthPanel;