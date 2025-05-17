import React from 'react';
import Card from '../ui/Card';
import { Alert } from '../../types';
import { Check, AlertTriangle, AlertCircle, Info } from 'lucide-react';
import { format } from 'date-fns';

interface AlertPanelProps {
  data: Alert[];
}

const AlertPanel: React.FC<AlertPanelProps> = ({ data }) => {
  const getSeverityIcon = (severity: 'critical' | 'warning' | 'info') => {
    switch (severity) {
      case 'critical':
        return <AlertCircle size={16} className="text-error-500" />;
      case 'warning':
        return <AlertTriangle size={16} className="text-warning-500" />;
      case 'info':
        return <Info size={16} className="text-primary-500" />;
    }
  };

  const getSeverityColor = (severity: 'critical' | 'warning' | 'info'): string => {
    switch (severity) {
      case 'critical':
        return 'bg-error-100/80 text-error-700 dark:bg-error-900/30 dark:text-error-400';
      case 'warning':
        return 'bg-warning-100/80 text-warning-700 dark:bg-warning-900/30 dark:text-warning-400';
      case 'info':
        return 'bg-primary-100/80 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400';
    }
  };

  return (
    <Card title="Active Alerts">
      <div className="space-y-3">
        {data.length === 0 ? (
          <div className="py-8 flex flex-col items-center justify-center text-primary-500 dark:text-primary-400">
            <Check size={24} className="mb-2 text-success-500" />
            <p>No active alerts</p>
          </div>
        ) : (
          data.map((alert) => (
            <div
              key={alert.id}
              className="p-3 rounded bg-primary-50/50 dark:bg-dark-background"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-start">
                  {getSeverityIcon(alert.severity)}
                  <div className="ml-2">
                    <p className="text-sm font-medium text-primary-900 dark:text-primary-100">
                      {alert.message}
                    </p>
                    <p className="text-xs text-primary-500 dark:text-primary-400 mt-1">
                      {format(new Date(alert.timestamp), 'MMM dd, HH:mm')}
                    </p>
                  </div>
                </div>
                <div className="flex items-center">
                  <span
                    className={`px-2 py-0.5 text-xs font-medium rounded ${getSeverityColor(
                      alert.severity
                    )}`}
                  >
                    {alert.severity.toUpperCase()}
                  </span>
                  {!alert.acknowledged && (
                    <button className="ml-2 px-2 py-0.5 text-xs font-medium rounded bg-primary-100/80 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400 hover:bg-primary-200 dark:hover:bg-primary-800/50">
                      Acknowledge
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </Card>
  );
};

export default AlertPanel;