import React, { useState } from 'react';
import { Bell, Menu } from 'lucide-react';
import TimeRangeSelector from '../ui/TimeRangeSelector';
import WebSocketStatus from '../ui/WebSocketStatus';
import ThemeSelector from '../ui/ThemeSelector';
import ThemeNameSelector from '../ui/ThemeNameSelector';
import AccessibilityMenu from '../ui/AccessibilityMenu';
import UserMenu from './UserMenu';
import { TimeRange } from '../../types';
import { alerts } from '../../mockData';

interface HeaderProps {
  timeRange: TimeRange;
  onTimeRangeChange: (range: TimeRange) => void;
  isDarkMode?: boolean; // Označeno kot opcijsko, ker trenutno ni uporabljeno
  toggleDarkMode?: () => void; // Označeno kot opcijsko, ker trenutno ni uporabljeno
  toggleSidebar: () => void;
}

const Header: React.FC<HeaderProps> = ({
  timeRange,
  onTimeRangeChange,
  // isDarkMode in toggleDarkMode trenutno nista uporabljena
  // isDarkMode,
  // toggleDarkMode,
  toggleSidebar
}) => {
  const [showAlerts, setShowAlerts] = useState(false);
  const activeAlerts = alerts.filter(alert => !alert.acknowledged);
  const lastUpdated = new Date().toLocaleTimeString();

  const getSeverityColor = (severity: 'critical' | 'warning' | 'info'): string => {
    switch (severity) {
      case 'critical':
        return 'bg-error-100 text-error-700 dark:bg-error-900/30 dark:text-error-400';
      case 'warning':
        return 'bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-400';
      case 'info':
        return 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400';
    }
  };

  return (
    <header className="sticky top-0 z-40 bg-white dark:bg-dark-card border-b border-gray-200 dark:border-dark-border">
      <div className="flex items-center justify-between p-4">
        <div className="flex items-center">
          <button
            onClick={toggleSidebar}
            className="mr-4 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-background lg:hidden"
          >
            <Menu size={20} className="text-gray-700 dark:text-gray-300" />
          </button>
          <div className="flex items-center">
            <h1 className="text-xl font-bold text-primary-600 dark:text-primary-400">TallyIO</h1>
            <span className="ml-2 text-xs font-medium text-gray-500 dark:text-gray-400 hidden sm:inline">
              Autonomous MEV &amp; Liquidation
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 md:gap-4">
          <div className="hidden md:block">
            <TimeRangeSelector
              value={timeRange}
              onChange={onTimeRangeChange}
            />
          </div>

          <div className="text-xs text-gray-500 dark:text-gray-400 hidden lg:flex items-center">
            <span className="mr-1">Last updated:</span>
            <span className="font-medium">{lastUpdated}</span>
            <span className="ml-2 w-2 h-2 bg-success-500 rounded-full animate-pulse"></span>
          </div>

          <div className="hidden md:block">
            <WebSocketStatus />
          </div>

          <div className="flex items-center space-x-2">
            <ThemeSelector />
            <ThemeNameSelector />
            <AccessibilityMenu />

            <div className="relative">
              <button
                onClick={() => setShowAlerts(!showAlerts)}
                className="relative p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-background"
                aria-label={`Alerts (${activeAlerts.length})`}
                aria-expanded={showAlerts}
              >
                <Bell size={18} className="text-gray-700 dark:text-gray-300" />
                {activeAlerts.length > 0 && (
                  <span className="absolute top-1 right-1 w-4 h-4 bg-error-500 rounded-full flex items-center justify-center text-white text-xs" aria-hidden="true">
                    {activeAlerts.length}
                  </span>
                )}
              </button>

              {showAlerts && activeAlerts.length > 0 && (
                <div className="absolute right-0 mt-2 w-96 bg-white dark:bg-dark-card rounded-lg shadow-lg border border-gray-200 dark:border-dark-border overflow-hidden">
                  <div className="p-3 border-b border-gray-200 dark:border-dark-border bg-gray-50 dark:bg-dark-background">
                    <h3 className="font-semibold text-gray-800 dark:text-gray-200">Active Alerts</h3>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    {activeAlerts.map((alert) => (
                      <div
                        key={alert.id}
                        className="p-3 border-b border-gray-100 dark:border-dark-border last:border-0 hover:bg-gray-50 dark:hover:bg-dark-background"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start">
                            <div className="flex-1">
                              <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
                                {alert.message}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                {new Date(alert.timestamp).toLocaleString()}
                              </p>
                            </div>
                          </div>
                          <span
                            className={`px-2 py-0.5 text-xs font-medium rounded-full ${getSeverityColor(
                              alert.severity
                            )}`}
                          >
                            {alert.severity.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <UserMenu />
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;