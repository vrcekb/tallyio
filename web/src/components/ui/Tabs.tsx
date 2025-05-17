import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
  content: React.ReactNode;
  disabled?: boolean;
}

interface TabsProps {
  tabs: Tab[];
  defaultTab?: string;
  onChange?: (tabId: string) => void;
  variant?: 'line' | 'enclosed' | 'pill';
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'success' | 'warning' | 'error' | 'gray';
  fullWidth?: boolean;
  className?: string;
  tabsClassName?: string;
  tabClassName?: string;
  contentClassName?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz zavihkov
 */
const Tabs: React.FC<TabsProps> = ({
  tabs,
  defaultTab,
  onChange,
  variant = 'line',
  size = 'md',
  color = 'primary',
  fullWidth = false,
  className = '',
  tabsClassName = '',
  tabClassName = '',
  contentClassName = '',
  animate = true
}) => {
  const [activeTab, setActiveTab] = useState<string>(defaultTab || (tabs.length > 0 ? tabs[0].id : ''));
  const tabsRef = useRef<HTMLDivElement>(null);
  const [indicatorStyle, setIndicatorStyle] = useState({ left: 0, width: 0 });

  useEffect(() => {
    if (defaultTab) {
      setActiveTab(defaultTab);
    }
  }, [defaultTab]);

  useEffect(() => {
    updateIndicator();
  }, [activeTab, tabs]);

  const updateIndicator = () => {
    if (variant === 'line' && tabsRef.current) {
      const activeTabElement = tabsRef.current.querySelector(`[data-tab-id="${activeTab}"]`) as HTMLElement;
      if (activeTabElement) {
        const { offsetLeft, offsetWidth } = activeTabElement;
        setIndicatorStyle({
          left: offsetLeft,
          width: offsetWidth
        });
      }
    }
  };

  const handleTabClick = (tabId: string) => {
    setActiveTab(tabId);
    if (onChange) {
      onChange(tabId);
    }
  };

  const getVariantClasses = () => {
    switch (variant) {
      case 'enclosed':
        return {
          container: 'border-b border-gray-200 dark:border-gray-700',
          tabs: 'space-x-1',
          tab: (isActive: boolean, isDisabled: boolean) => `
            rounded-t-lg border-b-2 border-l border-r border-t
            ${isActive
              ? `border-b-transparent border-l-gray-200 border-r-gray-200 border-t-gray-200 dark:border-l-gray-700 dark:border-r-gray-700 dark:border-t-gray-700 bg-white dark:bg-dark-card`
              : 'border-transparent'
            }
            ${isDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:text-gray-700 hover:border-gray-300 dark:hover:text-gray-300'}
          `,
          content: 'border border-t-0 border-gray-200 dark:border-gray-700 rounded-b-lg p-4'
        };
      case 'pill':
        return {
          container: '',
          tabs: 'space-x-1',
          tab: (isActive: boolean, isDisabled: boolean) => `
            rounded-full
            ${isActive
              ? getActiveColorClass()
              : 'bg-transparent'
            }
            ${isDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800'}
          `,
          content: 'pt-4'
        };
      default: // line
        return {
          container: 'relative border-b border-gray-200 dark:border-gray-700',
          tabs: '',
          tab: (isActive: boolean, isDisabled: boolean) => `
            border-b-2
            ${isActive
              ? getBorderColorClass()
              : 'border-transparent'
            }
            ${isDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:text-gray-700 hover:border-gray-300 dark:hover:text-gray-300'}
          `,
          content: 'pt-4'
        };
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return 'text-xs px-2 py-1';
      case 'lg':
        return 'text-base px-4 py-3';
      default:
        return 'text-sm px-3 py-2';
    }
  };

  const getActiveColorClass = () => {
    switch (color) {
      case 'success':
        return 'bg-success-500 text-white dark:bg-success-600';
      case 'warning':
        return 'bg-warning-500 text-white dark:bg-warning-600';
      case 'error':
        return 'bg-error-500 text-white dark:bg-error-600';
      case 'gray':
        return 'bg-gray-500 text-white dark:bg-gray-600';
      default:
        return 'bg-primary-500 text-white dark:bg-primary-600';
    }
  };

  const getBorderColorClass = () => {
    switch (color) {
      case 'success':
        return 'border-success-500 text-success-600 dark:border-success-400 dark:text-success-400';
      case 'warning':
        return 'border-warning-500 text-warning-600 dark:border-warning-400 dark:text-warning-400';
      case 'error':
        return 'border-error-500 text-error-600 dark:border-error-400 dark:text-error-400';
      case 'gray':
        return 'border-gray-500 text-gray-600 dark:border-gray-400 dark:text-gray-400';
      default:
        return 'border-primary-500 text-primary-600 dark:border-primary-400 dark:text-primary-400';
    }
  };

  const variantClasses = getVariantClasses();
  const sizeClasses = getSizeClasses();
  const activeTabContent = tabs.find(tab => tab.id === activeTab)?.content;

  return (
    <div className={className}>
      <div className={`${variantClasses.container}`}>
        <div
          ref={tabsRef}
          className={`flex ${fullWidth ? 'w-full' : ''} ${variantClasses.tabs} ${tabsClassName}`}
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              data-tab-id={tab.id}
              className={`
                ${fullWidth ? 'flex-1 text-center' : ''}
                ${sizeClasses}
                ${variantClasses.tab(activeTab === tab.id, !!tab.disabled)}
                ${activeTab === tab.id
                  ? 'font-medium'
                  : 'text-gray-500 dark:text-gray-400'
                }
                ${tabClassName}
              `}
              onClick={() => !tab.disabled && handleTabClick(tab.id)}
              disabled={tab.disabled}
            >
              <div className="flex items-center justify-center">
                {tab.icon && <span className="mr-2">{tab.icon}</span>}
                {tab.label}
              </div>
            </button>
          ))}
          {variant === 'line' && (
            <div
              className={`absolute bottom-0 h-0.5 transition-all duration-300 ease-in-out ${getBorderColorClass()}`}
              style={{ left: `${indicatorStyle.left}px`, width: `${indicatorStyle.width}px` }}
            />
          )}
        </div>
      </div>
      <div className={`${variantClasses.content} ${contentClassName}`}>
        {animate ? (
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.3 }}
          >
            {activeTabContent}
          </motion.div>
        ) : (
          activeTabContent
        )}
      </div>
    </div>
  );
};

export default Tabs;
