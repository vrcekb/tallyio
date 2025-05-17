import React from 'react';
import { motion } from 'framer-motion';
import { X } from 'lucide-react';

type NotificationType = 'info' | 'success' | 'warning' | 'error';

interface NotificationItemProps {
  title: string;
  message?: string;
  type?: NotificationType;
  icon?: React.ReactNode;
  timestamp?: string | Date;
  read?: boolean;
  onClose?: () => void;
  onClick?: () => void;
  className?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz elementa obvestila
 */
const NotificationItem: React.FC<NotificationItemProps> = ({
  title,
  message,
  type = 'info',
  icon,
  timestamp,
  read = false,
  onClose,
  onClick,
  className = '',
  animate = true
}) => {
  const getTypeClasses = () => {
    switch (type) {
      case 'success':
        return {
          dot: 'bg-success-500',
          icon: 'text-success-500 dark:text-success-400 bg-success-100 dark:bg-success-900/20'
        };
      case 'warning':
        return {
          dot: 'bg-warning-500',
          icon: 'text-warning-500 dark:text-warning-400 bg-warning-100 dark:bg-warning-900/20'
        };
      case 'error':
        return {
          dot: 'bg-error-500',
          icon: 'text-error-500 dark:text-error-400 bg-error-100 dark:bg-error-900/20'
        };
      default:
        return {
          dot: 'bg-primary-500',
          icon: 'text-primary-500 dark:text-primary-400 bg-primary-100 dark:bg-primary-900/20'
        };
    }
  };

  const formatTimestamp = (timestamp: string | Date) => {
    if (!timestamp) return '';
    
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    
    // If it's today, show only time
    const today = new Date();
    if (
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear()
    ) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // If it's yesterday, show "Yesterday"
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    if (
      date.getDate() === yesterday.getDate() &&
      date.getMonth() === yesterday.getMonth() &&
      date.getFullYear() === yesterday.getFullYear()
    ) {
      return 'Yesterday';
    }
    
    // Otherwise, show date
    return date.toLocaleDateString([], { 
      month: 'short', 
      day: 'numeric'
    });
  };

  const typeClasses = getTypeClasses();

  const notificationContent = (
    <div 
      className={`
        relative flex items-start p-4 rounded-lg
        ${read ? 'bg-white dark:bg-dark-card' : 'bg-primary-50/50 dark:bg-primary-900/10'}
        ${onClick ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-dark-background' : ''}
        transition-colors duration-200
        ${className}
      `}
      onClick={onClick}
    >
      {!read && (
        <div className="absolute top-4 left-0 w-1 h-1.5 rounded-full bg-primary-500" />
      )}
      
      {icon && (
        <div className={`
          flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mr-3
          ${typeClasses.icon}
        `}>
          {icon}
        </div>
      )}
      
      <div className="flex-1 min-w-0">
        <div className="flex justify-between">
          <p className="text-sm font-medium text-primary-900 dark:text-primary-100">
            {title}
          </p>
          {timestamp && (
            <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
              {formatTimestamp(timestamp)}
            </span>
          )}
        </div>
        {message && (
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {message}
          </p>
        )}
      </div>
      
      {onClose && (
        <button
          className="ml-4 flex-shrink-0 text-gray-400 hover:text-gray-500 dark:text-gray-500 dark:hover:text-gray-400"
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
        >
          <span className="sr-only">Close</span>
          <X size={16} />
        </button>
      )}
    </div>
  );

  if (!animate) {
    return notificationContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0, marginBottom: 0 }}
      transition={{ duration: 0.2 }}
    >
      {notificationContent}
    </motion.div>
  );
};

export default NotificationItem;
