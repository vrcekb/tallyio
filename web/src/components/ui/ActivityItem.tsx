import React from 'react';
import { motion } from 'framer-motion';

interface ActivityItemProps {
  title: React.ReactNode;
  description?: React.ReactNode;
  timestamp?: string | Date;
  icon?: React.ReactNode;
  iconBackground?: string;
  action?: React.ReactNode;
  isLast?: boolean;
  className?: string;
  iconClassName?: string;
  contentClassName?: string;
  titleClassName?: string;
  descriptionClassName?: string;
  timestampClassName?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz elementa aktivnosti
 */
const ActivityItem: React.FC<ActivityItemProps> = ({
  title,
  description,
  timestamp,
  icon,
  iconBackground = 'bg-primary-500',
  action,
  isLast = false,
  className = '',
  iconClassName = '',
  contentClassName = '',
  titleClassName = '',
  descriptionClassName = '',
  timestampClassName = '',
  animate = true
}) => {
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
    
    // If it's yesterday, show "Yesterday" and time
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    if (
      date.getDate() === yesterday.getDate() &&
      date.getMonth() === yesterday.getMonth() &&
      date.getFullYear() === yesterday.getFullYear()
    ) {
      return `Yesterday, ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    
    // Otherwise, show full date
    return date.toLocaleDateString([], { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const activityContent = (
    <div className={`flex ${className}`}>
      {icon && (
        <div className="flex-shrink-0 mr-4">
          <div className={`relative mt-1`}>
            <div className={`
              w-8 h-8 rounded-full flex items-center justify-center text-white
              ${iconBackground} ${iconClassName}
            `}>
              {icon}
            </div>
            {!isLast && (
              <div className="absolute top-8 left-1/2 w-px h-full -ml-px bg-gray-200 dark:bg-gray-700" />
            )}
          </div>
        </div>
      )}
      
      <div className={`flex-1 min-w-0 ${contentClassName}`}>
        <div className="flex items-start justify-between">
          <div>
            <div className={`text-sm font-medium text-primary-900 dark:text-primary-100 ${titleClassName}`}>
              {title}
            </div>
            {description && (
              <div className={`mt-1 text-sm text-gray-500 dark:text-gray-400 ${descriptionClassName}`}>
                {description}
              </div>
            )}
          </div>
          
          {timestamp && (
            <div className={`ml-4 flex-shrink-0 text-xs text-gray-500 dark:text-gray-400 ${timestampClassName}`}>
              {formatTimestamp(timestamp)}
            </div>
          )}
        </div>
        
        {action && (
          <div className="mt-2">
            {action}
          </div>
        )}
      </div>
    </div>
  );

  if (!animate) {
    return activityContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      {activityContent}
    </motion.div>
  );
};

export default ActivityItem;
