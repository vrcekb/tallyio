import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import NotificationItem from './NotificationItem';
import Card from './Card';
import Button from './Button';

type NotificationType = 'info' | 'success' | 'warning' | 'error';

interface Notification {
  id: string | number;
  title: string;
  message?: string;
  type?: NotificationType;
  icon?: React.ReactNode;
  timestamp?: string | Date;
  read?: boolean;
}

interface NotificationListProps {
  notifications: Notification[];
  title?: string;
  emptyState?: React.ReactNode;
  onNotificationClick?: (notification: Notification) => void;
  onNotificationClose?: (notification: Notification) => void;
  onMarkAllAsRead?: () => void;
  onClearAll?: () => void;
  className?: string;
  itemClassName?: string;
  animate?: boolean;
  showCard?: boolean;
  maxHeight?: number | string;
}

/**
 * Komponenta za prikaz seznama obvestil
 */
const NotificationList: React.FC<NotificationListProps> = ({
  notifications,
  title = 'Notifications',
  emptyState,
  onNotificationClick,
  onNotificationClose,
  onMarkAllAsRead,
  onClearAll,
  className = '',
  itemClassName = '',
  animate = true,
  showCard = true,
  maxHeight = 400
}) => {
  const hasUnread = notifications.some(notification => !notification.read);

  const content = (
    <div className={className}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100">
          {title}
        </h3>
        <div className="flex space-x-2">
          {hasUnread && onMarkAllAsRead && (
            <Button
              size="xs"
              variant="ghost"
              color="primary"
              onClick={onMarkAllAsRead}
            >
              Mark all as read
            </Button>
          )}
          {notifications.length > 0 && onClearAll && (
            <Button
              size="xs"
              variant="ghost"
              color="gray"
              onClick={onClearAll}
            >
              Clear all
            </Button>
          )}
        </div>
      </div>
      
      {notifications.length === 0 ? (
        emptyState || (
          <div className="text-center py-6 text-gray-500 dark:text-gray-400">
            No notifications to display
          </div>
        )
      ) : (
        <div 
          className="space-y-2 overflow-y-auto pr-1"
          style={{ maxHeight }}
        >
          <AnimatePresence>
            {notifications.map((notification) => (
              <NotificationItem
                key={notification.id}
                title={notification.title}
                message={notification.message}
                type={notification.type}
                icon={notification.icon}
                timestamp={notification.timestamp}
                read={notification.read}
                onClick={onNotificationClick ? () => onNotificationClick(notification) : undefined}
                onClose={onNotificationClose ? () => onNotificationClose(notification) : undefined}
                className={itemClassName}
                animate={animate}
              />
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );

  if (!showCard) {
    return content;
  }

  return (
    <Card>
      {content}
    </Card>
  );
};

export default NotificationList;
