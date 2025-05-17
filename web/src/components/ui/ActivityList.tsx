import React from 'react';
import { motion } from 'framer-motion';
import ActivityItem from './ActivityItem';
import Card from './Card';

interface Activity {
  id: string | number;
  title: React.ReactNode;
  description?: React.ReactNode;
  timestamp?: string | Date;
  icon?: React.ReactNode;
  iconBackground?: string;
  action?: React.ReactNode;
}

interface ActivityListProps {
  activities: Activity[];
  title?: string;
  emptyState?: React.ReactNode;
  className?: string;
  itemClassName?: string;
  animate?: boolean;
  showCard?: boolean;
  maxHeight?: number | string;
}

/**
 * Komponenta za prikaz seznama aktivnosti
 */
const ActivityList: React.FC<ActivityListProps> = ({
  activities,
  title,
  emptyState,
  className = '',
  itemClassName = '',
  animate = true,
  showCard = true,
  maxHeight
}) => {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, x: -10 },
    show: { opacity: 1, x: 0 }
  };

  const content = (
    <div className={className}>
      {title && (
        <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100 mb-4">
          {title}
        </h3>
      )}
      
      {activities.length === 0 ? (
        emptyState || (
          <div className="text-center py-6 text-gray-500 dark:text-gray-400">
            No activities to display
          </div>
        )
      ) : (
        <div 
          className="space-y-6"
          style={maxHeight ? { maxHeight, overflowY: 'auto' } : undefined}
        >
          {animate ? (
            <motion.div
              variants={container}
              initial="hidden"
              animate="show"
              className="space-y-6"
            >
              {activities.map((activity, index) => (
                <motion.div key={activity.id} variants={item}>
                  <ActivityItem
                    title={activity.title}
                    description={activity.description}
                    timestamp={activity.timestamp}
                    icon={activity.icon}
                    iconBackground={activity.iconBackground}
                    action={activity.action}
                    isLast={index === activities.length - 1}
                    className={itemClassName}
                    animate={false}
                  />
                </motion.div>
              ))}
            </motion.div>
          ) : (
            <div className="space-y-6">
              {activities.map((activity, index) => (
                <ActivityItem
                  key={activity.id}
                  title={activity.title}
                  description={activity.description}
                  timestamp={activity.timestamp}
                  icon={activity.icon}
                  iconBackground={activity.iconBackground}
                  action={activity.action}
                  isLast={index === activities.length - 1}
                  className={itemClassName}
                  animate={false}
                />
              ))}
            </div>
          )}
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

export default ActivityList;
