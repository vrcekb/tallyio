import React from 'react';
import { motion } from 'framer-motion';
import Button from './Button';

interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
  action?: {
    label: string;
    onClick: () => void;
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
  iconClassName?: string;
  titleClassName?: string;
  descriptionClassName?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz praznega stanja
 */
const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  icon,
  action,
  secondaryAction,
  className = '',
  iconClassName = '',
  titleClassName = '',
  descriptionClassName = '',
  animate = true
}) => {
  const emptyStateContent = (
    <div className={`
      flex flex-col items-center justify-center text-center p-8
      ${className}
    `}>
      {icon && (
        <div className={`
          text-gray-400 dark:text-gray-500 mb-4
          ${iconClassName}
        `}>
          {icon}
        </div>
      )}
      <h3 className={`
        text-lg font-medium text-primary-900 dark:text-primary-100 mb-2
        ${titleClassName}
      `}>
        {title}
      </h3>
      {description && (
        <p className={`
          text-sm text-gray-500 dark:text-gray-400 max-w-md mb-6
          ${descriptionClassName}
        `}>
          {description}
        </p>
      )}
      {(action || secondaryAction) && (
        <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
          {action && (
            <Button
              color="primary"
              onClick={action.onClick}
            >
              {action.label}
            </Button>
          )}
          {secondaryAction && (
            <Button
              variant="outline"
              color="gray"
              onClick={secondaryAction.onClick}
            >
              {secondaryAction.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );

  if (!animate) {
    return emptyStateContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      {emptyStateContent}
    </motion.div>
  );
};

export default EmptyState;
