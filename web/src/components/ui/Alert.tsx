import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, AlertTriangle, CheckCircle, Info, X } from 'lucide-react';

type AlertStatus = 'info' | 'success' | 'warning' | 'error';
type AlertVariant = 'solid' | 'subtle' | 'outline';

interface AlertProps {
  status?: AlertStatus;
  variant?: AlertVariant;
  title?: string;
  children: React.ReactNode;
  icon?: React.ReactNode;
  onClose?: () => void;
  className?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz opozorila
 */
const Alert: React.FC<AlertProps> = ({
  status = 'info',
  variant = 'subtle',
  title,
  children,
  icon,
  onClose,
  className = '',
  animate = true
}) => {
  const getStatusIcon = () => {
    if (icon) return icon;

    switch (status) {
      case 'success':
        return <CheckCircle size={20} />;
      case 'warning':
        return <AlertTriangle size={20} />;
      case 'error':
        return <AlertCircle size={20} />;
      default:
        return <Info size={20} />;
    }
  };

  const getVariantClasses = () => {
    switch (variant) {
      case 'solid':
        return {
          info: 'bg-primary-500 text-white',
          success: 'bg-success-500 text-white',
          warning: 'bg-warning-500 text-white',
          error: 'bg-error-500 text-white'
        }[status];
      case 'outline':
        return {
          info: 'bg-transparent border border-primary-500 text-primary-700 dark:text-primary-400',
          success: 'bg-transparent border border-success-500 text-success-700 dark:text-success-400',
          warning: 'bg-transparent border border-warning-500 text-warning-700 dark:text-warning-400',
          error: 'bg-transparent border border-error-500 text-error-700 dark:text-error-400'
        }[status];
      default:
        return {
          info: 'bg-primary-100/80 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400',
          success: 'bg-success-100/80 text-success-700 dark:bg-success-900/30 dark:text-success-400',
          warning: 'bg-warning-100/80 text-warning-700 dark:bg-warning-900/30 dark:text-warning-400',
          error: 'bg-error-100/80 text-error-700 dark:bg-error-900/30 dark:text-error-400'
        }[status];
    }
  };

  const getIconColor = () => {
    if (variant === 'solid') return 'text-white';

    return {
      info: 'text-primary-500 dark:text-primary-400',
      success: 'text-success-500 dark:text-success-400',
      warning: 'text-warning-500 dark:text-warning-400',
      error: 'text-error-500 dark:text-error-400'
    }[status];
  };

  const alertContent = (
    <div className={`
      flex rounded-md p-4 ${getVariantClasses()} ${className}
    `}>
      <div className={`flex-shrink-0 ${getIconColor()}`}>
        {getStatusIcon()}
      </div>
      <div className="ml-3 flex-1">
        {title && (
          <h3 className="text-sm font-medium">
            {title}
          </h3>
        )}
        <div className={`text-sm ${title ? 'mt-2' : ''}`}>
          {children}
        </div>
      </div>
      {onClose && (
        <div className="ml-auto pl-3">
          <button
            type="button"
            className={`
              inline-flex rounded-md p-1.5 focus:outline-none focus:ring-2 focus:ring-offset-2
              ${variant === 'solid' ? 'text-white hover:bg-white/20 focus:ring-white' : `${getIconColor()} hover:bg-gray-100 dark:hover:bg-gray-800 focus:ring-${status}-500`}
            `}
            onClick={onClose}
          >
            <span className="sr-only">Dismiss</span>
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  );

  if (!animate) {
    return alertContent;
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.3 }}
      >
        {alertContent}
      </motion.div>
    </AnimatePresence>
  );
};

export default Alert;
