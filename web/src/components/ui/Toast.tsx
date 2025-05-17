import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle, AlertTriangle, AlertCircle, Info } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface ToastProps {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
  onClose: (id: string) => void;
}

/**
 * Komponenta za prikaz obvestil (toast)
 */
const Toast: React.FC<ToastProps> = ({
  id,
  type,
  title,
  message,
  duration = 5000,
  onClose
}) => {
  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        onClose(id);
      }, duration);

      return () => {
        clearTimeout(timer);
      };
    }
  }, [id, duration, onClose]);

  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle size={20} />;
      case 'error':
        return <AlertCircle size={20} />;
      case 'warning':
        return <AlertTriangle size={20} />;
      case 'info':
        return <Info size={20} />;
    }
  };

  const getTypeClasses = () => {
    switch (type) {
      case 'success':
        return 'bg-success-100 border-success-500 text-success-700 dark:bg-success-900/30 dark:border-success-600 dark:text-success-400';
      case 'error':
        return 'bg-error-100 border-error-500 text-error-700 dark:bg-error-900/30 dark:border-error-600 dark:text-error-400';
      case 'warning':
        return 'bg-warning-100 border-warning-500 text-warning-700 dark:bg-warning-900/30 dark:border-warning-600 dark:text-warning-400';
      case 'info':
        return 'bg-primary-100 border-primary-500 text-primary-700 dark:bg-primary-900/30 dark:border-primary-600 dark:text-primary-400';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className={`w-full max-w-sm rounded-lg shadow-lg border-l-4 overflow-hidden ${getTypeClasses()}`}
    >
      <div className="p-4 flex">
        <div className="flex-shrink-0 mr-3">
          {getIcon()}
        </div>
        <div className="flex-1">
          <div className="flex items-start justify-between">
            <h3 className="text-sm font-medium">{title}</h3>
            <button
              onClick={() => onClose(id)}
              className="ml-4 flex-shrink-0 inline-flex text-current opacity-50 hover:opacity-100 focus:outline-none"
            >
              <X size={16} />
              <span className="sr-only">Close</span>
            </button>
          </div>
          {message && (
            <p className="mt-1 text-sm opacity-90">{message}</p>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default Toast;
