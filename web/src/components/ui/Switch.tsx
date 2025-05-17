import React, { forwardRef } from 'react';
import { motion } from 'framer-motion';

interface SwitchProps {
  checked: boolean;
  onChange?: (checked: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'success' | 'warning' | 'error';
  className?: string;
  labelClassName?: string;
  descriptionClassName?: string;
  labelPosition?: 'left' | 'right';
}

/**
 * Komponenta za prikaz stikala
 */
const Switch = forwardRef<HTMLInputElement, SwitchProps>(({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  size = 'md',
  color = 'primary',
  className = '',
  labelClassName = '',
  descriptionClassName = '',
  labelPosition = 'right',
}, ref) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!disabled && onChange) {
      onChange(e.target.checked);
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return {
          container: 'w-8 h-4',
          circle: 'w-3 h-3',
          translate: 'translate-x-4',
        };
      case 'lg':
        return {
          container: 'w-14 h-7',
          circle: 'w-6 h-6',
          translate: 'translate-x-7',
        };
      default:
        return {
          container: 'w-11 h-6',
          circle: 'w-5 h-5',
          translate: 'translate-x-5',
        };
    }
  };

  const getColorClass = () => {
    switch (color) {
      case 'success':
        return 'bg-success-500 dark:bg-success-600';
      case 'warning':
        return 'bg-warning-500 dark:bg-warning-600';
      case 'error':
        return 'bg-error-500 dark:bg-error-600';
      default:
        return 'bg-primary-500 dark:bg-primary-600';
    }
  };

  const sizeClasses = getSizeClasses();
  const colorClass = getColorClass();

  const renderLabel = () => (
    <div>
      {label && (
        <span className={`font-medium text-primary-900 dark:text-primary-100 ${labelClassName}`}>
          {label}
        </span>
      )}
      {description && (
        <p className={`text-sm text-gray-500 dark:text-gray-400 ${descriptionClassName}`}>
          {description}
        </p>
      )}
    </div>
  );

  return (
    <label className={`flex items-center ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'} ${className}`}>
      {label && labelPosition === 'left' && (
        <div className="mr-3">
          {renderLabel()}
        </div>
      )}
      
      <div className="relative inline-flex items-center">
        <input
          ref={ref}
          type="checkbox"
          className="sr-only"
          checked={checked}
          onChange={handleChange}
          disabled={disabled}
        />
        <div
          className={`
            ${sizeClasses.container} rounded-full transition-colors duration-200 ease-in-out
            ${checked ? colorClass : 'bg-gray-200 dark:bg-gray-700'}
          `}
        >
          <motion.div
            initial={false}
            animate={{
              x: checked ? parseInt(sizeClasses.translate.split('-x-')[1]) : 0,
            }}
            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            className={`
              ${sizeClasses.circle} rounded-full bg-white shadow-sm transform ring-0
              absolute left-0.5 top-0.5
            `}
          />
        </div>
      </div>
      
      {label && labelPosition === 'right' && (
        <div className="ml-3">
          {renderLabel()}
        </div>
      )}
    </label>
  );
});

Switch.displayName = 'Switch';

export default Switch;
