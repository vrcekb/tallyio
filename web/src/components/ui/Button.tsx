import React from 'react';
import { motion } from 'framer-motion';

type ButtonVariant = 'solid' | 'outline' | 'ghost' | 'link';
type ButtonSize = 'xs' | 'sm' | 'md' | 'lg';
type ButtonColor = 'primary' | 'success' | 'warning' | 'error' | 'gray';

interface ButtonProps {
  children: React.ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  color?: ButtonColor;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
  disabled?: boolean;
  loading?: boolean;
  loadingText?: string;
  className?: string;
  animate?: boolean;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  type?: 'button' | 'submit' | 'reset';
}

/**
 * Komponenta za prikaz gumba
 */
const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'solid',
  size = 'md',
  color = 'primary',
  leftIcon,
  rightIcon,
  fullWidth = false,
  disabled = false,
  loading = false,
  loadingText,
  className = '',
  animate = true,
  onClick,
  type = 'button'
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'outline':
        return {
          primary: 'border border-primary-500 text-primary-700 hover:bg-primary-50 dark:text-primary-400 dark:hover:bg-primary-900/20',
          success: 'border border-success-500 text-success-700 hover:bg-success-50 dark:text-success-400 dark:hover:bg-success-900/20',
          warning: 'border border-warning-500 text-warning-700 hover:bg-warning-50 dark:text-warning-400 dark:hover:bg-warning-900/20',
          error: 'border border-error-500 text-error-700 hover:bg-error-50 dark:text-error-400 dark:hover:bg-error-900/20',
          gray: 'border border-gray-300 text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800'
        }[color];
      case 'ghost':
        return {
          primary: 'text-primary-700 hover:bg-primary-50 dark:text-primary-400 dark:hover:bg-primary-900/20',
          success: 'text-success-700 hover:bg-success-50 dark:text-success-400 dark:hover:bg-success-900/20',
          warning: 'text-warning-700 hover:bg-warning-50 dark:text-warning-400 dark:hover:bg-warning-900/20',
          error: 'text-error-700 hover:bg-error-50 dark:text-error-400 dark:hover:bg-error-900/20',
          gray: 'text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-800'
        }[color];
      case 'link':
        return {
          primary: 'text-primary-700 hover:underline dark:text-primary-400',
          success: 'text-success-700 hover:underline dark:text-success-400',
          warning: 'text-warning-700 hover:underline dark:text-warning-400',
          error: 'text-error-700 hover:underline dark:text-error-400',
          gray: 'text-gray-700 hover:underline dark:text-gray-300'
        }[color];
      default:
        return {
          primary: 'bg-primary-500 hover:bg-primary-600 text-white dark:bg-primary-600 dark:hover:bg-primary-700',
          success: 'bg-success-500 hover:bg-success-600 text-white dark:bg-success-600 dark:hover:bg-success-700',
          warning: 'bg-warning-500 hover:bg-warning-600 text-white dark:bg-warning-600 dark:hover:bg-warning-700',
          error: 'bg-error-500 hover:bg-error-600 text-white dark:bg-error-600 dark:hover:bg-error-700',
          gray: 'bg-gray-500 hover:bg-gray-600 text-white dark:bg-gray-600 dark:hover:bg-gray-700'
        }[color];
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'xs':
        return 'text-xs px-2 py-1';
      case 'sm':
        return 'text-sm px-3 py-1.5';
      case 'lg':
        return 'text-base px-5 py-2.5';
      default:
        return 'text-sm px-4 py-2';
    }
  };

  const getDisabledClasses = () => {
    if (variant === 'link') {
      return 'opacity-50 cursor-not-allowed';
    }
    return 'opacity-50 cursor-not-allowed pointer-events-none';
  };

  const buttonContent = (
    <button
      type={type}
      className={`
        inline-flex items-center justify-center font-medium rounded-md
        transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2
        ${variant !== 'link' ? 'shadow-sm' : ''}
        ${getVariantClasses()}
        ${getSizeClasses()}
        ${fullWidth ? 'w-full' : ''}
        ${(disabled || loading) ? getDisabledClasses() : ''}
        ${className}
      `}
      disabled={disabled || loading}
      onClick={onClick}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {leftIcon && !loading && <span className="mr-2">{leftIcon}</span>}
      {loading && loadingText ? loadingText : children}
      {rightIcon && !loading && <span className="ml-2">{rightIcon}</span>}
    </button>
  );

  if (!animate || disabled || loading) {
    return buttonContent;
  }

  return (
    <motion.div
      whileTap={{ scale: 0.98 }}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      {buttonContent}
    </motion.div>
  );
};

export default Button;
