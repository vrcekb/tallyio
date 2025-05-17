import React from 'react';

type BadgeVariant = 'solid' | 'outline' | 'subtle';
type BadgeSize = 'sm' | 'md' | 'lg';
type BadgeColor = 'primary' | 'success' | 'warning' | 'error' | 'gray';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  size?: BadgeSize;
  color?: BadgeColor;
  icon?: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

/**
 * Komponenta za prikaz značke
 */
const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'solid',
  size = 'md',
  color = 'primary',
  icon,
  className = '',
  onClick
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'outline':
        return {
          primary: 'bg-transparent border border-primary-500 text-primary-700 dark:text-primary-400',
          success: 'bg-transparent border border-success-500 text-success-700 dark:text-success-400',
          warning: 'bg-transparent border border-warning-500 text-warning-700 dark:text-warning-400',
          error: 'bg-transparent border border-error-500 text-error-700 dark:text-error-400',
          gray: 'bg-transparent border border-gray-300 text-gray-700 dark:border-gray-600 dark:text-gray-300'
        }[color];
      case 'subtle':
        return {
          primary: 'bg-primary-100/80 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400',
          success: 'bg-success-100/80 text-success-700 dark:bg-success-900/30 dark:text-success-400',
          warning: 'bg-warning-100/80 text-warning-700 dark:bg-warning-900/30 dark:text-warning-400',
          error: 'bg-error-100/80 text-error-700 dark:bg-error-900/30 dark:text-error-400',
          gray: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
        }[color];
      default:
        return {
          primary: 'bg-primary-500 text-white dark:bg-primary-600',
          success: 'bg-success-500 text-white dark:bg-success-600',
          warning: 'bg-warning-500 text-white dark:bg-warning-600',
          error: 'bg-error-500 text-white dark:bg-error-600',
          gray: 'bg-gray-500 text-white dark:bg-gray-600'
        }[color];
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return 'text-xs px-2 py-0.5';
      case 'lg':
        return 'text-sm px-3 py-1';
      default:
        return 'text-xs px-2.5 py-0.5';
    }
  };

  return (
    <span
      className={`
        inline-flex items-center font-medium rounded-full
        ${getVariantClasses()}
        ${getSizeClasses()}
        ${onClick ? 'cursor-pointer hover:opacity-90' : ''}
        ${className}
      `}
      onClick={onClick}
    >
      {icon && <span className="mr-1">{icon}</span>}
      {children}
    </span>
  );
};

export default Badge;
