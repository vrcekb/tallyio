import React, { forwardRef } from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  helperText?: string;
  error?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
  className?: string;
  inputClassName?: string;
  labelClassName?: string;
  helperTextClassName?: string;
  errorClassName?: string;
}

/**
 * Komponenta za prikaz vnosnega polja
 */
const Input = forwardRef<HTMLInputElement, InputProps>(({
  label,
  helperText,
  error,
  leftIcon,
  rightIcon,
  fullWidth = false,
  className = '',
  inputClassName = '',
  labelClassName = '',
  helperTextClassName = '',
  errorClassName = '',
  ...props
}, ref) => {
  const hasError = !!error;

  return (
    <div className={`${fullWidth ? 'w-full' : ''} ${className}`}>
      {label && (
        <label
          htmlFor={props.id}
          className={`block text-sm font-medium text-primary-900 dark:text-primary-100 mb-1 ${labelClassName}`}
        >
          {label}
        </label>
      )}
      <div className="relative">
        {leftIcon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            {leftIcon}
          </div>
        )}
        <input
          ref={ref}
          className={`
            block rounded-md shadow-sm
            ${fullWidth ? 'w-full' : ''}
            ${leftIcon ? 'pl-10' : ''}
            ${rightIcon ? 'pr-10' : ''}
            ${hasError
              ? 'border-error-500 focus:ring-error-500 focus:border-error-500 dark:border-error-700 dark:focus:ring-error-700 dark:focus:border-error-700'
              : 'border-gray-300 focus:ring-primary-500 focus:border-primary-500 dark:border-gray-600 dark:focus:ring-primary-500 dark:focus:border-primary-500'
            }
            bg-white dark:bg-dark-card text-primary-900 dark:text-primary-100
            disabled:bg-gray-100 disabled:text-gray-500 disabled:cursor-not-allowed dark:disabled:bg-gray-800
            ${inputClassName}
          `}
          {...props}
        />
        {rightIcon && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
            {rightIcon}
          </div>
        )}
      </div>
      {(helperText || error) && (
        <div className="mt-1 text-xs">
          {helperText && !hasError && (
            <p className={`text-gray-500 dark:text-gray-400 ${helperTextClassName}`}>
              {helperText}
            </p>
          )}
          {hasError && (
            <p className={`text-error-600 dark:text-error-400 ${errorClassName}`}>
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
});

Input.displayName = 'Input';

export default Input;
