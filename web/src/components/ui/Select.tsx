import React, { forwardRef, useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Check } from 'lucide-react';

interface SelectOption {
  value: string;
  label: string;
  icon?: React.ReactNode;
}

interface SelectProps {
  options: SelectOption[];
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  label?: string;
  helperText?: string;
  error?: string;
  disabled?: boolean;
  fullWidth?: boolean;
  className?: string;
  selectClassName?: string;
  labelClassName?: string;
  helperTextClassName?: string;
  errorClassName?: string;
}

/**
 * Komponenta za prikaz izbirnega polja
 */
const Select = forwardRef<HTMLDivElement, SelectProps>(({
  options,
  value,
  onChange,
  placeholder = 'Izberi možnost',
  label,
  helperText,
  error,
  disabled = false,
  fullWidth = false,
  className = '',
  selectClassName = '',
  labelClassName = '',
  helperTextClassName = '',
  errorClassName = '',
}, ref) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedValue, setSelectedValue] = useState(value);
  const selectRef = useRef<HTMLDivElement>(null);
  const hasError = !!error;

  useEffect(() => {
    setSelectedValue(value);
  }, [value]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSelect = (option: SelectOption) => {
    setSelectedValue(option.value);
    onChange && onChange(option.value);
    setIsOpen(false);
  };

  const selectedOption = options.find(option => option.value === selectedValue);

  return (
    <div className={`${fullWidth ? 'w-full' : ''} ${className}`} ref={ref}>
      {label && (
        <label className={`block text-sm font-medium text-primary-900 dark:text-primary-100 mb-1 ${labelClassName}`}>
          {label}
        </label>
      )}
      <div ref={selectRef} className="relative">
        <div
          className={`
            flex items-center justify-between px-4 py-2 border rounded-md shadow-sm cursor-pointer
            ${fullWidth ? 'w-full' : ''}
            ${hasError
              ? 'border-error-500 focus:ring-error-500 focus:border-error-500 dark:border-error-700 dark:focus:ring-error-700 dark:focus:border-error-700'
              : 'border-gray-300 hover:border-primary-500 dark:border-gray-600 dark:hover:border-primary-500'
            }
            ${disabled ? 'bg-gray-100 text-gray-500 cursor-not-allowed dark:bg-gray-800' : 'bg-white dark:bg-dark-card text-primary-900 dark:text-primary-100'}
            ${selectClassName}
          `}
          onClick={() => !disabled && setIsOpen(!isOpen)}
        >
          <div className="flex items-center">
            {selectedOption?.icon && <span className="mr-2">{selectedOption.icon}</span>}
            <span className={!selectedValue ? 'text-gray-500 dark:text-gray-400' : ''}>
              {selectedOption?.label || placeholder}
            </span>
          </div>
          <ChevronDown
            size={18}
            className={`transition-transform duration-200 ${isOpen ? 'transform rotate-180' : ''}`}
          />
        </div>

        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="absolute z-10 w-full mt-1 bg-white dark:bg-dark-card rounded-md shadow-lg border border-gray-200 dark:border-gray-700 max-h-60 overflow-auto"
            >
              {options.map((option) => (
                <div
                  key={option.value}
                  className={`
                    flex items-center px-4 py-2 cursor-pointer
                    ${selectedValue === option.value
                      ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-900 dark:text-primary-100'
                      : 'hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-900 dark:text-gray-100'
                    }
                  `}
                  onClick={() => handleSelect(option)}
                >
                  {option.icon && <span className="mr-2">{option.icon}</span>}
                  <span className="flex-1">{option.label}</span>
                  {selectedValue === option.value && (
                    <Check size={16} className="text-primary-500 dark:text-primary-400" />
                  )}
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
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

Select.displayName = 'Select';

export default Select;
