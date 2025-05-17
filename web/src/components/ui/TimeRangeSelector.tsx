import React from 'react';
import { TimeRange } from '../../types';

interface TimeRangeSelectorProps {
  value: TimeRange;
  onChange: (value: TimeRange) => void;
}

const TimeRangeSelector: React.FC<TimeRangeSelectorProps> = ({ value, onChange }) => {
  const options: TimeRange[] = ['1h', '24h', '7d', '30d'];
  
  return (
    <div className="flex items-center bg-primary-50/50 dark:bg-dark-background/50 rounded-lg p-1">
      {options.map((option) => (
        <button
          key={option}
          className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
            value === option
              ? 'bg-white dark:bg-dark-card text-primary-800 dark:text-primary-200 shadow-sm'
              : 'text-primary-600 dark:text-primary-400 hover:text-primary-800 dark:hover:text-primary-200'
          }`}
          onClick={() => onChange(option)}
        >
          {option}
        </button>
      ))}
    </div>
  );
};

export default TimeRangeSelector;