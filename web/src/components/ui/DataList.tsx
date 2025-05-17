import React from 'react';
import { motion } from 'framer-motion';

interface DataItem {
  id: string | number;
  label: string;
  value: string | number | React.ReactNode;
  icon?: React.ReactNode;
  color?: string;
}

interface DataListProps {
  items: DataItem[];
  className?: string;
  itemClassName?: string;
  labelClassName?: string;
  valueClassName?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz podatkov v obliki seznama
 */
const DataList: React.FC<DataListProps> = ({
  items,
  className = '',
  itemClassName = '',
  labelClassName = '',
  valueClassName = '',
  animate = true
}) => {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0 }
  };

  const listContent = (
    <div className={`space-y-3 ${className}`}>
      {items.map((dataItem) => (
        <div 
          key={dataItem.id}
          className={`
            flex items-center justify-between p-3 rounded-lg
            bg-primary-50/50 dark:bg-dark-background
            transition-all duration-300 hover:shadow-md hover:translate-y-[-2px]
            ${itemClassName}
          `}
        >
          <div className="flex items-center">
            {dataItem.icon && (
              <div className={`mr-3 ${dataItem.color || ''}`}>
                {dataItem.icon}
              </div>
            )}
            <span className={`font-medium text-primary-900 dark:text-primary-100 ${labelClassName}`}>
              {dataItem.label}
            </span>
          </div>
          <div className={`text-primary-700 dark:text-primary-300 ${valueClassName}`}>
            {dataItem.value}
          </div>
        </div>
      ))}
    </div>
  );

  if (!animate) {
    return listContent;
  }

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
    >
      {items.map((dataItem) => (
        <motion.div 
          key={dataItem.id}
          variants={item}
          className={`
            flex items-center justify-between p-3 rounded-lg
            bg-primary-50/50 dark:bg-dark-background
            transition-all duration-300 hover:shadow-md hover:translate-y-[-2px]
            ${itemClassName}
          `}
        >
          <div className="flex items-center">
            {dataItem.icon && (
              <div className={`mr-3 ${dataItem.color || ''}`}>
                {dataItem.icon}
              </div>
            )}
            <span className={`font-medium text-primary-900 dark:text-primary-100 ${labelClassName}`}>
              {dataItem.label}
            </span>
          </div>
          <div className={`text-primary-700 dark:text-primary-300 ${valueClassName}`}>
            {dataItem.value}
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

export default DataList;
