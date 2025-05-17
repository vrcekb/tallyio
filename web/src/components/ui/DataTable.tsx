import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface Column<T> {
  key: string;
  header: string;
  render?: (item: T) => React.ReactNode;
  sortable?: boolean;
  className?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  keyExtractor: (item: T) => string | number;
  className?: string;
  headerClassName?: string;
  rowClassName?: string;
  cellClassName?: string;
  animate?: boolean;
  emptyMessage?: string;
}

/**
 * Komponenta za prikaz podatkov v obliki tabele
 */
function DataTable<T>({
  data,
  columns,
  keyExtractor,
  className = '',
  headerClassName = '',
  rowClassName = '',
  cellClassName = '',
  animate = true,
  emptyMessage = 'Ni podatkov za prikaz'
}: DataTableProps<T>) {
  const [sortConfig, setSortConfig] = useState<{
    key: string;
    direction: 'asc' | 'desc';
  } | null>(null);

  const handleSort = (key: string) => {
    if (sortConfig && sortConfig.key === key) {
      setSortConfig({
        key,
        direction: sortConfig.direction === 'asc' ? 'desc' : 'asc'
      });
    } else {
      setSortConfig({ key, direction: 'asc' });
    }
  };

  const sortedData = useMemo(() => {
    if (!sortConfig) return data;

    return [...data].sort((a: any, b: any) => {
      if (a[sortConfig.key] < b[sortConfig.key]) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (a[sortConfig.key] > b[sortConfig.key]) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [data, sortConfig]);

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.02
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 10 },
    show: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.2,
        ease: [0.4, 0, 0.2, 1]
      }
    }
  };

  const tableContent = (
    <div className={`overflow-x-auto ${className}`}>
      <table className="min-w-full divide-y divide-gray-200 dark:divide-dark-border">
        <thead className="bg-primary-50/50 dark:bg-dark-background">
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={`
                  px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 
                  uppercase tracking-wider ${column.sortable ? 'cursor-pointer' : ''} ${headerClassName}
                `}
                onClick={() => column.sortable && handleSort(column.key)}
              >
                <div className="flex items-center">
                  <span>{column.header}</span>
                  {column.sortable && sortConfig && sortConfig.key === column.key && (
                    <span className="ml-1">
                      {sortConfig.direction === 'asc' ? (
                        <ChevronUp size={16} />
                      ) : (
                        <ChevronDown size={16} />
                      )}
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-dark-card divide-y divide-gray-200 dark:divide-dark-border">
          {sortedData.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="px-6 py-4 text-center text-gray-500 dark:text-gray-400"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sortedData.map((item) => (
              <tr
                key={keyExtractor(item)}
                className={`
                  hover:bg-primary-50/30 dark:hover:bg-dark-background/30 
                  transition-colors duration-200 ${rowClassName}
                `}
              >
                {columns.map((column) => (
                  <td
                    key={`${keyExtractor(item)}-${column.key}`}
                    className={`px-6 py-4 whitespace-nowrap ${cellClassName} ${column.className || ''}`}
                  >
                    {column.render
                      ? column.render(item)
                      : (item as any)[column.key]}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );

  if (!animate || sortedData.length === 0) {
    return tableContent;
  }

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="min-w-full divide-y divide-gray-200 dark:divide-dark-border">
        <thead className="bg-primary-50/50 dark:bg-dark-background">
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={`
                  px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 
                  uppercase tracking-wider ${column.sortable ? 'cursor-pointer' : ''} ${headerClassName}
                `}
                onClick={() => column.sortable && handleSort(column.key)}
              >
                <div className="flex items-center">
                  <span>{column.header}</span>
                  {column.sortable && sortConfig && sortConfig.key === column.key && (
                    <span className="ml-1">
                      {sortConfig.direction === 'asc' ? (
                        <ChevronUp size={16} />
                      ) : (
                        <ChevronDown size={16} />
                      )}
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <motion.tbody
          variants={container}
          initial="hidden"
          animate="show"
          className="bg-white dark:bg-dark-card divide-y divide-gray-200 dark:divide-dark-border"
        >
          {sortedData.map((item) => (
            <motion.tr
              key={keyExtractor(item)}
              variants={item}
              className={`
                hover:bg-primary-50/30 dark:hover:bg-dark-background/30 
                transition-colors duration-200 ${rowClassName}
              `}
            >
              {columns.map((column) => (
                <td
                  key={`${keyExtractor(item)}-${column.key}`}
                  className={`px-6 py-4 whitespace-nowrap ${cellClassName} ${column.className || ''}`}
                >
                  {column.render
                    ? column.render(item)
                    : (item as any)[column.key]}
                </td>
              ))}
            </motion.tr>
          ))}
        </motion.tbody>
      </table>
    </div>
  );
}

export default DataTable;
