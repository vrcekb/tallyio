import React from 'react';
import { motion } from 'framer-motion';

interface GridItem {
  id: string | number;
  title: string;
  content: React.ReactNode;
  icon?: React.ReactNode;
  color?: string;
  className?: string;
}

interface GridViewProps {
  items: GridItem[];
  columns?: number;
  gap?: number;
  className?: string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz podatkov v obliki mrežnega pogleda
 */
const GridView: React.FC<GridViewProps> = ({
  items,
  columns = 3,
  gap = 6,
  className = '',
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
    hidden: { opacity: 0, scale: 0.95 },
    show: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.3,
        ease: [0.4, 0, 0.2, 1]
      }
    }
  };

  const gridContent = (
    <div 
      className={`
        grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-${columns} gap-${gap}
        ${className}
      `}
    >
      {items.map((gridItem) => (
        <div 
          key={gridItem.id}
          className={`
            bg-white dark:bg-dark-card rounded-lg border border-primary-100/50 dark:border-dark-border
            transition-all duration-300 hover:shadow-md overflow-hidden
            ${gridItem.className || ''}
          `}
        >
          <div className={`
            px-6 py-4 border-b border-primary-100/50 dark:border-dark-border
            bg-primary-50/20 dark:bg-dark-background/20
          `}>
            <div className="flex items-center">
              {gridItem.icon && (
                <div className={`mr-3 ${gridItem.color || ''}`}>
                  {gridItem.icon}
                </div>
              )}
              <h3 className="font-medium text-primary-900 dark:text-primary-100">{gridItem.title}</h3>
            </div>
          </div>
          <div className="p-6">
            {gridItem.content}
          </div>
        </div>
      ))}
    </div>
  );

  if (!animate) {
    return gridContent;
  }

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className={`
        grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-${columns} gap-${gap}
        ${className}
      `}
    >
      {items.map((gridItem) => (
        <motion.div 
          key={gridItem.id}
          variants={item}
          className={`
            bg-white dark:bg-dark-card rounded-lg border border-primary-100/50 dark:border-dark-border
            transition-all duration-300 hover:shadow-md overflow-hidden
            ${gridItem.className || ''}
          `}
        >
          <div className={`
            px-6 py-4 border-b border-primary-100/50 dark:border-dark-border
            bg-primary-50/20 dark:bg-dark-background/20
          `}>
            <div className="flex items-center">
              {gridItem.icon && (
                <div className={`mr-3 ${gridItem.color || ''}`}>
                  {gridItem.icon}
                </div>
              )}
              <h3 className="font-medium text-primary-900 dark:text-primary-100">{gridItem.title}</h3>
            </div>
          </div>
          <div className="p-6">
            {gridItem.content}
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

export default GridView;
