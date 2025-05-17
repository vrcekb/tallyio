import React from 'react';
import { motion } from 'framer-motion';

interface CardProps {
  title?: string;
  className?: string;
  children: React.ReactNode;
  glassEffect?: boolean;
  animate?: boolean;
}

export const Card = ({
  title,
  className = '',
  children,
  glassEffect = false,
  animate = true
}: CardProps) => {
  const cardContent = (
    <div className={`
      h-full rounded-lg overflow-hidden transition-all duration-300 flex flex-col
      ${glassEffect ? 'glass-card glass-effect' : 'bg-white dark:bg-dark-card border border-primary-100/50 dark:border-dark-border'}
      hover:border-primary-200/50 dark:hover:border-primary-700/50 shadow-sm hover:shadow-md
      ${className}
    `}>
      {title && (
        <div className={`
          px-6 py-4 border-b border-primary-100/50 dark:border-dark-border
          ${glassEffect ? 'bg-white/5 dark:bg-dark-background/20' : 'bg-primary-50/20 dark:bg-dark-background/20'}
        `}>
          <h3 className="font-medium text-primary-900 dark:text-primary-100">{title}</h3>
        </div>
      )}
      <div className="p-6 flex-1 overflow-hidden">{children}</div>
    </div>
  );

  if (!animate) {
    return cardContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
    >
      {cardContent}
    </motion.div>
  );
};

export default Card;