import React from 'react';
import { motion } from 'framer-motion';
import Card from './Card';

interface ChartCardProps {
  title: string;
  subtitle?: string;
  chart: React.ReactNode;
  legend?: React.ReactNode;
  toolbar?: React.ReactNode;
  className?: string;
  height?: number | string;
  animate?: boolean;
}

/**
 * Komponenta za prikaz grafikonov
 */
const ChartCard: React.FC<ChartCardProps> = ({
  title,
  subtitle,
  chart,
  legend,
  toolbar,
  className = '',
  height = 300,
  animate = true
}) => {
  const chartHeight = typeof height === 'number' ? `${height}px` : height;

  const chartContent = (
    <Card className={className}>
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100">{title}</h3>
            {subtitle && (
              <p className="text-sm text-primary-600 dark:text-primary-400 mt-1">{subtitle}</p>
            )}
          </div>
          {toolbar && (
            <div className="flex items-center">
              {toolbar}
            </div>
          )}
        </div>
        
        <div className="flex-1" style={{ height: chartHeight }}>
          {chart}
        </div>
        
        {legend && (
          <div className="mt-4 pt-4 border-t border-primary-100/50 dark:border-dark-border">
            {legend}
          </div>
        )}
      </div>
    </Card>
  );

  if (!animate) {
    return chartContent;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
    >
      {chartContent}
    </motion.div>
  );
};

export default ChartCard;
