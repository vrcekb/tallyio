import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import StrategyItem from './StrategyItem';
import Card from './Card';
import EmptyState from './EmptyState';
import { Zap } from 'lucide-react';

type StrategyStatus = 'active' | 'paused' | 'disabled' | 'error';

interface Strategy {
  id: string | number;
  name: string;
  description?: string;
  status: StrategyStatus;
  profit?: number;
  change?: number;
  transactions?: number;
  networks?: string[];
  enabled?: boolean;
}

interface StrategyListProps {
  strategies: Strategy[];
  title?: string;
  emptyState?: React.ReactNode;
  onStrategyClick?: (strategy: Strategy) => void;
  onStrategyToggle?: (strategy: Strategy, enabled: boolean) => void;
  className?: string;
  itemClassName?: string;
  animate?: boolean;
  showCard?: boolean;
  maxHeight?: number | string;
}

/**
 * Komponenta za prikaz seznama strategij
 */
const StrategyList: React.FC<StrategyListProps> = ({
  strategies,
  title = 'Strategies',
  emptyState,
  onStrategyClick,
  onStrategyToggle,
  className = '',
  itemClassName = '',
  animate = true,
  showCard = true,
  maxHeight = 600
}) => {
  const defaultEmptyState = (
    <EmptyState
      title="No strategies"
      description="There are no strategies to display at this time."
      icon={<Zap size={24} />}
    />
  );

  const handleToggle = (strategy: Strategy, enabled: boolean) => {
    if (onStrategyToggle) {
      onStrategyToggle(strategy, enabled);
    }
  };

  const content = (
    <div className={className}>
      {title && (
        <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100 mb-4">
          {title}
        </h3>
      )}
      
      {strategies.length === 0 ? (
        emptyState || defaultEmptyState
      ) : (
        <div 
          className="space-y-3 overflow-y-auto pr-1"
          style={{ maxHeight }}
        >
          <AnimatePresence>
            {strategies.map((strategy) => (
              <StrategyItem
                key={strategy.id}
                name={strategy.name}
                description={strategy.description}
                status={strategy.status}
                profit={strategy.profit}
                change={strategy.change}
                transactions={strategy.transactions}
                networks={strategy.networks}
                enabled={strategy.enabled}
                onToggle={onStrategyToggle ? (enabled) => handleToggle(strategy, enabled) : undefined}
                onClick={onStrategyClick ? () => onStrategyClick(strategy) : undefined}
                className={itemClassName}
                animate={animate}
              />
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );

  if (!showCard) {
    return content;
  }

  return (
    <Card>
      {content}
    </Card>
  );
};

export default StrategyList;
