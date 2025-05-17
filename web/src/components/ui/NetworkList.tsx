import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import NetworkItem from './NetworkItem';
import Card from './Card';
import EmptyState from './EmptyState';
import { Globe } from 'lucide-react';

type NetworkStatus = 'online' | 'degraded' | 'offline';

interface Network {
  id: string | number;
  name: string;
  status: NetworkStatus;
  icon?: React.ReactNode;
  gasPrice?: number;
  gasTrend?: number;
  blockHeight?: number;
  connections?: number;
  latency?: number;
  mempoolSize?: number;
}

interface NetworkListProps {
  networks: Network[];
  title?: string;
  emptyState?: React.ReactNode;
  onNetworkClick?: (network: Network) => void;
  className?: string;
  itemClassName?: string;
  animate?: boolean;
  showCard?: boolean;
  maxHeight?: number | string;
}

/**
 * Komponenta za prikaz seznama omrežij
 */
const NetworkList: React.FC<NetworkListProps> = ({
  networks,
  title = 'Networks',
  emptyState,
  onNetworkClick,
  className = '',
  itemClassName = '',
  animate = true,
  showCard = true,
  maxHeight = 600
}) => {
  const defaultEmptyState = (
    <EmptyState
      title="No networks"
      description="There are no networks to display at this time."
      icon={<Globe size={24} />}
    />
  );

  const content = (
    <div className={className}>
      {title && (
        <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100 mb-4">
          {title}
        </h3>
      )}
      
      {networks.length === 0 ? (
        emptyState || defaultEmptyState
      ) : (
        <div 
          className="space-y-3 overflow-y-auto pr-1"
          style={{ maxHeight }}
        >
          <AnimatePresence>
            {networks.map((network) => (
              <NetworkItem
                key={network.id}
                name={network.name}
                status={network.status}
                icon={network.icon}
                gasPrice={network.gasPrice}
                gasTrend={network.gasTrend}
                blockHeight={network.blockHeight}
                connections={network.connections}
                latency={network.latency}
                mempoolSize={network.mempoolSize}
                onClick={onNetworkClick ? () => onNetworkClick(network) : undefined}
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

export default NetworkList;
