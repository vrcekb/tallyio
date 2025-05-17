import React, { memo } from 'react';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { Wifi, WifiOff } from 'lucide-react';

interface WebSocketStatusProps {
  showText?: boolean;
  className?: string;
}

/**
 * Komponenta za prikaz stanja WebSocket povezave
 */
const WebSocketStatus: React.FC<WebSocketStatusProps> = ({
  showText = true,
  className = ''
}) => {
  const { isConnected } = useWebSocket();

  return (
    <div
      className={`flex items-center ${className}`}
      title={isConnected ? 'WebSocket povezava je aktivna' : 'WebSocket povezava ni aktivna'}
    >
      {isConnected ? (
        <>
          <Wifi size={16} className="text-success-500" />
          {showText && (
            <span className="ml-1 text-xs text-success-500">Povezano</span>
          )}
        </>
      ) : (
        <>
          <WifiOff size={16} className="text-error-500" />
          {showText && (
            <span className="ml-1 text-xs text-error-500">Ni povezave</span>
          )}
        </>
      )}
    </div>
  );
};

// Memorizirana verzija komponente za boljšo učinkovitost
export default memo(WebSocketStatus);
