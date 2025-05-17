import React, { createContext, useContext, useEffect, useState, useCallback, ReactNode } from 'react';
import websocketService from '../services/websocket';
import {
  SystemStatus,
  ChainStatus,
  Strategy,
  Transaction,
  RpcLocation,
  MempoolData,
  WebSocketMessageType,
  WebSocketMessage
} from '../types';

// Vmesnik za stanje WebSocket konteksta
interface WebSocketContextState {
  isConnected: boolean;
  isAuthenticated: boolean;
  connect: (authToken?: string) => void;
  disconnect: () => void;
  send: (type: WebSocketMessageType | string, data: unknown) => boolean;
  lastMessage: WebSocketMessage<unknown> | null;
  systemStatus: SystemStatus | null;
  chainStatus: ChainStatus[] | null;
  transactions: Transaction[];
  strategies: Strategy[];
  rpcStatus: RpcLocation[] | null;
  mempoolData: MempoolData | null;
}

// Privzeto stanje
const defaultState: WebSocketContextState = {
  isConnected: false,
  isAuthenticated: false,
  connect: () => {},
  disconnect: () => {},
  send: () => false,
  lastMessage: null,
  systemStatus: null,
  chainStatus: null,
  transactions: [],
  strategies: [],
  rpcStatus: null,
  mempoolData: null
};

// Ustvarjanje konteksta
const WebSocketContext = createContext<WebSocketContextState>(defaultState);

// Hook za uporabo WebSocket konteksta
export const useWebSocket = () => useContext(WebSocketContext);

// Props za WebSocketProvider
interface WebSocketProviderProps {
  children: ReactNode;
  autoConnect?: boolean;
  url?: string;
}

/**
 * WebSocketProvider komponenta za zagotavljanje WebSocket funkcionalnosti
 * vsem komponentam v aplikaciji
 */
export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  children,
  autoConnect = true,
  url
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage<unknown> | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [chainStatus, setChainStatus] = useState<ChainStatus[] | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [rpcStatus, setRpcStatus] = useState<RpcLocation[] | null>(null);
  const [mempoolData, setMempoolData] = useState<MempoolData | null>(null);

  // Funkcija za povezavo z WebSocket strežnikom
  const connect = useCallback((authToken?: string) => {
    if (url) {
      websocketService.config = { ...websocketService.config, url };
    }
    websocketService.connect(authToken);
  }, [url]);

  // Funkcija za prekinitev povezave z WebSocket strežnikom
  const disconnect = useCallback(() => {
    websocketService.disconnect();
  }, []);

  // Funkcija za pošiljanje sporočil
  const send = useCallback((type: WebSocketMessageType | string, data: unknown) => {
    return websocketService.send(type, data);
  }, []);

  // Nastavitev poslušalcev dogodkov ob montiranju komponente
  useEffect(() => {
    // Poslušalec za povezavo
    const handleConnected = () => {
      setIsConnected(true);
    };

    // Poslušalec za avtentikacijo
    const handleAuthenticated = () => {
      setIsAuthenticated(true);
    };

    // Poslušalec za prekinitev povezave
    const handleDisconnected = () => {
      setIsConnected(false);
      setIsAuthenticated(false);
    };

    // Poslušalec za sporočila
    const handleMessage = (message: WebSocketMessage<unknown>) => {
      setLastMessage(message);
    };

    // Poslušalci za specifične tipe sporočil
    const handleSystemStatus = (data: SystemStatus) => {
      setSystemStatus(data);
    };

    const handleChainStatus = (data: ChainStatus[]) => {
      setChainStatus(data);
    };

    const handleTransaction = (data: Transaction) => {
      setTransactions(prev => [data, ...prev].slice(0, 100)); // Omejimo na 100 transakcij
    };

    const handleStrategyUpdate = (data: Strategy) => {
      setStrategies(prev => {
        const index = prev.findIndex(s => s.id === data.id);
        if (index >= 0) {
          const newStrategies = [...prev];
          newStrategies[index] = data;
          return newStrategies;
        }
        return [data, ...prev];
      });
    };

    const handleRpcStatus = (data: RpcLocation[]) => {
      setRpcStatus(data);
    };

    const handleMempoolData = (data: MempoolData) => {
      setMempoolData(data);
    };

    // Registracija poslušalcev
    websocketService.on('connected', handleConnected);
    websocketService.on('authenticated', handleAuthenticated);
    websocketService.on('disconnected', handleDisconnected);
    websocketService.on('message', handleMessage);
    websocketService.on(WebSocketMessageType.SYSTEM_STATUS, handleSystemStatus);
    websocketService.on(WebSocketMessageType.CHAIN_STATUS, handleChainStatus);
    websocketService.on(WebSocketMessageType.TRANSACTION, handleTransaction);
    websocketService.on(WebSocketMessageType.STRATEGY_UPDATE, handleStrategyUpdate);
    websocketService.on(WebSocketMessageType.RPC_STATUS, handleRpcStatus);
    websocketService.on(WebSocketMessageType.MEMPOOL_DATA, handleMempoolData);

    // Avtomatska povezava, če je nastavljena
    if (autoConnect) {
      connect();
    }

    // Čiščenje poslušalcev ob odmontiranju komponente
    return () => {
      websocketService.removeListener('connected', handleConnected);
      websocketService.removeListener('authenticated', handleAuthenticated);
      websocketService.removeListener('disconnected', handleDisconnected);
      websocketService.removeListener('message', handleMessage);
      websocketService.removeListener(WebSocketMessageType.SYSTEM_STATUS, handleSystemStatus);
      websocketService.removeListener(WebSocketMessageType.CHAIN_STATUS, handleChainStatus);
      websocketService.removeListener(WebSocketMessageType.TRANSACTION, handleTransaction);
      websocketService.removeListener(WebSocketMessageType.STRATEGY_UPDATE, handleStrategyUpdate);
      websocketService.removeListener(WebSocketMessageType.RPC_STATUS, handleRpcStatus);
      websocketService.removeListener(WebSocketMessageType.MEMPOOL_DATA, handleMempoolData);
    };
  }, [autoConnect, connect]);

  // Vrednost konteksta
  const contextValue: WebSocketContextState = {
    isConnected,
    isAuthenticated,
    connect,
    disconnect,
    send,
    lastMessage,
    systemStatus,
    chainStatus,
    transactions,
    strategies,
    rpcStatus,
    mempoolData
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

export default WebSocketContext;
