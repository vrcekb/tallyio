/**
 * Tipi za WebSocket komunikacijo
 */

// Sistemski status
export interface SystemStatusComponent {
  name: string;
  status: string;
  details?: Record<string, unknown>;
}

export interface SystemStatus {
  components: SystemStatusComponent[];
}

// Blockchain status
export interface ChainStatus {
  id: string;
  name: string;
  status: string;
  latency: number;
  blockHeight: number;
  lastUpdate: string;
}

// Strategije
export interface Strategy {
  id: string;
  name: string;
  type: string;
  status: string;
  profit: number;
  transactions: number;
  successRate: number;
}

// Transakcije
export interface Transaction {
  id: string;
  hash: string;
  type: string;
  status: string;
  amount: number;
  profit: number;
  timestamp: string;
  chain: string;
}

// RPC lokacije
export interface RpcLocation {
  id: string;
  name: string;
  location: [number, number];
  status: string;
  latency: number;
}

// Mempool podatki
export interface MempoolData {
  pending: number;
  queued: number;
  transactions: Transaction[];
}

// Tipi sporočil
export enum WebSocketMessageType {
  SYSTEM_STATUS = 'SYSTEM_STATUS',
  CHAIN_STATUS = 'CHAIN_STATUS',
  TRANSACTION = 'TRANSACTION',
  STRATEGY_UPDATE = 'STRATEGY_UPDATE',
  RPC_STATUS = 'RPC_STATUS',
  MEMPOOL_DATA = 'MEMPOOL_DATA',
  AUTH = 'AUTH',
  ERROR = 'ERROR'
}

// Sporočilo
export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType | string;
  data: T;
  timestamp: number;
}
