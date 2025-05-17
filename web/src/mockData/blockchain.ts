export interface NetworkStats {
  name: string;
  blockHeight: number;
  tps: number;
  pendingTx: number;
  gasPrice: number;
  nodeCount: number;
  status: 'online' | 'degraded' | 'offline';
}

export interface NodeMetrics {
  id: string;
  location: string;
  latency: number;
  uptime: number;
  peers: number;
  syncStatus: number;
  lastBlock: number;
}

export interface BlockData {
  hash: string;
  number: number;
  timestamp: string;
  transactions: number;
  gasUsed: number;
  miner: string;
}

export const networkStats: NetworkStats[] = [
  {
    name: 'Ethereum',
    blockHeight: 19234567,
    tps: 15.4,
    pendingTx: 1245,
    gasPrice: 25,
    nodeCount: 8,
    status: 'online'
  },
  {
    name: 'Arbitrum',
    blockHeight: 156789012,
    tps: 127.8,
    pendingTx: 856,
    gasPrice: 0.1,
    nodeCount: 5,
    status: 'online'
  },
  {
    name: 'Optimism',
    blockHeight: 98765432,
    tps: 85.2,
    pendingTx: 542,
    gasPrice: 0.15,
    nodeCount: 4,
    status: 'degraded'
  },
  {
    name: 'Base',
    blockHeight: 45678901,
    tps: 95.6,
    pendingTx: 324,
    gasPrice: 0.08,
    nodeCount: 3,
    status: 'online'
  },
  {
    name: 'Polygon',
    blockHeight: 234567890,
    tps: 156.7,
    pendingTx: 1876,
    gasPrice: 45,
    nodeCount: 6,
    status: 'online'
  }
];

export const nodeMetrics: NodeMetrics[] = [
  {
    id: 'node-1',
    location: 'US East',
    latency: 45,
    uptime: 99.998,
    peers: 85,
    syncStatus: 100,
    lastBlock: 19234567
  },
  {
    id: 'node-2',
    location: 'US West',
    latency: 82,
    uptime: 99.985,
    peers: 76,
    syncStatus: 100,
    lastBlock: 19234567
  },
  {
    id: 'node-3',
    location: 'EU Central',
    latency: 112,
    uptime: 99.942,
    peers: 92,
    syncStatus: 99.98,
    lastBlock: 19234566
  },
  {
    id: 'node-4',
    location: 'Asia Pacific',
    latency: 156,
    uptime: 99.876,
    peers: 68,
    syncStatus: 99.95,
    lastBlock: 19234565
  }
];

export const generateBlockData = (count: number): BlockData[] => {
  const blocks: BlockData[] = [];
  const baseBlock = 19234567;
  
  for (let i = 0; i < count; i++) {
    const blockNumber = baseBlock - i;
    const timestamp = new Date(Date.now() - i * 12000).toISOString();
    
    blocks.push({
      hash: `0x${Math.random().toString(16).slice(2)}${Math.random().toString(16).slice(2)}`,
      number: blockNumber,
      timestamp,
      transactions: Math.floor(Math.random() * 200 + 100),
      gasUsed: Math.floor(Math.random() * 15000000 + 5000000),
      miner: `0x${Math.random().toString(16).slice(2)}${Math.random().toString(16).slice(2)}`
    });
  }
  
  return blocks;
};