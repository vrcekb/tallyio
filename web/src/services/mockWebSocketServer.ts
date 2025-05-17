import { WebSocketMessageType } from '../types/websocket';
import {
  systemHealthStatus,
  chainStatus,
  generateRecentTransactions,
  strategies,
  rpcLocations
} from '../mockData';

/**
 * Mock WebSocket strežnik za testiranje WebSocket funkcionalnosti
 * v razvojnem okolju brez pravega backend strežnika
 */
class MockWebSocketServer {
  private clients: Set<WebSocket> = new Set();
  private intervalIds: number[] = [];
  private isRunning = false;

  /**
   * Zažene mock WebSocket strežnik
   */
  public start(): void {
    if (this.isRunning) {
      console.log('Mock WebSocket server is already running');
      return;
    }

    // Ustvari WebSocket strežnik
    this.createMockServer();
    this.isRunning = true;
    console.log('Mock WebSocket server started');

    // Registriraj mock odjemalca
    this.registerMockClient();
  }

  /**
   * Registrira mock odjemalca
   */
  private registerMockClient(): void {
    // Simuliraj odjemalca
    const mockClient = {
      readyState: WebSocket.OPEN,
      send: (message: string) => {
        console.log('Mock client received message:', message);
      },
      addEventListener: (event: string, listener: EventListener) => {
        console.log(`Mock client added listener for ${event}`);
      }
    } as unknown as WebSocket;

    this.registerClient(mockClient);
  }

  /**
   * Ustavi mock WebSocket strežnik
   */
  public stop(): void {
    this.intervalIds.forEach(clearInterval);
    this.intervalIds = [];
    this.clients.clear();
    this.isRunning = false;
    console.log('Mock WebSocket server stopped');
  }

  /**
   * Ustvari mock WebSocket strežnik
   */
  private createMockServer(): void {
    // Simuliraj pošiljanje posodobitev sistema vsakih 10 sekund
    this.intervalIds.push(
      setInterval(() => {
        this.broadcast(WebSocketMessageType.SYSTEM_STATUS, this.generateSystemStatus());
      }, 10000)
    );

    // Simuliraj pošiljanje posodobitev stanja verige vsakih 15 sekund
    this.intervalIds.push(
      setInterval(() => {
        this.broadcast(WebSocketMessageType.CHAIN_STATUS, this.generateChainStatus());
      }, 15000)
    );

    // Simuliraj pošiljanje novih transakcij vsakih 3-7 sekund
    this.intervalIds.push(
      setInterval(() => {
        this.broadcast(WebSocketMessageType.TRANSACTION, this.generateTransaction());
      }, this.randomInterval(3000, 7000))
    );

    // Simuliraj pošiljanje posodobitev strategij vsakih 20-30 sekund
    this.intervalIds.push(
      setInterval(() => {
        this.broadcast(WebSocketMessageType.STRATEGY_UPDATE, this.generateStrategyUpdate());
      }, this.randomInterval(20000, 30000))
    );

    // Simuliraj pošiljanje posodobitev RPC stanja vsakih 8 sekund
    this.intervalIds.push(
      setInterval(() => {
        this.broadcast(WebSocketMessageType.RPC_STATUS, this.generateRpcStatus());
      }, 8000)
    );

    // Simuliraj pošiljanje podatkov mempoola vsakih 5 sekund
    this.intervalIds.push(
      setInterval(() => {
        this.broadcast(WebSocketMessageType.MEMPOOL_DATA, this.generateMempoolData());
      }, 5000)
    );
  }

  /**
   * Registrira novega WebSocket odjemalca
   */
  public registerClient(client: WebSocket): void {
    this.clients.add(client);
    console.log(`Client registered, total clients: ${this.clients.size}`);

    // Pošlji začetne podatke
    this.sendInitialData(client);

    // Nastavi poslušalca za zaprtje povezave
    client.addEventListener('close', () => {
      this.clients.delete(client);
      console.log(`Client disconnected, total clients: ${this.clients.size}`);
    });
  }

  /**
   * Pošlje začetne podatke novemu odjemalcu
   */
  private sendInitialData(client: WebSocket): void {
    // Pošlji stanje sistema
    this.sendToClient(client, WebSocketMessageType.SYSTEM_STATUS, this.generateSystemStatus());

    // Pošlji stanje verige
    this.sendToClient(client, WebSocketMessageType.CHAIN_STATUS, this.generateChainStatus());

    // Pošlji zadnjih 10 transakcij
    const transactions = generateRecentTransactions(10);
    transactions.forEach(tx => {
      this.sendToClient(client, WebSocketMessageType.TRANSACTION, tx);
    });

    // Pošlji strategije
    strategies.forEach(strategy => {
      this.sendToClient(client, WebSocketMessageType.STRATEGY_UPDATE, strategy);
    });

    // Pošlji RPC stanje
    this.sendToClient(client, WebSocketMessageType.RPC_STATUS, this.generateRpcStatus());

    // Pošlji podatke mempoola
    this.sendToClient(client, WebSocketMessageType.MEMPOOL_DATA, this.generateMempoolData());
  }

  /**
   * Pošlje sporočilo vsem povezanim odjemalcem
   */
  private broadcast(type: WebSocketMessageType, data: unknown): void {
    const message = JSON.stringify({
      type,
      data,
      timestamp: Date.now()
    });

    this.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  }

  /**
   * Pošlje sporočilo določenemu odjemalcu
   */
  private sendToClient(client: WebSocket, type: WebSocketMessageType, data: unknown): void {
    if (client.readyState !== WebSocket.OPEN) {
      return;
    }

    const message = JSON.stringify({
      type,
      data,
      timestamp: Date.now()
    });

    client.send(message);
  }

  /**
   * Generira naključni interval med min in max
   */
  private randomInterval(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min + 1) + min);
  }

  /**
   * Generira podatke o stanju sistema
   */
  private generateSystemStatus(): Record<string, unknown> {
    return {
      components: systemHealthStatus,
      uptime: Math.floor(Date.now() / 1000) - Math.floor(Math.random() * 86400),
      cpu_usage: Math.random() * 30 + 10,
      memory_usage: Math.random() * 40 + 20,
      disk_usage: Math.random() * 30 + 40
    };
  }

  /**
   * Generira podatke o stanju verige
   */
  private generateChainStatus(): Record<string, unknown>[] {
    return chainStatus.map(chain => ({
      ...chain,
      gasPrice: chain.gasPrice * (0.9 + Math.random() * 0.2),
      mempoolSize: chain.mempoolSize * (0.8 + Math.random() * 0.4)
    }));
  }

  /**
   * Generira podatke o transakciji
   */
  private generateTransaction(): Record<string, unknown> {
    return generateRecentTransactions(1)[0];
  }

  /**
   * Generira podatke o posodobitvi strategije
   */
  private generateStrategyUpdate(): Record<string, unknown> {
    const strategyIndex = Math.floor(Math.random() * strategies.length);
    const strategy = { ...strategies[strategyIndex] };

    // Naključno posodobi profit in spremembo
    strategy.profit = strategy.profit * (0.95 + Math.random() * 0.1);
    strategy.change = strategy.change + (Math.random() * 2 - 1);

    return strategy;
  }

  /**
   * Generira podatke o stanju RPC
   */
  private generateRpcStatus(): Record<string, unknown>[] {
    return rpcLocations.map(location => ({
      ...location,
      latency: location.latency * (0.9 + Math.random() * 0.2),
      status: Math.random() > 0.95 ? 'degraded' : location.status
    }));
  }

  /**
   * Generira podatke o mempoolu
   */
  private generateMempoolData(): Record<string, unknown> {
    return {
      total_transactions: Math.floor(Math.random() * 5000 + 10000),
      average_gas_price: Math.floor(Math.random() * 50 + 20),
      highest_gas_price: Math.floor(Math.random() * 200 + 100),
      pending_transactions: Math.floor(Math.random() * 2000 + 5000),
      timestamp: Date.now()
    };
  }
}

// Ustvari singleton instanco
const mockWebSocketServer = new MockWebSocketServer();

export default mockWebSocketServer;
