import { EventEmitter } from '../utils/EventEmitter';
import { WebSocketMessageType, WebSocketMessage } from '../types/websocket';

// Vmesnik za konfiguracijo WebSocket povezave
export interface WebSocketConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

// Privzeta konfiguracija
const DEFAULT_CONFIG: WebSocketConfig = {
  url: 'ws://localhost:8080/ws', // Privzeti URL za WebSocket povezavo
  reconnectInterval: 3000, // 3 sekunde med poskusi ponovne povezave
  maxReconnectAttempts: 5 // Največ 5 poskusov ponovne povezave
};

/**
 * WebSocketService razred za upravljanje WebSocket povezave
 * Uporablja EventEmitter za pošiljanje dogodkov drugim delom aplikacije
 */
class WebSocketService extends EventEmitter {
  private socket: WebSocket | null = null;
  private config: WebSocketConfig;
  private reconnectAttempts = 0;
  private reconnectTimeout: number | null = null;
  private isConnecting = false;
  private messageQueue: WebSocketMessage<unknown>[] = [];
  private isAuthenticated = false;

  constructor(config: Partial<WebSocketConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Vzpostavi povezavo z WebSocket strežnikom
   */
  public connect(authToken?: string): void {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      console.log('WebSocket connection already exists');
      return;
    }

    if (this.isConnecting) {
      console.log('WebSocket connection already in progress');
      return;
    }

    this.isConnecting = true;

    // Dodaj token za avtentikacijo, če je na voljo
    const url = authToken ? `${this.config.url}?token=${authToken}` : this.config.url;

    // Preveri, ali smo v razvojnem okolju in uporabljamo mock
    if (import.meta.env.DEV && url.includes('localhost')) {
      console.log('Using mock WebSocket in development mode');
      // Simuliraj uspešno povezavo po kratkem zamiku
      setTimeout(() => {
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.emit('connected');

        // Simuliraj avtentikacijo
        setTimeout(() => {
          this.isAuthenticated = true;
          this.emit('authenticated');
        }, 100);
      }, 300);
      return;
    }

    try {
      this.socket = new WebSocket(url);

      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onerror = this.handleError.bind(this);
      this.socket.onclose = this.handleClose.bind(this);

      console.log('WebSocket connecting to', url);
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.isConnecting = false;
      this.attemptReconnect();
    }
  }

  /**
   * Zapre WebSocket povezavo
   */
  public disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.isAuthenticated = false;
    console.log('WebSocket disconnected');
  }

  /**
   * Pošlje sporočilo strežniku
   */
  public send(type: WebSocketMessageType | string, data: unknown): boolean {
    // Če smo v razvojnem okolju in uporabljamo mock
    if (import.meta.env.DEV && this.config.url.includes('localhost') && !this.socket) {
      console.log('Mock WebSocket: message sent', { type, data });
      return true;
    }

    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, message queued');
      this.messageQueue.push({
        type: type as WebSocketMessageType,
        data,
        timestamp: Date.now()
      });
      return false;
    }

    try {
      const message = JSON.stringify({
        type,
        data,
        timestamp: Date.now()
      });

      this.socket.send(message);
      return true;
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      return false;
    }
  }

  /**
   * Preveri, ali je WebSocket povezava aktivna
   */
  public isConnected(): boolean {
    // Če smo v razvojnem okolju in uporabljamo mock
    if (import.meta.env.DEV && this.config.url.includes('localhost')) {
      return !this.isConnecting && this.isAuthenticated;
    }
    return this.socket !== null && this.socket.readyState === WebSocket.OPEN;
  }

  /**
   * Obravnava odprtje WebSocket povezave
   */
  private handleOpen(_event: Event): void {
    console.log('WebSocket connection established');
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.emit('connected');

    // Pošlji sporočila, ki so bila v čakalni vrsti
    this.flushMessageQueue();
  }

  /**
   * Obravnava prejeto WebSocket sporočilo
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data) as WebSocketMessage<unknown>;

      // Preveri, ali je sporočilo za avtentikacijo
      if (message.type === 'auth_success') {
        this.isAuthenticated = true;
        this.emit('authenticated');
        return;
      }

      // Posreduj sporočilo kot dogodek
      this.emit(message.type, message.data);
      this.emit('message', message); // Splošni dogodek za vsa sporočila
    } catch (error) {
      console.error('Error parsing WebSocket message:', error, event.data);
    }
  }

  /**
   * Obravnava napako WebSocket povezave
   */
  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.emit('error', event);
  }

  /**
   * Obravnava zaprtje WebSocket povezave
   */
  private handleClose(event: CloseEvent): void {
    console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
    this.socket = null;
    this.isConnecting = false;
    this.isAuthenticated = false;
    this.emit('disconnected', event);

    // Poskusi ponovno vzpostaviti povezavo, če ni bilo namerno zaprto
    if (event.code !== 1000) {
      this.attemptReconnect();
    }
  }

  /**
   * Poskusi ponovno vzpostaviti povezavo
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= (this.config.maxReconnectAttempts || 5)) {
      console.log('Maximum reconnect attempts reached');
      this.emit('reconnect_failed');
      return;
    }

    this.reconnectAttempts++;
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.config.maxReconnectAttempts})...`);

    this.reconnectTimeout = setTimeout(() => {
      this.emit('reconnecting', this.reconnectAttempts);
      this.connect();
    }, this.config.reconnectInterval);
  }

  /**
   * Pošlje vsa sporočila iz čakalne vrste
   */
  private flushMessageQueue(): void {
    if (this.messageQueue.length === 0) {
      return;
    }

    console.log(`Sending ${this.messageQueue.length} queued messages`);

    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message.type, message.data);
      }
    }
  }
}

// Ustvari singleton instanco
const websocketService = new WebSocketService();

export default websocketService;
