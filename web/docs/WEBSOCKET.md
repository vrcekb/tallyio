# WebSocket

V tem dokumentu je opisana uporaba WebSocket-a za komunikacijo s strežnikom v realnem času.

## Uvod

WebSocket omogoča dvosmerno komunikacijo med odjemalcem in strežnikom v realnem času. V naši aplikaciji uporabljamo WebSocket za prejemanje podatkov o stanju sistema, transakcijah, strategijah in drugih pomembnih informacijah.

## WebSocketService

WebSocketService je razred, ki upravlja z WebSocket povezavo in omogoča pošiljanje in prejemanje sporočil.

### Inicializacija

```typescript
const websocketService = new WebSocketService();
```

### Povezovanje

```typescript
websocketService.connect('wss://example.com/ws', 'auth-token');
```

### Prekinjanje povezave

```typescript
websocketService.disconnect();
```

### Pošiljanje sporočila

```typescript
websocketService.send(WebSocketMessageType.AUTH, { token: 'auth-token' });
```

### Poslušanje dogodkov

```typescript
websocketService.on('message', (message: WebSocketMessage) => {
  // Obdelaj sporočilo
});

websocketService.on('connect', () => {
  console.log('Connected to WebSocket server');
});

websocketService.on('disconnect', () => {
  console.log('Disconnected from WebSocket server');
});

websocketService.on('error', (error: Error) => {
  console.error('WebSocket error:', error);
});
```

## WebSocketContext

WebSocketContext je React kontekst, ki omogoča dostop do WebSocket funkcionalnosti v celotni aplikaciji.

### Uporaba WebSocketContext

```tsx
const App = () => {
  return (
    <WebSocketProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          {/* ... */}
        </Routes>
      </Router>
    </WebSocketProvider>
  );
};
```

### Uporaba WebSocket funkcionalnosti v komponentah

```tsx
const HomePage = () => {
  const { 
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
  } = useWebSocket();

  useEffect(() => {
    if (!isConnected) {
      connect('auth-token');
    }
  }, [isConnected, connect]);

  const handleSendMessage = () => {
    send(WebSocketMessageType.AUTH, { token: 'auth-token' });
  };

  return (
    <div>
      <h1>Home Page</h1>
      <p>Connection status: {isConnected ? 'Connected' : 'Disconnected'}</p>
      <p>Authentication status: {isAuthenticated ? 'Authenticated' : 'Not authenticated'}</p>
      <button onClick={handleSendMessage}>Send message</button>
      <button onClick={disconnect}>Disconnect</button>
      
      {systemStatus && (
        <div>
          <h2>System Status</h2>
          <ul>
            {systemStatus.components.map((component) => (
              <li key={component.name}>
                {component.name}: {component.status}
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {/* ... */}
    </div>
  );
};
```

## Tipi sporočil

WebSocket lahko pošilja in prejema naslednje tipe sporočil:

```typescript
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
```

## Vmesniki za sporočila

### WebSocketMessage

Vmesnik za WebSocket sporočilo:

```typescript
export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType | string;
  data: T;
  timestamp: number;
}
```

## Prednosti uporabe WebSocket-a

1. **Komunikacija v realnem času** - Omogoča prejemanje podatkov v realnem času brez potrebe po periodičnem poizvedovanju.
2. **Dvosmerna komunikacija** - Omogoča pošiljanje in prejemanje sporočil v obe smeri.
3. **Manjša obremenitev omrežja** - Zmanjša količino podatkov, ki se prenašajo po omrežju, saj ni potrebe po periodičnem poizvedovanju.
4. **Boljša uporabniška izkušnja** - Omogoča takojšnje posodabljanje uporabniškega vmesnika ob spremembah na strežniku.

## Omejitve

1. **Potreba po podpori na strežniku** - Strežnik mora podpirati WebSocket protokol.
2. **Težave s požarnimi zidovi in posredniškimi strežniki** - Nekateri požarni zidovi in posredniški strežniki lahko blokirajo WebSocket povezave.
3. **Potreba po obravnavi prekinitev povezave** - WebSocket povezave se lahko prekinejo, zato je potrebno implementirati mehanizme za ponovno vzpostavitev povezave.
4. **Omejena podpora v starejših brskalnikih** - Nekateri starejši brskalniki ne podpirajo WebSocket protokola.

## Obravnava prekinitev povezave

WebSocketService samodejno poskuša ponovno vzpostaviti povezavo v primeru prekinitve. Število poskusov in časovni interval med poskusi sta nastavljiva:

```typescript
const websocketService = new WebSocketService({
  maxReconnectAttempts: 5,
  reconnectInterval: 1000
});
```

## Avtentikacija

WebSocketService podpira avtentikacijo z žetonom:

```typescript
websocketService.connect('wss://example.com/ws', 'auth-token');
```

Žeton se pošlje strežniku ob vzpostavitvi povezave in ob vsaki ponovni vzpostavitvi povezave.

## Obravnava napak

WebSocketService oddaja dogodke ob napakah, ki jih lahko poslušamo:

```typescript
websocketService.on('error', (error: Error) => {
  console.error('WebSocket error:', error);
});
```

## Zaključek

WebSocket je močno orodje za komunikacijo v realnem času, ki omogoča boljšo uporabniško izkušnjo in manjšo obremenitev omrežja. V naši aplikaciji ga uporabljamo za prejemanje podatkov o stanju sistema, transakcijah, strategijah in drugih pomembnih informacijah.
