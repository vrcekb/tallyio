# Services

Ta mapa vsebuje vse storitve, ki se uporabljajo v aplikaciji. Storitve so razredi ali funkcije, ki zagotavljajo funkcionalnost, ki ni neposredno povezana z UI komponentami.

## Seznam storitev

### WebSocket Service

Storitev za upravljanje WebSocket povezave za real-time posodobitve.

```typescript
import websocketService, { WebSocketMessageType } from '../services/websocket';

// Vzpostavi povezavo
websocketService.connect();

// Pošlji sporočilo
websocketService.send(WebSocketMessageType.SYSTEM_STATUS, { status: 'ok' });

// Preveri, ali je povezava aktivna
const isConnected = websocketService.isConnected();

// Prekini povezavo
websocketService.disconnect();
```

**Funkcije**:
- `connect(authToken?: string)`: Vzpostavi povezavo z WebSocket strežnikom
- `disconnect()`: Prekine povezavo z WebSocket strežnikom
- `send(type: string, data: any)`: Pošlje sporočilo WebSocket strežniku
- `isConnected()`: Preveri, ali je WebSocket povezava aktivna

**Dogodki**:
- `connected`: Sproži se, ko je vzpostavljena povezava
- `authenticated`: Sproži se, ko je uporabnik avtenticiran
- `disconnected`: Sproži se, ko je prekinjena povezava
- `message`: Sproži se, ko je prejeto sporočilo
- `error`: Sproži se, ko pride do napake
- `reconnecting`: Sproži se, ko se poskuša ponovno vzpostaviti povezavo
- `reconnect_failed`: Sproži se, ko ni mogoče ponovno vzpostaviti povezave

### Mock WebSocket Server

Storitev za simulacijo WebSocket strežnika v razvojnem okolju.

```typescript
import mockWebSocketServer from '../services/mockWebSocketServer';

// Zaženi mock WebSocket strežnik
mockWebSocketServer.start();

// Ustavi mock WebSocket strežnik
mockWebSocketServer.stop();
```

**Funkcije**:
- `start()`: Zažene mock WebSocket strežnik
- `stop()`: Ustavi mock WebSocket strežnik
- `registerClient(client: WebSocket)`: Registrira novega WebSocket odjemalca

## Najboljše prakse

1. **Singleton vzorec**: Storitve so implementirane kot singleton instance, kar omogoča deljenje stanja med različnimi deli aplikacije.
2. **Ločevanje odgovornosti**: Vsaka storitev ima jasno določeno odgovornost.
3. **Dogodkovno vodena arhitektura**: Storitve uporabljajo dogodke za komunikacijo z drugimi deli aplikacije.
4. **Testiranje**: Storitve je mogoče testirati neodvisno od UI komponent.

## Primer ustvarjanja nove storitve

```typescript
// api.ts
class ApiService {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  public setToken(token: string): void {
    this.token = token;
  }

  public async get<T>(endpoint: string, params?: Record<string, any>): Promise<T> {
    const url = new URL(`${this.baseUrl}/${endpoint}`);
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value));
      });
    }
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers,
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  }

  public async post<T>(endpoint: string, data: any): Promise<T> {
    const url = `${this.baseUrl}/${endpoint}`;
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  }
}

// Ustvari singleton instanco
const apiService = new ApiService('https://api.example.com');

export default apiService;
```

## Uporaba v komponentah

```jsx
import { useState, useEffect } from 'react';
import apiService from '../services/api';

const UserProfile = ({ userId }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUser = async () => {
      try {
        setLoading(true);
        const userData = await apiService.get(`users/${userId}`);
        setUser(userData);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, [userId]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
};
```
