# Hooks

Ta mapa vsebuje vse custom hooks, ki se uporabljajo v aplikaciji. Hooks so funkcije, ki omogočajo uporabo stanja in drugih React funkcionalnosti v funkcijskih komponentah.

## Seznam hooks

### useWorker

Hook za uporabo WebWorker-jev za zahtevne operacije v ozadju.

```typescript
import useWorker, { MessageType } from '../hooks/useWorker';

// V komponenti
const { loading, error, data, execute } = useWorker<any>(
  new URL('../workers/dataProcessor.worker.ts', import.meta.url).href,
  MessageType.PROCESS_TRANSACTIONS
);

// Uporaba
useEffect(() => {
  if (transactions) {
    execute(transactions)
      .then(result => {
        console.log('Processed data:', result);
      })
      .catch(err => {
        console.error('Error processing data:', err);
      });
  }
}, [transactions, execute]);
```

**Parametri**:
- `workerPath`: Pot do WebWorker datoteke
- `messageType`: Tip sporočila, ki ga pošljemo WebWorker-ju

**Vrne**:
- `loading`: Ali se trenutno izvaja operacija
- `error`: Napaka, če je prišlo do napake
- `data`: Podatki, ki jih vrne WebWorker
- `execute`: Funkcija za pošiljanje podatkov WebWorker-ju

### useWebSocket

Hook za uporabo WebSocket povezave za real-time posodobitve.

```typescript
import { useWebSocket } from '../contexts/WebSocketContext';

// V komponenti
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

// Uporaba
useEffect(() => {
  if (!isConnected) {
    connect();
  }
  
  return () => {
    disconnect();
  };
}, [isConnected, connect, disconnect]);
```

**Vrne**:
- `isConnected`: Ali je WebSocket povezava aktivna
- `isAuthenticated`: Ali je uporabnik avtenticiran
- `connect`: Funkcija za vzpostavitev povezave
- `disconnect`: Funkcija za prekinitev povezave
- `send`: Funkcija za pošiljanje sporočil
- `lastMessage`: Zadnje prejeto sporočilo
- `systemStatus`: Stanje sistema
- `chainStatus`: Stanje blockchain omrežij
- `transactions`: Seznam transakcij
- `strategies`: Seznam strategij
- `rpcStatus`: Stanje RPC vozlišč
- `mempoolData`: Podatki o mempoolu

### useTheme

Hook za uporabo teme (svetla, temna, sistemska).

```typescript
import { useTheme } from '../theme/ThemeContext';

// V komponenti
const { theme, mode, setMode, isDark, toggleMode } = useTheme();

// Uporaba
const handleThemeChange = () => {
  toggleMode();
};
```

**Vrne**:
- `theme`: Trenutna tema
- `mode`: Trenutni način teme (light, dark, system)
- `setMode`: Funkcija za nastavitev načina teme
- `isDark`: Ali je trenutno aktivna temna tema
- `toggleMode`: Funkcija za preklop med svetlo in temno temo

## Najboljše prakse

1. **Ločevanje logike**: Hooks omogočajo ločevanje logike od UI komponent.
2. **Ponovna uporaba**: Hooks omogočajo ponovno uporabo logike v različnih komponentah.
3. **Testiranje**: Hooks je lažje testirati kot komponente.
4. **Kompozicija**: Hooks je mogoče sestaviti v kompleksnejše hooks.

## Primer ustvarjanja novega hook-a

```typescript
import { useState, useEffect } from 'react';

function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
  // Stanje za shranjevanje vrednosti
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      // Poskusi pridobiti vrednost iz localStorage
      const item = window.localStorage.getItem(key);
      // Vrni shranjeno vrednost ali initialValue
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(error);
      return initialValue;
    }
  });

  // Funkcija za posodobitev vrednosti
  const setValue = (value: T) => {
    try {
      // Shrani vrednost v stanje
      setStoredValue(value);
      // Shrani vrednost v localStorage
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(error);
    }
  };

  return [storedValue, setValue];
}

export default useLocalStorage;
```

## Uporaba v komponentah

```jsx
import useLocalStorage from '../hooks/useLocalStorage';

const MyComponent = () => {
  const [name, setName] = useLocalStorage('name', '');

  return (
    <div>
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Enter your name"
      />
      <p>Hello, {name || 'Guest'}!</p>
    </div>
  );
};
```
