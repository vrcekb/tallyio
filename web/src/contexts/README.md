# Contexts

Ta mapa vsebuje vse React kontekste, ki se uporabljajo v aplikaciji. Konteksti omogočajo deljenje stanja med različnimi komponentami brez potrebe po posredovanju props-ov skozi vse vmesne komponente.

## Seznam kontekstov

### WebSocketContext

Kontekst za zagotavljanje WebSocket funkcionalnosti vsem komponentam v aplikaciji.

```jsx
import { WebSocketProvider, useWebSocket } from '../contexts/WebSocketContext';

// Ovijanje aplikacije v WebSocketProvider
const App = () => {
  return (
    <WebSocketProvider autoConnect={true} url="ws://localhost:8080/ws">
      <div className="app">
        {/* Vsebina aplikacije */}
      </div>
    </WebSocketProvider>
  );
};

// Uporaba v komponenti
const MyComponent = () => {
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

  return (
    <div>
      <p>Connection status: {isConnected ? 'Connected' : 'Disconnected'}</p>
      <button onClick={connect}>Connect</button>
      <button onClick={disconnect}>Disconnect</button>
      <button onClick={() => send('ping', {})}>Send Ping</button>
    </div>
  );
};
```

**Props za WebSocketProvider**:
- `children`: Vsebina, ki jo ovija provider
- `autoConnect`: Ali naj se WebSocket povezava vzpostavi samodejno (privzeto: true)
- `url`: URL za WebSocket povezavo

**Vrednosti, ki jih zagotavlja useWebSocket hook**:
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

## Najboljše prakse

1. **Ločevanje odgovornosti**: Vsak kontekst naj ima jasno določeno odgovornost.
2. **Optimizacija**: Izogibajte se nepotrebnim ponovnim renderiranjem z uporabo `useMemo` in `useCallback`.
3. **Testiranje**: Testirajte kontekste neodvisno od UI komponent.
4. **Tipizacija**: Uporabljajte TypeScript za boljšo tipsko varnost.

## Primer ustvarjanja novega konteksta

```tsx
// contexts/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Vmesnik za uporabnika
interface User {
  id: string;
  name: string;
  email: string;
  role: string;
}

// Vmesnik za stanje avtentikacije
interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

// Vmesnik za kontekst avtentikacije
interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  resetPassword: (email: string) => Promise<void>;
}

// Privzeto stanje
const defaultState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,
};

// Ustvarjanje konteksta
const AuthContext = createContext<AuthContextType>({
  ...defaultState,
  login: async () => {},
  logout: async () => {},
  register: async () => {},
  resetPassword: async () => {},
});

// Hook za uporabo konteksta
export const useAuth = () => useContext(AuthContext);

// Props za AuthProvider
interface AuthProviderProps {
  children: ReactNode;
}

// AuthProvider komponenta
export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [state, setState] = useState<AuthState>(defaultState);

  // Preveri, ali je uporabnik že prijavljen
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Preveri, ali obstaja token v localStorage
        const token = localStorage.getItem('auth_token');
        
        if (!token) {
          setState({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,
          });
          return;
        }
        
        // Preveri veljavnost tokena
        const response = await fetch('/api/auth/me', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        
        if (!response.ok) {
          throw new Error('Invalid token');
        }
        
        const user = await response.json();
        
        setState({
          user,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });
      } catch (error) {
        // Odstrani neveljaven token
        localStorage.removeItem('auth_token');
        
        setState({
          user: null,
          isAuthenticated: false,
          isLoading: false,
          error: error instanceof Error ? error.message : 'An error occurred',
        });
      }
    };
    
    checkAuth();
  }, []);

  // Funkcija za prijavo
  const login = async (email: string, password: string) => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));
      
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });
      
      if (!response.ok) {
        throw new Error('Invalid credentials');
      }
      
      const { user, token } = await response.json();
      
      // Shrani token v localStorage
      localStorage.setItem('auth_token', token);
      
      setState({
        user,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Funkcija za odjavo
  const logout = async () => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));
      
      // Odstrani token iz localStorage
      localStorage.removeItem('auth_token');
      
      setState({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Funkcija za registracijo
  const register = async (name: string, email: string, password: string) => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));
      
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, email, password }),
      });
      
      if (!response.ok) {
        throw new Error('Registration failed');
      }
      
      const { user, token } = await response.json();
      
      // Shrani token v localStorage
      localStorage.setItem('auth_token', token);
      
      setState({
        user,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Funkcija za ponastavitev gesla
  const resetPassword = async (email: string) => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));
      
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });
      
      if (!response.ok) {
        throw new Error('Password reset failed');
      }
      
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: null,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Vrednost konteksta
  const contextValue: AuthContextType = {
    ...state,
    login,
    logout,
    register,
    resetPassword,
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
```

## Uporaba v aplikaciji

```jsx
// App.tsx
import { AuthProvider } from './contexts/AuthContext';

const App = () => {
  return (
    <AuthProvider>
      <div className="app">
        {/* Vsebina aplikacije */}
      </div>
    </AuthProvider>
  );
};

// LoginPage.tsx
import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

const LoginPage = () => {
  const { login, isLoading, error } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await login(email, password);
  };

  return (
    <div className="login-page">
      <h1>Login</h1>
      {error && <div className="error">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Loading...' : 'Login'}
        </button>
      </form>
    </div>
  );
};
```
