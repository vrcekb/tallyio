# Tipi v aplikaciji

V tem dokumentu so opisani tipi, ki se uporabljajo v aplikaciji.

## WebSocket tipi

### WebSocketMessageType

Enum za tipe sporočil, ki jih lahko prejmemo od strežnika:

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

### WebSocketMessage

Vmesnik za WebSocket sporočilo:

```typescript
export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType | string;
  data: T;
  timestamp: number;
}
```

### SystemStatus

Vmesnik za sistemski status:

```typescript
export interface SystemStatusComponent {
  name: string;
  status: string;
  details?: Record<string, unknown>;
}

export interface SystemStatus {
  components: SystemStatusComponent[];
}
```

### ChainStatus

Vmesnik za status blockchain omrežja:

```typescript
export interface ChainStatus {
  id: string;
  name: string;
  status: string;
  latency: number;
  blockHeight: number;
  lastUpdate: string;
}
```

### Strategy

Vmesnik za strategijo:

```typescript
export interface Strategy {
  id: string;
  name: string;
  type: string;
  status: string;
  profit: number;
  transactions: number;
  successRate: number;
}
```

### Transaction

Vmesnik za transakcijo:

```typescript
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
```

### RpcLocation

Vmesnik za RPC lokacijo:

```typescript
export interface RpcLocation {
  id: string;
  name: string;
  location: [number, number];
  status: string;
  latency: number;
}
```

### MempoolData

Vmesnik za podatke mempoola:

```typescript
export interface MempoolData {
  pending: number;
  queued: number;
  transactions: Transaction[];
}
```

## Strategije in analize

### StrategyTransaction

Vmesnik za transakcijo strategije:

```typescript
export interface StrategyTransaction {
  id: string;
  timestamp: string;
  status: string;
  profitLoss: number;
  executionTime: number;
}
```

### AnalysisStrategy

Vmesnik za strategijo za analizo:

```typescript
export interface AnalysisStrategy {
  id: string;
  name: string;
  type: string;
  status: string;
  profit: number;
  transactions: StrategyTransaction[];
  successRate: number;
}
```

### TimeSeriesData

Vmesnik za podatke časovne vrste:

```typescript
export interface TimeSeriesData {
  timestamp: string;
  profitLoss: number;
}
```

### MovingAverageData

Vmesnik za podatke drsečega povprečja:

```typescript
export interface MovingAverageData {
  timestamp: string;
  value: number;
}
```

### StrategyAnalysisResult

Vmesnik za rezultat analize strategije:

```typescript
export interface StrategyAnalysisResult {
  name: string;
  profitLoss: number;
  successRate: number;
  transactionCount: number;
  timeSeriesData: TimeSeriesData[];
  movingAverage: MovingAverageData[];
}
```

## API tipi

### ApiResponse

Vmesnik za odgovor API-ja:

```typescript
export interface ApiResponse<T = unknown> {
  data: T | null;
  error: string | null;
  status: number;
}
```

### RequestOptions

Vmesnik za možnosti zahteve:

```typescript
export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  body?: unknown;
  params?: Record<string, string | number | boolean>;
  timeout?: number;
  cache?: RequestCache;
  shouldFail?: boolean;
  errorMessage?: string;
  errorStatus?: number;
  delay?: number;
}
```

### ApiOptions

Vmesnik za možnosti API klica:

```typescript
export interface ApiOptions<T> {
  skip?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (error: string, status: number) => void;
  useMock?: boolean;
}
```

## Uporabnik in avtentikacija

### User

Vmesnik za uporabnika:

```typescript
export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
  avatar?: string;
}
```

### AuthState

Vmesnik za stanje avtentikacije:

```typescript
export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}
```

### AuthContextType

Vmesnik za tip konteksta avtentikacije:

```typescript
export interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  resetPassword: (email: string) => Promise<void>;
  updateProfile: (data: Partial<User>) => Promise<void>;
  clearError: () => void;
}
```

## Tema

### ThemeMode

Tip za način teme:

```typescript
export type ThemeMode = 'light' | 'dark' | 'system';
```

### ThemeColors

Vmesnik za barve teme:

```typescript
export interface ThemeColors {
  primary: {
    50: string;
    100: string;
    // ...
    900: string;
  };
  secondary: {
    50: string;
    100: string;
    // ...
    900: string;
  };
  // ...
}
```

### Theme

Vmesnik za temo:

```typescript
export interface Theme {
  name: string;
  colors: ThemeColors;
  fonts: {
    body: string;
    heading: string;
    mono: string;
  };
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
}
```

### ThemeContextType

Vmesnik za tip konteksta teme:

```typescript
export interface ThemeContextType {
  theme: Theme;
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
  isDark: boolean;
  toggleMode: () => void;
}
```
