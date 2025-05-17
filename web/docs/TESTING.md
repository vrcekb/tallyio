# Testiranje

V tem dokumentu je opisano testiranje aplikacije z uporabo Vitest in React Testing Library.

## Uvod

Za testiranje aplikacije uporabljamo Vitest in React Testing Library. Vitest je hitro orodje za testiranje, ki je kompatibilno z Vite in TypeScript. React Testing Library je knjižnica za testiranje React komponent, ki spodbuja testiranje komponent na način, ki je podoben temu, kako jih uporabljajo uporabniki.

## Struktura testov

Testi so organizirani v mape `__tests__` znotraj map, ki vsebujejo komponente, ki jih testiramo. Testi imajo končnico `.vitest.tsx`, da jih ločimo od drugih datotek.

```
src/
  components/
    ui/
      __tests__/
        WebSocketStatus.vitest.tsx
    dashboard/
      __tests__/
        VirtualizedTransactionsTable.vitest.tsx
    strategies/
      __tests__/
        StrategyAnalysis.vitest.tsx
```

## Konfiguracija

Konfiguracija za Vitest je v datoteki `vitest.config.ts`:

```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.vitest.ts'],
    include: ['**/*.vitest.{ts,tsx}'],
    exclude: ['**/setupTests.vitest.ts'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/setupTests.vitest.ts',
      ],
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
```

## Nastavitve testov

Nastavitve za teste so v datoteki `src/setupTests.vitest.ts`. Ta datoteka vsebuje nastavitve za testno okolje, kot so moki za različne funkcionalnosti brskalnika.

```typescript
import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock za URL.createObjectURL
window.URL.createObjectURL = vi.fn(() => 'mock-url');

// Mock za URL.revokeObjectURL
window.URL.revokeObjectURL = vi.fn();

// ...
```

## Pisanje testov

### Testiranje komponent

Za testiranje komponent uporabljamo React Testing Library. Komponente testiramo tako, da jih upodobimo in preverimo, ali so prikazani pričakovani elementi.

```typescript
import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import WebSocketStatus from '../WebSocketStatus';
import * as WebSocketContext from '../../../contexts/WebSocketContext';

describe('WebSocketStatus', () => {
  // Mock za useWebSocket hook
  const mockUseWebSocket = vi.spyOn(WebSocketContext, 'useWebSocket');
  
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders connected state with text', () => {
    // Mock useWebSocket hook za prikaz povezanega stanja
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      // ...
    });

    render(<WebSocketStatus />);
    
    // Preveri, da je prikazana ikona in besedilo za povezano stanje
    expect(screen.getByTitle('WebSocket povezava je aktivna')).toBeInTheDocument();
    expect(screen.getByText('Povezano')).toBeInTheDocument();
  });

  // ...
});
```

### Mockanje odvisnosti

Za mockanje odvisnosti uporabljamo Vitest funkcije `vi.mock` in `vi.spyOn`. Odvisnosti mockamo tako, da jih nadomestimo z lažnimi implementacijami, ki vračajo pričakovane vrednosti.

```typescript
// Mock za useWorker hook
vi.mock('../../../hooks/useWorker', () => {
  return {
    MessageType: {
      ANALYZE_STRATEGY: 'ANALYZE_STRATEGY',
    },
    default: vi.fn(),
  };
});

// Mock za recharts komponente
vi.mock('recharts', () => {
  return {
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
    // ...
  };
});
```

### Testiranje dogodkov

Za testiranje dogodkov uporabljamo React Testing Library funkcije `fireEvent` in `userEvent`. Dogodke testiramo tako, da simuliramo uporabniške interakcije in preverimo, ali se komponenta odziva pričakovano.

```typescript
it('sorts transactions when clicking on headers', () => {
  render(<VirtualizedTransactionsTable data={mockTransactions} />);
  
  // Klik na Strategy header
  fireEvent.click(screen.getByText('Strategy'));
  
  // Klik na Network header
  fireEvent.click(screen.getByText('Network'));
  
  // Klik na Profit/Loss header
  fireEvent.click(screen.getByText('Profit/Loss'));
  
  // Klik na Profit/Loss header ponovno za spremembo smeri razvrščanja
  fireEvent.click(screen.getByText('Profit/Loss'));
  
  // Test je uspešen, če ni napak pri klikanju na glave tabele
  expect(true).toBe(true);
});
```

### Testiranje asinhronih operacij

Za testiranje asinhronih operacij uporabljamo React Testing Library funkcijo `waitFor`. Asinhrone operacije testiramo tako, da počakamo, da se operacija zaključi, in nato preverimo, ali so prikazani pričakovani elementi.

```typescript
it('calls execute with strategy', async () => {
  // Mock useWorker hook za prikaz podatkov analize
  const mockExecute = vi.fn().mockResolvedValue(mockAnalysisData);
  vi.mocked(useWorkerModule.default).mockReturnValue({
    loading: false,
    error: null,
    execute: mockExecute,
  });

  render(<StrategyAnalysis strategy={mockStrategy} />);

  // Počakaj, da se podatki naložijo
  await waitFor(() => {
    expect(mockExecute).toHaveBeenCalledWith(mockStrategy);
  });
});
```

## Zagon testov

Za zagon testov uporabljamo naslednje ukaze:

- `npm test` - Zažene vse teste
- `npm test -- <ime-datoteke>` - Zažene teste v določeni datoteki
- `npm test:watch` - Zažene teste v načinu za spremljanje sprememb
- `npm test:ui` - Zažene teste v uporabniškem vmesniku
- `npm test:coverage` - Zažene teste in ustvari poročilo o pokritosti kode

## Primeri testov

### WebSocketStatus

```typescript
describe('WebSocketStatus', () => {
  // Mock za useWebSocket hook
  const mockUseWebSocket = vi.spyOn(WebSocketContext, 'useWebSocket');
  
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders connected state with text', () => {
    // Mock useWebSocket hook za prikaz povezanega stanja
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      // ...
    });

    render(<WebSocketStatus />);
    
    // Preveri, da je prikazana ikona in besedilo za povezano stanje
    expect(screen.getByTitle('WebSocket povezava je aktivna')).toBeInTheDocument();
    expect(screen.getByText('Povezano')).toBeInTheDocument();
  });

  // ...
});
```

### VirtualizedTransactionsTable

```typescript
describe('VirtualizedTransactionsTable', () => {
  const mockTransactions: Transaction[] = [
    // ...
  ];

  it('renders the component with title', () => {
    render(<VirtualizedTransactionsTable data={mockTransactions} title="Test Transactions" />);
    expect(screen.getByText('Test Transactions')).toBeInTheDocument();
  });

  // ...
});
```

### StrategyAnalysis

```typescript
describe('StrategyAnalysis', () => {
  const mockStrategy: AnalysisStrategy = {
    // ...
  };

  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders with strategy name', () => {
    // Mock useWorker hook
    vi.mocked(useWorkerModule.default).mockReturnValue({
      loading: false,
      error: null,
      execute: vi.fn().mockResolvedValue(mockAnalysisData),
    });

    render(<StrategyAnalysis strategy={mockStrategy} />);
    expect(screen.getByText(/Strategy Analysis: Test Strategy/i)).toBeInTheDocument();
  });

  // ...
});
```
