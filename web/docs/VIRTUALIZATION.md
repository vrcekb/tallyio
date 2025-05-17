# Virtualizacija seznamov

V tem dokumentu je opisana uporaba virtualizacije seznamov za izboljšanje učinkovitosti upodabljanja velikih seznamov.

## Uvod

Virtualizacija seznamov je tehnika, ki omogoča učinkovito upodabljanje velikih seznamov z upodabljanjem samo tistih elementov, ki so trenutno vidni na zaslonu. To zmanjša število DOM elementov in izboljša učinkovitost upodabljanja.

## VirtualizedList komponenta

VirtualizedList je komponenta, ki uporablja `react-window` knjižnico za virtualizacijo seznamov.

### Uporaba

```tsx
import VirtualizedList from '../components/ui/VirtualizedList';

const MyComponent = () => {
  const data = [
    { id: 1, name: 'Item 1' },
    { id: 2, name: 'Item 2' },
    // ...
    { id: 1000, name: 'Item 1000' }
  ];

  const renderItem = (item, index, style) => (
    <div style={style} className="p-4">
      <h3>{item.name}</h3>
    </div>
  );

  return (
    <VirtualizedList
      data={data}
      height={400}
      itemHeight={50}
      renderItem={renderItem}
      className="border rounded"
      itemClassName="hover:bg-gray-100"
      overscanCount={5}
    />
  );
};
```

### Props

| Prop | Tip | Privzeta vrednost | Opis |
|------|-----|-------------------|------|
| `data` | `T[]` | - | Seznam podatkov za prikaz |
| `height` | `number` | `400` | Višina seznama v pikslih |
| `itemHeight` | `number` | `50` | Višina posameznega elementa v pikslih |
| `renderItem` | `(item: T, index: number, style: React.CSSProperties) => React.ReactNode` | - | Funkcija za upodabljanje posameznega elementa |
| `className` | `string` | `''` | CSS razred za seznam |
| `itemClassName` | `string` | `''` | CSS razred za posamezni element |
| `overscanCount` | `number` | `5` | Število elementov, ki se upodabljajo izven vidnega območja |
| `onScroll` | `(params: { scrollOffset: number; scrollDirection: 'forward' | 'backward' }) => void` | - | Funkcija, ki se pokliče ob pomikanju seznama |
| `scrollToIndex` | `number` | - | Indeks elementa, na katerega se seznam pomakne |

### Implementacija

VirtualizedList komponenta uporablja `react-window` knjižnico za virtualizacijo seznamov. Komponenta je optimizirana z uporabo `React.memo` in `useCallback` za preprečevanje nepotrebnega ponovnega upodabljanja.

```tsx
import React, { useRef, useState, useEffect, useCallback, memo } from 'react';
import { FixedSizeList as List, ListChildComponentProps } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';

// ...

// Funkcija za upodabljanje elementa
const Row = memo(({ index, style }: ListChildComponentProps) => {
  const item = data[index];
  return (
    <div
      style={style}
      className={`${itemClassName} ${index % 2 === 0 ? 'bg-white dark:bg-dark-card' : 'bg-gray-50 dark:bg-dark-background'}`}
    >
      {renderItem(item, index, style)}
    </div>
  );
});

// ...

// Memorizirana verzija komponente za boljšo učinkovitost
const VirtualizedList = memo(VirtualizedListComponent) as typeof VirtualizedListComponent;
```

## VirtualizedTransactionsTable komponenta

VirtualizedTransactionsTable je komponenta, ki uporablja VirtualizedList za prikaz transakcij.

### Uporaba

```tsx
import VirtualizedTransactionsTable from '../components/dashboard/VirtualizedTransactionsTable';

const MyComponent = () => {
  const transactions = [
    { id: 1, strategy: 'Strategy 1', network: 'Ethereum', status: 'success', profitLoss: 0.1, timestamp: '2023-01-01T12:00:00Z' },
    { id: 2, strategy: 'Strategy 2', network: 'Polygon', status: 'failed', profitLoss: -0.05, timestamp: '2023-01-02T12:00:00Z' },
    // ...
  ];

  return (
    <VirtualizedTransactionsTable
      data={transactions}
      title="Recent Transactions"
      height={400}
    />
  );
};
```

### Props

| Prop | Tip | Privzeta vrednost | Opis |
|------|-----|-------------------|------|
| `data` | `Transaction[]` | - | Seznam transakcij za prikaz |
| `title` | `string` | `'Transactions'` | Naslov tabele |
| `height` | `number` | `400` | Višina tabele v pikslih |
| `className` | `string` | `''` | CSS razred za tabelo |

### Implementacija

VirtualizedTransactionsTable komponenta uporablja VirtualizedList za prikaz transakcij. Komponenta je optimizirana z uporabo `React.memo` in `useCallback` za preprečevanje nepotrebnega ponovnega upodabljanja.

```tsx
import React, { useState, useMemo, useCallback, memo } from 'react';
import { CheckCircle, AlertCircle, Clock } from 'lucide-react';
import VirtualizedList from '../ui/VirtualizedList';
import Card from '../ui/Card';
import { Transaction } from '../../types';

// ...

// Funkcija za prikaz ikone stanja
const renderStatusIcon = useCallback((status: 'success' | 'pending' | 'failed') => {
  switch (status) {
    case 'success':
      return <CheckCircle size={16} className="text-success-500" />;
    case 'pending':
      return <Clock size={16} className="text-warning-500" />;
    case 'failed':
      return <AlertCircle size={16} className="text-error-500" />;
  }
}, []);

// Funkcija za prikaz elementa
const renderTransaction = useCallback((transaction: Transaction, _index: number, _style: React.CSSProperties) => (
  // ...
), [renderStatusIcon]);

// ...

// Memorizirana verzija komponente za boljšo učinkovitost
export default memo(VirtualizedTransactionsTable);
```

## Prednosti virtualizacije seznamov

1. **Izboljšana učinkovitost upodabljanja** - Upodabljajo se samo elementi, ki so trenutno vidni na zaslonu, kar zmanjša število DOM elementov.
2. **Manjša poraba pomnilnika** - Manjše število DOM elementov pomeni manjšo porabo pomnilnika.
3. **Boljša odzivnost uporabniškega vmesnika** - Manjše število DOM elementov pomeni hitrejše upodabljanje in boljšo odzivnost uporabniškega vmesnika.
4. **Podpora za velike sezname** - Omogoča učinkovito upodabljanje seznamov z več tisoč elementi.

## Omejitve

1. **Fiksna višina elementov** - Elementi morajo imeti fiksno višino, kar lahko omeji fleksibilnost oblikovanja.
2. **Kompleksnost implementacije** - Implementacija virtualizacije seznamov je bolj kompleksna kot implementacija običajnih seznamov.
3. **Težave z animacijami** - Animacije elementov lahko povzročijo težave z virtualizacijo seznamov.
4. **Težave z dostopnostjo** - Virtualizacija seznamov lahko povzroči težave z dostopnostjo, saj elementi, ki niso vidni, niso prisotni v DOM.

## Zaključek

Virtualizacija seznamov je močno orodje za izboljšanje učinkovitosti upodabljanja velikih seznamov. V naši aplikaciji jo uporabljamo za prikaz transakcij, strategij in drugih podatkov, ki lahko vsebujejo veliko število elementov.
