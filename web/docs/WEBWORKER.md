# WebWorker

V tem dokumentu je opisana uporaba WebWorker-ja za obdelavo podatkov v ozadju.

## Uvod

WebWorker omogoča izvajanje JavaScript kode v ločeni niti, kar preprečuje blokiranje glavne niti in izboljša odzivnost uporabniškega vmesnika. V naši aplikaciji uporabljamo WebWorker za obdelavo podatkov, kot so transakcije, metrike in strategije.

## Tipi sporočil

WebWorker lahko obdeluje naslednje tipe sporočil:

```typescript
enum MessageType {
  PROCESS_TRANSACTIONS = 'PROCESS_TRANSACTIONS',
  CALCULATE_METRICS = 'CALCULATE_METRICS',
  ANALYZE_STRATEGY = 'ANALYZE_STRATEGY',
  FILTER_DATA = 'FILTER_DATA',
  SORT_DATA = 'SORT_DATA'
}
```

## Vmesniki za sporočila

### WorkerMessage

Vmesnik za sporočilo, ki ga pošljemo WebWorker-ju:

```typescript
interface WorkerMessage<T = unknown> {
  type: MessageType;
  data: T;
  id: string;
}
```

### WorkerResponse

Vmesnik za odgovor, ki ga prejmemo od WebWorker-ja:

```typescript
interface WorkerResponse<T = unknown> {
  type: MessageType;
  data: T | null;
  id: string;
  error?: string;
}
```

## Funkcije za obdelavo podatkov

### processTransactions

Funkcija za obdelavo transakcij:

```typescript
function processTransactions(transactions: Transaction[]): ProcessTransactionsResult {
  // ...
}
```

Vrne rezultat, ki vsebuje:
- Skupni dobiček/izgubo
- Povprečni dobiček/izgubo
- Stopnjo uspešnosti
- Povprečni čas izvajanja
- Najbolj dobičkonosne transakcije
- Nedavne transakcije
- Metrike po omrežjih

### calculateMetrics

Funkcija za izračun metrik:

```typescript
function calculateMetrics(data: MetricsData): MetricsResult {
  // ...
}
```

Vrne rezultat, ki vsebuje:
- Število vrednosti
- Vsoto
- Povprečje
- Minimum
- Maksimum
- Standardni odklon
- Mediano
- 90., 95. in 99. percentil

### analyzeStrategy

Funkcija za analizo strategije:

```typescript
function analyzeStrategy(strategy: Strategy): StrategyAnalysisResult {
  // ...
}
```

Vrne rezultat, ki vsebuje:
- Ime strategije
- Dobiček/izgubo
- Stopnjo uspešnosti
- Število transakcij
- Podatke časovne vrste
- Drseče povprečje

### filterData

Funkcija za filtriranje podatkov:

```typescript
function filterData(data: Record<string, unknown>[], filters: Record<string, unknown>): Record<string, unknown>[] {
  // ...
}
```

Vrne filtrirane podatke glede na podane filtre.

### sortData

Funkcija za razvrščanje podatkov:

```typescript
function sortData(data: Record<string, unknown>[], sortField: string, sortDirection: 'asc' | 'desc'): Record<string, unknown>[] {
  // ...
}
```

Vrne razvrščene podatke glede na podano polje in smer razvrščanja.

## Uporaba WebWorker-ja

### Pošiljanje sporočila WebWorker-ju

```typescript
const worker = new Worker(new URL('../workers/dataProcessor.worker.ts', import.meta.url));

worker.postMessage({
  type: MessageType.PROCESS_TRANSACTIONS,
  data: transactions,
  id: 'unique-id'
});
```

### Prejemanje odgovora od WebWorker-ja

```typescript
worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
  const { type, data, id, error } = event.data;
  
  if (error) {
    console.error(`Error processing ${type}:`, error);
    return;
  }
  
  switch (type) {
    case MessageType.PROCESS_TRANSACTIONS:
      // Obdelaj rezultat
      break;
    // ...
  }
});
```

### Uporaba useWorker hook-a

Za lažjo uporabo WebWorker-ja smo ustvarili hook `useWorker`:

```typescript
const { execute, loading, error } = useWorker<Transaction[], ProcessTransactionsResult>(
  MessageType.PROCESS_TRANSACTIONS
);

// Uporaba
useEffect(() => {
  const fetchData = async () => {
    try {
      const result = await execute(transactions);
      // Obdelaj rezultat
    } catch (error) {
      console.error('Error processing transactions:', error);
    }
  };
  
  fetchData();
}, [transactions, execute]);
```

## Prednosti uporabe WebWorker-ja

1. **Izboljšana odzivnost uporabniškega vmesnika** - Zahtevne operacije se izvajajo v ločeni niti, kar preprečuje blokiranje glavne niti.
2. **Boljša učinkovitost** - Omogoča paralelno izvajanje kode, kar izboljša učinkovitost aplikacije.
3. **Ločena obdelava podatkov** - Ločitev logike za obdelavo podatkov od logike za upodabljanje uporabniškega vmesnika.
4. **Zmanjšana obremenitev glavne niti** - Glavna nit se lahko osredotoči na upodabljanje uporabniškega vmesnika in odzivanje na uporabniške interakcije.

## Omejitve

1. **Ni dostopa do DOM** - WebWorker nima dostopa do DOM, zato ne more neposredno manipulirati z elementi na strani.
2. **Komunikacija preko sporočil** - Komunikacija med glavno nitjo in WebWorker-jem poteka preko sporočil, kar lahko povzroči dodatno režijo.
3. **Serializacija podatkov** - Podatki, ki se pošiljajo med glavno nitjo in WebWorker-jem, morajo biti serializabilni.
4. **Omejitve pri deljenju podatkov** - Podatki se kopirajo med nitmi, kar lahko povzroči težave pri deljenju velikih količin podatkov.
