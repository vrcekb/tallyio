# Workers

Ta mapa vsebuje vse Web Workers, ki se uporabljajo v aplikaciji. Web Workers omogočajo izvajanje zahtevnih operacij v ozadju, ne da bi blokirali glavno nit.

## Seznam Web Workers

### dataProcessor.worker.ts

Worker za obdelavo podatkov, kot so transakcije, metrike, strategije itd.

**Tipi sporočil**:
- `PROCESS_TRANSACTIONS`: Obdelava transakcij
- `CALCULATE_METRICS`: Izračun metrik
- `ANALYZE_STRATEGY`: Analiza strategije
- `FILTER_DATA`: Filtriranje podatkov
- `SORT_DATA`: Razvrščanje podatkov

**Primer uporabe**:

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

## Ustvarjanje novega Web Worker-ja

Za ustvarjanje novega Web Worker-ja je treba ustvariti novo datoteko v mapi `workers` in jo registrirati v aplikaciji.

```typescript
// workers/myWorker.worker.ts
/* eslint-disable no-restricted-globals */

// Tipi sporočil
enum MessageType {
  DO_SOMETHING = 'DO_SOMETHING',
  DO_SOMETHING_ELSE = 'DO_SOMETHING_ELSE'
}

// Vmesnik za sporočilo
interface WorkerMessage {
  type: MessageType;
  data: any;
  id: string;
}

// Vmesnik za odgovor
interface WorkerResponse {
  type: MessageType;
  data: any;
  id: string;
  error?: string;
}

// Funkcija za izvajanje operacije
function doSomething(data: any) {
  // Simulacija zahtevne operacije
  const startTime = Date.now();
  
  // Izvedi operacijo
  const result = {
    // ...
  };
  
  // Simulacija trajanja zahtevne operacije
  while (Date.now() - startTime < 500) {
    // Simulacija zahtevne operacije
  }
  
  return result;
}

// Funkcija za izvajanje druge operacije
function doSomethingElse(data: any) {
  // Simulacija zahtevne operacije
  const startTime = Date.now();
  
  // Izvedi operacijo
  const result = {
    // ...
  };
  
  // Simulacija trajanja zahtevne operacije
  while (Date.now() - startTime < 300) {
    // Simulacija zahtevne operacije
  }
  
  return result;
}

// Poslušalec za sporočila
self.addEventListener('message', (event: MessageEvent<WorkerMessage>) => {
  const { type, data, id } = event.data;
  
  try {
    let result;
    
    switch (type) {
      case MessageType.DO_SOMETHING:
        result = doSomething(data);
        break;
      case MessageType.DO_SOMETHING_ELSE:
        result = doSomethingElse(data);
        break;
      default:
        throw new Error(`Neznani tip sporočila: ${type}`);
    }
    
    const response: WorkerResponse = {
      type,
      data: result,
      id
    };
    
    self.postMessage(response);
  } catch (error) {
    const response: WorkerResponse = {
      type,
      data: null,
      id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    self.postMessage(response);
  }
});

export {};
```

## Najboljše prakse

1. **Ločevanje odgovornosti**: Vsak Web Worker naj ima jasno določeno odgovornost.
2. **Minimalna komunikacija**: Zmanjšajte količino podatkov, ki se prenašajo med glavno nitjo in Web Worker-jem.
3. **Obravnava napak**: Vedno obravnavajte napake in jih posredujte nazaj glavni niti.
4. **Tipizacija**: Uporabljajte TypeScript za boljšo tipsko varnost.
5. **Testiranje**: Testirajte Web Worker-je neodvisno od UI komponent.

## Omejitve

1. **Ni dostopa do DOM**: Web Worker-ji nimajo dostopa do DOM-a.
2. **Ni dostopa do `window`**: Web Worker-ji nimajo dostopa do `window` objekta.
3. **Ni dostopa do `document`**: Web Worker-ji nimajo dostopa do `document` objekta.
4. **Ni dostopa do `parent`**: Web Worker-ji nimajo dostopa do `parent` objekta.
5. **Ni dostopa do `localStorage`**: Web Worker-ji nimajo dostopa do `localStorage`.
6. **Ni dostopa do `sessionStorage`**: Web Worker-ji nimajo dostopa do `sessionStorage`.

## Komunikacija

Komunikacija med glavno nitjo in Web Worker-jem poteka preko sporočil. Glavna nit pošlje sporočilo Web Worker-ju, ta pa odgovori z rezultatom.

```typescript
// Glavna nit
const worker = new Worker('myWorker.js');

worker.postMessage({
  type: 'DO_SOMETHING',
  data: { /* ... */ },
  id: '123'
});

worker.addEventListener('message', (event) => {
  const { type, data, id, error } = event.data;
  
  if (error) {
    console.error('Error in worker:', error);
    return;
  }
  
  console.log('Result from worker:', data);
});

// Web Worker
self.addEventListener('message', (event) => {
  const { type, data, id } = event.data;
  
  // Izvedi operacijo
  const result = doSomething(data);
  
  // Pošlji odgovor
  self.postMessage({
    type,
    data: result,
    id
  });
});
```
