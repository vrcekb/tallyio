# Utils

Ta mapa vsebuje vse pomožne funkcije, ki se uporabljajo v aplikaciji. Pomožne funkcije so funkcije, ki zagotavljajo splošno funkcionalnost, ki se uporablja na več mestih v aplikaciji.

## Seznam pomožnih funkcij

### accessibility.ts

Funkcije za izboljšanje dostopnosti aplikacije.

```typescript
import { initAccessibility } from '../utils/accessibility';

// V komponenti
useEffect(() => {
  initAccessibility();
}, []);
```

**Funkcije**:
- `setupKeyboardNavigation()`: Nastavi poslušalca za zaznavanje navigacije s tipkovnico
- `setupReducedMotionListener()`: Nastavi poslušalca za zaznavanje preferenc za zmanjšano gibanje
- `setupHighContrastListener()`: Nastavi poslušalca za zaznavanje preferenc za visok kontrast
- `initAccessibility()`: Inicializira vse funkcije za dostopnost

### mockData.ts

Funkcije za generiranje testnih podatkov.

```typescript
import { fetchActivityData, fetchRecentTransactions } from '../utils/mockData';

// V komponenti
useEffect(() => {
  const fetchData = async () => {
    const [activityData, transactions] = await Promise.all([
      fetchActivityData('24h'),
      fetchRecentTransactions(10)
    ]);
    
    setActivityData(activityData);
    setTransactions(transactions);
  };
  
  fetchData();
}, []);
```

**Funkcije**:
- `fetchActivityData(range: TimeRange)`: Pridobi podatke o aktivnosti za določeno časovno obdobje
- `fetchRecentTransactions(count: number)`: Pridobi zadnjih `count` transakcij

### preloadRoutes.ts

Funkcije za predhodno nalaganje strani in komponent.

```typescript
import { preloadRoute, preloadComponent, preloadRouteData } from '../utils/preloadRoutes';

// V komponenti
useEffect(() => {
  // Predhodno naloži stran
  preloadRoute('/dashboard');
  
  // Predhodno naloži komponento
  preloadComponent('../components/dashboard/ActivityChart', 'ActivityChart');
  
  // Predhodno naloži podatke za stran
  preloadRouteData('/dashboard', '24h');
}, []);
```

**Funkcije**:
- `preloadRoute(route: string)`: Predhodno naloži stran
- `preloadComponent(path: string, componentName: string)`: Predhodno naloži komponento
- `preloadRouteData(route: string, timeRange?: TimeRange)`: Predhodno naloži podatke za stran
- `getPreloadedData(route: string)`: Pridobi predhodno naložene podatke za stran
- `getPreloadedComponent(componentName: string)`: Pridobi predhodno naloženo komponento

## Najboljše prakse

1. **Čiste funkcije**: Pomožne funkcije naj bodo čiste funkcije, ki ne spreminjajo stanja.
2. **Ponovna uporaba**: Pomožne funkcije naj bodo zasnovane za ponovno uporabo na več mestih v aplikaciji.
3. **Testiranje**: Pomožne funkcije naj bodo enostavne za testiranje.
4. **Dokumentacija**: Pomožne funkcije naj bodo dobro dokumentirane.

## Primer ustvarjanja nove pomožne funkcije

```typescript
// utils/formatters.ts

/**
 * Formatira število v valuto
 * @param value Vrednost za formatiranje
 * @param currency Valuta (privzeto: 'USD')
 * @param locale Lokalizacija (privzeto: 'en-US')
 * @returns Formatirana vrednost
 */
export function formatCurrency(
  value: number,
  currency: string = 'USD',
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(value);
}

/**
 * Formatira število z ločili
 * @param value Vrednost za formatiranje
 * @param locale Lokalizacija (privzeto: 'en-US')
 * @returns Formatirana vrednost
 */
export function formatNumber(
  value: number,
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale).format(value);
}

/**
 * Formatira datum
 * @param date Datum za formatiranje
 * @param locale Lokalizacija (privzeto: 'en-US')
 * @param options Možnosti za formatiranje
 * @returns Formatiran datum
 */
export function formatDate(
  date: Date | string | number,
  locale: string = 'en-US',
  options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }
): string {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? new Date(date)
    : date;
  
  return new Intl.DateTimeFormat(locale, options).format(dateObj);
}

/**
 * Formatira čas
 * @param date Datum za formatiranje
 * @param locale Lokalizacija (privzeto: 'en-US')
 * @param options Možnosti za formatiranje
 * @returns Formatiran čas
 */
export function formatTime(
  date: Date | string | number,
  locale: string = 'en-US',
  options: Intl.DateTimeFormatOptions = {
    hour: 'numeric',
    minute: 'numeric',
    second: 'numeric',
  }
): string {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? new Date(date)
    : date;
  
  return new Intl.DateTimeFormat(locale, options).format(dateObj);
}

/**
 * Formatira relativni čas
 * @param date Datum za formatiranje
 * @param locale Lokalizacija (privzeto: 'en-US')
 * @returns Formatiran relativni čas
 */
export function formatRelativeTime(
  date: Date | string | number,
  locale: string = 'en-US'
): string {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? new Date(date)
    : date;
  
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - dateObj.getTime()) / 1000);
  
  if (diffInSeconds < 60) {
    return 'just now';
  }
  
  const diffInMinutes = Math.floor(diffInSeconds / 60);
  
  if (diffInMinutes < 60) {
    return `${diffInMinutes} minute${diffInMinutes === 1 ? '' : 's'} ago`;
  }
  
  const diffInHours = Math.floor(diffInMinutes / 60);
  
  if (diffInHours < 24) {
    return `${diffInHours} hour${diffInHours === 1 ? '' : 's'} ago`;
  }
  
  const diffInDays = Math.floor(diffInHours / 24);
  
  if (diffInDays < 30) {
    return `${diffInDays} day${diffInDays === 1 ? '' : 's'} ago`;
  }
  
  const diffInMonths = Math.floor(diffInDays / 30);
  
  if (diffInMonths < 12) {
    return `${diffInMonths} month${diffInMonths === 1 ? '' : 's'} ago`;
  }
  
  const diffInYears = Math.floor(diffInMonths / 12);
  
  return `${diffInYears} year${diffInYears === 1 ? '' : 's'} ago`;
}
```

## Uporaba v komponentah

```jsx
import { formatCurrency, formatDate, formatRelativeTime } from '../utils/formatters';

const TransactionItem = ({ transaction }) => {
  return (
    <div className="transaction-item">
      <div className="transaction-amount">
        {formatCurrency(transaction.amount)}
      </div>
      <div className="transaction-date">
        {formatDate(transaction.date)}
      </div>
      <div className="transaction-time">
        {formatRelativeTime(transaction.date)}
      </div>
    </div>
  );
};
```
