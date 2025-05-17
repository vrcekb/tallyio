# Komponente

Ta mapa vsebuje vse komponente, ki se uporabljajo v aplikaciji. Komponente so organizirane v podmape glede na njihovo funkcionalnost.

## Struktura map

- **dashboard**: Komponente za nadzorno ploščo (grafi, tabele, paneli)
- **layout**: Komponente za postavitev (glava, stranska vrstica, osnovna postavitev)
- **strategies**: Komponente za upravljanje strategij
- **ui**: Osnovne UI komponente (kartice, izbirniki, gumbi)

## Komponente za nadzorno ploščo

### ActivityChart

Komponenta za prikaz grafa aktivnosti v realnem času.

**Props**:
- `data`: Podatki za prikaz na grafu
- `timeRange`: Časovno obdobje za prikaz podatkov

### ChainStatusPanel

Komponenta za prikaz stanja blockchain omrežij.

**Props**:
- `data`: Podatki o stanju blockchain omrežij

### MetricCard

Komponenta za prikaz ključnih metrik.

**Props**:
- `data`: Podatki o metriki

### RpcWorldMap

Komponenta za prikaz zemljevida RPC vozlišč po svetu.

**Props**:
- `data`: Podatki o RPC vozliščih

### SystemHealthPanel

Komponenta za prikaz zdravja sistema.

**Props**:
- `data`: Podatki o zdravju sistema

### TopStrategiesChart

Komponenta za prikaz najboljših strategij.

**Props**:
- `data`: Podatki o strategijah

### TransactionsTable

Komponenta za prikaz tabele transakcij.

**Props**:
- `data`: Podatki o transakcijah

### VirtualizedTransactionsTable

Optimizirana komponenta za prikaz tabele transakcij z virtualizacijo.

**Props**:
- `data`: Podatki o transakcijah
- `title`: Naslov tabele (privzeto: 'Recent Transactions')
- `maxHeight`: Maksimalna višina tabele (privzeto: 400)
- `itemHeight`: Višina posameznega elementa (privzeto: 60)
- `className`: Dodatni CSS razredi

## Komponente za postavitev

### Header

Komponenta za prikaz glave aplikacije.

**Props**:
- `timeRange`: Trenutno izbrano časovno obdobje
- `onTimeRangeChange`: Funkcija za spremembo časovnega obdobja
- `isDarkMode`: Ali je trenutno aktivna temna tema
- `toggleDarkMode`: Funkcija za preklop med svetlo in temno temo
- `toggleSidebar`: Funkcija za preklop stranske vrstice

### Layout

Osnovna komponenta za postavitev strani.

**Props**:
- `children`: Vsebina strani

### Sidebar

Komponenta za prikaz stranske vrstice z navigacijo.

**Props**:
- `isOpen`: Ali je stranska vrstica odprta
- `onClose`: Funkcija za zapiranje stranske vrstice

## Komponente za strategije

### StrategyAnalysis

Komponenta za prikaz analize strategije.

**Props**:
- `strategy`: Podatki o strategiji
- `className`: Dodatni CSS razredi

## UI komponente

### AccessibilityMenu

Komponenta za dostopnost, ki omogoča prilagajanje uporabniškega vmesnika.

**Props**:
- `className`: Dodatni CSS razredi

### Card

Komponenta za prikaz kartice.

**Props**:
- `title`: Naslov kartice
- `children`: Vsebina kartice
- `className`: Dodatni CSS razredi

### SkipToContent

Komponenta za preskok na vsebino za uporabnike, ki uporabljajo tipkovnico.

**Props**:
- `contentId`: ID elementa, na katerega se preskoči (privzeto: 'main-content')
- `className`: Dodatni CSS razredi

### ThemeSelector

Komponenta za izbiro teme (svetla, temna, sistemska).

**Props**:
- `className`: Dodatni CSS razredi

### TimeRangeSelector

Komponenta za izbiro časovnega obdobja.

**Props**:
- `value`: Trenutno izbrano časovno obdobje
- `onChange`: Funkcija za spremembo časovnega obdobja
- `className`: Dodatni CSS razredi

### VirtualizedList

Komponenta za virtualiziran seznam.

**Props**:
- `data`: Podatki za prikaz v seznamu
- `height`: Višina seznama (privzeto: 400)
- `itemHeight`: Višina posameznega elementa (privzeto: 50)
- `renderItem`: Funkcija za upodabljanje elementa
- `className`: Dodatni CSS razredi
- `itemClassName`: Dodatni CSS razredi za posamezni element
- `overscanCount`: Število elementov, ki se naložijo izven vidnega območja (privzeto: 5)
- `onScroll`: Funkcija, ki se pokliče ob pomikanju
- `scrollToIndex`: Indeks elementa, na katerega se pomakne

### WebSocketStatus

Komponenta za prikaz stanja WebSocket povezave.

**Props**:
- `showText`: Ali naj se prikaže besedilo (privzeto: true)
- `className`: Dodatni CSS razredi

## Uporaba

Primer uporabe komponente:

```jsx
import Card from '../components/ui/Card';

const MyComponent = () => {
  return (
    <Card title="My Card">
      <p>This is the content of my card.</p>
    </Card>
  );
};
```

## Najboljše prakse

1. **Uporaba TypeScript**: Vse komponente so napisane v TypeScript za boljšo tipsko varnost.
2. **Memoizacija**: Uporaba `React.memo` in `useMemo` za preprečevanje nepotrebnih ponovnih renderiranj.
3. **Lazy loading**: Komponente se nalagajo šele, ko so potrebne.
4. **Dostopnost**: Vse komponente so dostopne in uporabljajo ARIA atribute.
5. **Odzivnost**: Vse komponente so odzivne in se prilagajajo različnim velikostim zaslonov.
