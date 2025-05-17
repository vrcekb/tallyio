# TallyIO Web App

Spletna aplikacija za TallyIO, ki omogoča upravljanje in spremljanje DeFi strategij, blockchain omrežij, DEX-ov in protokolov.

## Funkcionalnosti

- **Nadzorna plošča**: Pregled ključnih metrik, aktivnosti, zdravja sistema, strategij in transakcij
- **Blockchain**: Spremljanje stanja blockchain omrežij, RPC vozlišč in transakcij
- **Strategije**: Upravljanje trgovalnih strategij
- **Risk Management**: Upravljanje tveganj
- **Machine Learning**: Strojno učenje in napovedi
- **DEX Management**: Upravljanje decentraliziranih borz
- **Protocol Management**: Upravljanje protokolov
- **Wallet**: Upravljanje denarnic
- **Smart Contracts**: Upravljanje pametnih pogodb

## Tehnologije

- **React**: Knjižnica za gradnjo uporabniških vmesnikov
- **TypeScript**: Nadgradnja JavaScript-a s tipi
- **Tailwind CSS**: Ogrodje za oblikovanje
- **Vite**: Orodje za gradnjo in razvoj
- **React Router**: Knjižnica za usmerjanje
- **Recharts**: Knjižnica za grafe
- **React Window**: Knjižnica za virtualizacijo seznamov
- **Web Workers**: Za zahtevne operacije v ozadju
- **WebSockets**: Za real-time posodobitve

## Struktura projekta

- **src/components**: Komponente, razdeljene v podmape glede na funkcionalnost
  - **dashboard**: Komponente za nadzorno ploščo
  - **layout**: Komponente za postavitev
  - **strategies**: Komponente za upravljanje strategij
  - **ui**: Osnovne UI komponente
- **src/contexts**: React konteksti za deljenje stanja
- **src/hooks**: Custom hooks
- **src/pages**: Strani aplikacije
- **src/services**: Storitve za komunikacijo z API-ji
- **src/theme**: Sistem tem
- **src/types**: TypeScript tipi
- **src/utils**: Pomožne funkcije
- **src/workers**: Web Workers za zahtevne operacije

## Namestitev

```bash
# Namestitev odvisnosti
npm install

# Zagon razvojnega strežnika
npm run dev

# Gradnja za produkcijo
npm run build

# Predogled produkcijske gradnje
npm run preview
```

## Razvoj

### Komponente

Komponente so organizirane v podmape glede na njihovo funkcionalnost. Vse komponente so napisane v TypeScript in uporabljajo funkcijske komponente s hooks.

```tsx
// Primer komponente
import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  disabled = false,
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'primary':
        return 'bg-primary-500 hover:bg-primary-600 text-white';
      case 'secondary':
        return 'bg-gray-200 hover:bg-gray-300 text-gray-800';
      case 'danger':
        return 'bg-error-500 hover:bg-error-600 text-white';
      default:
        return 'bg-primary-500 hover:bg-primary-600 text-white';
    }
  };

  return (
    <button
      className={`px-4 py-2 rounded-lg transition-colors ${getVariantClasses()} ${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      }`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

export default Button;
```

### Hooks

Custom hooks omogočajo ponovno uporabo logike v različnih komponentah.

```tsx
// Primer hook-a
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

### Konteksti

Konteksti omogočajo deljenje stanja med različnimi komponentami.

```tsx
// Primer konteksta
import React, { createContext, useContext, useState, ReactNode } from 'react';

interface ThemeContextType {
  isDark: boolean;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType>({
  isDark: false,
  toggleTheme: () => {},
});

export const useTheme = () => useContext(ThemeContext);

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [isDark, setIsDark] = useState(false);

  const toggleTheme = () => {
    setIsDark(!isDark);
  };

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
```

### Strani

Strani so komponente, ki se uporabljajo za prikaz različnih delov aplikacije.

```tsx
// Primer strani
import React from 'react';
import Layout from '../components/layout/Layout';
import Card from '../components/ui/Card';

const DashboardPage: React.FC = () => {
  return (
    <Layout>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card title="Card 1">
          <p>This is the content of card 1.</p>
        </Card>
        <Card title="Card 2">
          <p>This is the content of card 2.</p>
        </Card>
        <Card title="Card 3">
          <p>This is the content of card 3.</p>
        </Card>
      </div>
    </Layout>
  );
};

export default DashboardPage;
```

## Optimizacije

Aplikacija vključuje številne optimizacije za izboljšanje zmogljivosti:

1. **Lazy loading**: Komponente se nalagajo šele, ko so potrebne
2. **Suspense**: Prikaz nadomestne vsebine med nalaganjem komponent
3. **Memoizacija**: Preprečevanje nepotrebnih ponovnih renderiranj
4. **Predhodno nalaganje**: Strani in komponente se predhodno naložijo ob premiku miške nad povezavo
5. **Predpomnjenje podatkov**: Podatki se predpomnijo za hitrejši dostop
6. **Virtualizacija seznamov**: Optimiziran prikaz velikih seznamov
7. **Web Workers**: Zahtevne operacije se izvajajo v ozadju
8. **WebSockets**: Real-time posodobitve

## Dostopnost

Aplikacija je zasnovana z mislijo na dostopnost:

1. **ARIA atributi**: Vsi elementi imajo ustrezne ARIA atribute
2. **Navigacija s tipkovnico**: Aplikacija je v celoti dostopna s tipkovnico
3. **Visok kontrast**: Podpora za visok kontrast
4. **Povečava besedila**: Možnost povečave besedila
5. **Zmanjšano gibanje**: Podpora za zmanjšano gibanje
6. **Preskok na vsebino**: Možnost preskoka na glavno vsebino

## Teme

Aplikacija podpira svetlo in temno temo:

1. **Svetla tema**: Privzeta tema
2. **Temna tema**: Tema za uporabo v temnih okoljih
3. **Sistemska tema**: Tema, ki sledi sistemskim nastavitvam

## Testiranje

Aplikacija vključuje teste za komponente, hooks in storitve:

1. **Unit testi**: Testi za posamezne komponente, hooks in storitve
2. **Integracijski testi**: Testi za interakcijo med komponentami
3. **E2E testi**: Testi za celotno aplikacijo

## Prispevanje

1. Ustvarite fork repozitorija
2. Ustvarite branch za vašo funkcionalnost (`git checkout -b feature/amazing-feature`)
3. Commitajte vaše spremembe (`git commit -m 'Add some amazing feature'`)
4. Pushajte v branch (`git push origin feature/amazing-feature`)
5. Odprite Pull Request

## Licenca

Ta projekt je licenciran pod MIT licenco - glejte datoteko [LICENSE](LICENSE) za podrobnosti.
