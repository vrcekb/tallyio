# Theme

Ta mapa vsebuje vse datoteke, povezane s sistemom tem v aplikaciji. Sistem tem omogoča prilagajanje videza aplikacije z uporabo različnih tem.

## Struktura map

- **themes**: Vsebuje definicije tem (privzeta, temna)
- **types.ts**: Definicije tipov za teme
- **ThemeContext.tsx**: Kontekst za zagotavljanje teme vsem komponentam
- **ThemeVariables.tsx**: Komponenta za nastavitev CSS spremenljivk za temo

## Uporaba

### Ovijanje aplikacije v ThemeProvider

```jsx
import { ThemeProvider } from './theme/ThemeContext';
import ThemeVariables from './theme/ThemeVariables';

const App = () => {
  return (
    <ThemeProvider defaultMode="light">
      <ThemeVariables />
      <div className="app">
        {/* Vsebina aplikacije */}
      </div>
    </ThemeProvider>
  );
};
```

### Uporaba teme v komponentah

```jsx
import { useTheme } from '../theme/ThemeContext';

const MyComponent = () => {
  const { theme, mode, setMode, isDark, toggleMode } = useTheme();

  return (
    <div>
      <h1>Current theme: {theme.name}</h1>
      <p>Mode: {mode}</p>
      <p>Is dark: {isDark ? 'Yes' : 'No'}</p>
      <button onClick={() => setMode('light')}>Light</button>
      <button onClick={() => setMode('dark')}>Dark</button>
      <button onClick={() => setMode('system')}>System</button>
      <button onClick={toggleMode}>Toggle</button>
    </div>
  );
};
```

### Uporaba CSS spremenljivk

```css
.my-component {
  background-color: var(--color-background);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-md);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal) var(--easing-ease-in-out);
}

.my-component:hover {
  background-color: var(--color-primary-100);
  box-shadow: var(--shadow-lg);
}
```

## Teme

### Privzeta tema

Privzeta tema je svetla tema, ki se uporablja, ko uporabnik ne izbere druge teme.

```typescript
const defaultTheme: Theme = {
  name: 'default',
  colors: {
    primary: {
      50: '#f0f9ff',
      100: '#e0f2fe',
      200: '#bae6fd',
      300: '#7dd3fc',
      400: '#38bdf8',
      500: '#0ea5e9',
      600: '#0284c7',
      700: '#0369a1',
      800: '#075985',
      900: '#0c4a6e',
    },
    // ...
  },
  // ...
};
```

### Temna tema

Temna tema je tema, ki se uporablja, ko uporabnik izbere temno temo ali ko je sistemska tema temna.

```typescript
const darkTheme: Theme = {
  ...defaultTheme,
  name: 'dark',
  colors: {
    ...defaultTheme.colors,
    background: {
      light: '#f9fafb',
      dark: '#0f172a', // Temnejša barva ozadja
    },
    card: {
      light: '#ffffff',
      dark: '#1e293b', // Temnejša barva kartice
    },
    // ...
  },
};
```

## Ustvarjanje nove teme

Za ustvarjanje nove teme je treba ustvariti novo datoteko v mapi `themes` in jo dodati v `themes/index.ts`.

```typescript
// themes/blue.ts
import { Theme } from '../types';
import defaultTheme from './default';

const blueTheme: Theme = {
  ...defaultTheme,
  name: 'blue',
  colors: {
    ...defaultTheme.colors,
    primary: {
      50: '#eff6ff',
      100: '#dbeafe',
      200: '#bfdbfe',
      300: '#93c5fd',
      400: '#60a5fa',
      500: '#3b82f6',
      600: '#2563eb',
      700: '#1d4ed8',
      800: '#1e40af',
      900: '#1e3a8a',
    },
    // ...
  },
};

export default blueTheme;
```

```typescript
// themes/index.ts
import { Theme } from '../types';
import defaultTheme from './default';
import darkTheme from './dark';
import blueTheme from './blue';

const themes: Record<string, Theme> = {
  default: defaultTheme,
  dark: darkTheme,
  blue: blueTheme,
};

export default themes;
```

## CSS spremenljivke

Komponenta `ThemeVariables` nastavi CSS spremenljivke na podlagi trenutne teme. Te spremenljivke so nato na voljo v celotni aplikaciji.

```typescript
// Primer CSS spremenljivk
root.style.setProperty('--color-primary-500', theme.colors.primary[500]);
root.style.setProperty('--color-background', isDark ? theme.colors.background.dark : theme.colors.background.light);
root.style.setProperty('--border-radius-md', theme.borderRadius.md);
root.style.setProperty('--shadow-md', theme.shadows.md);
root.style.setProperty('--font-family-sans', theme.typography.fontFamily.sans);
root.style.setProperty('--transition-normal', theme.animations.transition.normal);
```

## Najboljše prakse

1. **Uporaba CSS spremenljivk**: Namesto neposredne uporabe vrednosti v CSS uporabljajte CSS spremenljivke.
2. **Ločevanje tem**: Vsaka tema naj bo v svoji datoteki.
3. **Dedovanje tem**: Nove teme naj dedujejo od privzete teme in spremenijo samo tisto, kar je potrebno.
4. **Konsistentnost**: Uporabljajte konsistentne vrednosti za barve, zaobljenost robov, senčenje itd.
5. **Dostopnost**: Zagotovite, da imajo vse teme dovolj kontrasta med besedilom in ozadjem.
