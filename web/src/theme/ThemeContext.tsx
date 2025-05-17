import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ThemeMode, ThemeContextType, ThemeName } from './types';
import themes from './themes';

// Ustvarjanje konteksta teme
const ThemeContext = createContext<ThemeContextType>({
  theme: themes.default,
  mode: 'light',
  setMode: () => {},
  themeName: 'default',
  setThemeName: () => {},
  isDark: false,
  toggleMode: () => {},
});

// Hook za uporabo konteksta teme
export const useTheme = () => useContext(ThemeContext);

// Props za ThemeProvider
interface ThemeProviderProps {
  children: ReactNode;
  defaultMode?: ThemeMode;
  defaultTheme?: ThemeName;
}

/**
 * ThemeProvider komponenta za zagotavljanje teme vsem komponentam v aplikaciji
 */
export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  defaultMode = 'light',
  defaultTheme = 'default',
}) => {
  // Stanje za način teme (light, dark, system)
  const [mode, setMode] = useState<ThemeMode>(() => {
    // Poskusi pridobiti shranjeni način iz localStorage
    const savedMode = localStorage.getItem('themeMode') as ThemeMode;
    return savedMode || defaultMode;
  });

  // Stanje za ime teme
  const [themeName, setThemeName] = useState<ThemeName>(() => {
    // Poskusi pridobiti shranjeno ime teme iz localStorage
    const savedTheme = localStorage.getItem('themeName') as ThemeName;
    return savedTheme || defaultTheme;
  });

  // Stanje za dejansko temo (light ali dark)
  const [isDark, setIsDark] = useState<boolean>(false);

  // Učinek za nastavitev teme glede na način
  useEffect(() => {
    // Shrani način v localStorage
    localStorage.setItem('themeMode', mode);

    // Določi, ali naj bo tema temna
    if (mode === 'dark') {
      setIsDark(true);
    } else if (mode === 'light') {
      setIsDark(false);
    } else if (mode === 'system') {
      // Uporabi sistemsko nastavitev
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setIsDark(prefersDark);
    }
  }, [mode]);

  // Učinek za shranjevanje imena teme
  useEffect(() => {
    // Shrani ime teme v localStorage
    localStorage.setItem('themeName', themeName);
  }, [themeName]);

  // Učinek za poslušanje sprememb sistemske teme
  useEffect(() => {
    if (mode !== 'system') return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = (e: MediaQueryListEvent) => {
      setIsDark(e.matches);
    };

    mediaQuery.addEventListener('change', handleChange);

    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, [mode]);

  // Učinek za nastavitev CSS razreda na dokumentu
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  // Funkcija za preklop med svetlo in temno temo
  const toggleMode = () => {
    if (mode === 'light') {
      setMode('dark');
    } else if (mode === 'dark') {
      setMode('light');
    } else {
      // Če je način 'system', preklopi na nasprotno od trenutne sistemske nastavitve
      setMode(isDark ? 'light' : 'dark');
    }
  };

  // Vrednost konteksta
  const contextValue: ThemeContextType = {
    theme: themes[themeName],
    mode,
    setMode,
    themeName,
    setThemeName,
    isDark,
    toggleMode,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeContext;
