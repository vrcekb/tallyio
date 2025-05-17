import { Theme } from '../types';
import defaultTheme from './default';

/**
 * Temna tema za aplikacijo
 * Temelji na privzeti temi, vendar z drugačnimi barvami
 */
const darkTheme: Theme = {
  ...defaultTheme,
  name: 'dark',
  colors: {
    ...defaultTheme.colors,
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
    background: {
      light: '#f9fafb',
      dark: '#0f172a', // Temnejša barva ozadja
    },
    card: {
      light: '#ffffff',
      dark: '#1e293b', // Temnejša barva kartice
    },
    text: {
      primary: {
        light: '#111827',
        dark: '#f8fafc', // Svetlejša barva besedila
      },
      secondary: {
        light: '#4b5563',
        dark: '#94a3b8', // Svetlejša sekundarna barva besedila
      },
      muted: {
        light: '#6b7280',
        dark: '#64748b', // Svetlejša barva zatemnjenih besedil
      },
    },
    border: {
      light: '#e5e7eb',
      dark: '#334155', // Temnejša barva robov
    },
  },
};

export default darkTheme;
