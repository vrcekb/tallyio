import { Theme } from '../types';
import defaultTheme from './default';

/**
 * Nordijska tema "Zimsko jutro" za aplikacijo
 * Temelji na privzeti temi, vendar z drugačnimi barvami
 */
const nordicTheme: Theme = {
  ...defaultTheme,
  name: 'nordic',
  colors: {
    ...defaultTheme.colors,
    primary: {
      50: '#F5F7FA', // snežno bela
      100: '#E8EDF2', // jutranja megla
      200: '#D1DBE6',
      300: '#B0C4D8',
      400: '#8FABC9',
      500: '#526D82', // zimsko nebo
      600: '#435A6B',
      700: '#354754',
      800: '#26343D',
      900: '#172026',
    },
    success: {
      50: '#F2F9F5',
      100: '#E5F3EB',
      200: '#CCE7D7',
      300: '#99D0B0',
      400: '#66B888',
      500: '#5D7285', // umirjena sivka
      600: '#4A5B6A',
      700: '#384450',
      800: '#252D35',
      900: '#13171B',
    },
    warning: {
      50: '#FEF8EE',
      100: '#FDF1DD',
      200: '#FBE3BB',
      300: '#F7C778',
      400: '#F4AB34',
      500: '#DDA15E', // zahod sonca
      600: '#B1814B',
      700: '#856138',
      800: '#584025',
      900: '#2C2013',
    },
    error: {
      50: '#FCF2F2',
      100: '#F9E5E5',
      200: '#F3CCCC',
      300: '#E79999',
      400: '#DB6666',
      500: '#BC4749', // skandinavska rdeča
      600: '#96393A',
      700: '#712B2C',
      800: '#4B1D1D',
      900: '#260E0F',
    },
    gray: {
      50: '#F8F9FA',
      100: '#F1F3F5',
      200: '#E9ECF0',
      300: '#DEE2E6',
      400: '#CED4DA',
      500: '#ADB5BD',
      600: '#868E96',
      700: '#495057',
      800: '#343A40',
      900: '#212529',
    },
    background: {
      light: '#F5F7FA', // snežno bela
      dark: '#172026', // temno zimsko nebo
    },
    card: {
      light: '#FFFFFF',
      dark: '#26343D', // temna modrina
    },
    text: {
      primary: {
        light: '#172026', // temno zimsko nebo
        dark: '#F5F7FA', // snežno bela
      },
      secondary: {
        light: '#526D82', // zimsko nebo
        dark: '#9DB2BF', // ledeni odtenek
      },
      muted: {
        light: '#6C757D',
        dark: '#9DB2BF', // ledeni odtenek
      },
    },
    border: {
      light: '#E8EDF2', // jutranja megla
      dark: '#354754', // temna modrina
    },
  },
  borderRadius: {
    ...defaultTheme.borderRadius,
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
  },
  shadows: {
    ...defaultTheme.shadows,
    sm: '0 2px 4px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1)',
    md: '0 4px 6px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px rgba(0, 0, 0, 0.05), 0 4px 6px rgba(0, 0, 0, 0.1)',
    xl: '0 20px 25px rgba(0, 0, 0, 0.05), 0 10px 10px rgba(0, 0, 0, 0.1)',
  },
  typography: {
    ...defaultTheme.typography,
    fontFamily: {
      sans: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
    },
    lineHeight: {
      ...defaultTheme.typography.lineHeight,
      normal: '1.6',
      relaxed: '1.75',
    },
  },
  animations: {
    ...defaultTheme.animations,
    transition: {
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
    },
  },
};

export default nordicTheme;
