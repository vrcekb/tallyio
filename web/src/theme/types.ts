/**
 * Tipi za modularni sistem tem
 */

// Tip za barvno paleto
export interface ColorPalette {
  50: string;
  100: string;
  200: string;
  300: string;
  400: string;
  500: string;
  600: string;
  700: string;
  800: string;
  900: string;
}

// Tip za barve teme
export interface ThemeColors {
  primary: ColorPalette;
  success: ColorPalette;
  warning: ColorPalette;
  error: ColorPalette;
  gray: ColorPalette;
  background: {
    light: string;
    dark: string;
  };
  card: {
    light: string;
    dark: string;
  };
  text: {
    primary: {
      light: string;
      dark: string;
    };
    secondary: {
      light: string;
      dark: string;
    };
    muted: {
      light: string;
      dark: string;
    };
  };
  border: {
    light: string;
    dark: string;
  };
}

// Tip za zaobljenost robov
export interface BorderRadius {
  none: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  full: string;
}

// Tip za senčenje
export interface Shadows {
  none: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
}

// Tip za tipografijo
export interface Typography {
  fontFamily: {
    sans: string;
    mono: string;
  };
  fontSize: {
    xs: string;
    sm: string;
    base: string;
    lg: string;
    xl: string;
    '2xl': string;
    '3xl': string;
    '4xl': string;
    '5xl': string;
  };
  fontWeight: {
    light: string;
    normal: string;
    medium: string;
    semibold: string;
    bold: string;
  };
  lineHeight: {
    none: string;
    tight: string;
    normal: string;
    relaxed: string;
    loose: string;
  };
}

// Tip za animacije
export interface Animations {
  transition: {
    fast: string;
    normal: string;
    slow: string;
  };
  easing: {
    easeInOut: string;
    easeOut: string;
    easeIn: string;
  };
}

// Tip za razmike
export interface Spacing {
  0: string;
  1: string;
  2: string;
  3: string;
  4: string;
  5: string;
  6: string;
  8: string;
  10: string;
  12: string;
  16: string;
  20: string;
  24: string;
  32: string;
  40: string;
  48: string;
  64: string;
}

// Tip za celotno temo
export interface Theme {
  name: string;
  colors: ThemeColors;
  borderRadius: BorderRadius;
  shadows: Shadows;
  typography: Typography;
  animations: Animations;
  spacing: Spacing;
}

// Tip za način teme
export type ThemeMode = 'light' | 'dark' | 'system';

// Tip za ime teme
export type ThemeName = 'default' | 'dark' | 'nordic';

// Tip za kontekst teme
export interface ThemeContextType {
  theme: Theme;
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
  themeName: ThemeName;
  setThemeName: (name: ThemeName) => void;
  isDark: boolean;
  toggleMode: () => void;
}
