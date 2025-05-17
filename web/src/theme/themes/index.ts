import { Theme } from '../types';
import defaultTheme from './default';
import darkTheme from './dark';
import nordicTheme from './nordic';

// Slovar vseh tem
const themes: Record<string, Theme> = {
  default: defaultTheme,
  dark: darkTheme,
  nordic: nordicTheme,
};

export default themes;
