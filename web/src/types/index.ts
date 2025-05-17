/**
 * Izvoz vseh tipov
 */

// Izvoz tipov za WebSocket
export * from './websocket';

// Izvoz tipov za strategije
export * from './strategy';

// Izvoz tipov za API
export * from './api';

// Izvoz tipov za uporabnika
export * from './user';

// Izvoz tipov za temo
export * from './theme';

// Časovni razpon
export type TimeRange = '1h' | '24h' | '7d' | '30d' | '90d' | '1y' | 'all';

// Stari tipi, ki jih še vedno uporabljamo
export interface SystemHealthStatus {
  component: string;
  status: 'healthy' | 'warning' | 'critical';
  message?: string;
}

export interface KeyMetric {
  title: string;
  value: string | number;
  change: number;
  unit?: string;
  icon: string;
}

export interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface ActivityData {
  timestamp: string;
  transactions: number;
  profit: number;
}