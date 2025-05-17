import { lazy } from 'react';
import { fetchActivityData, fetchRecentTransactions } from './mockData';
import { TimeRange } from '../types';

// Predhodno nalaganje komponent
const preloadedComponents: Record<string, unknown> = {};

// Predhodno nalaganje podatkov
const preloadedData: Record<string, unknown> = {};

/**
 * Predhodno naloži komponento
 * @param path Pot do komponente
 * @param componentName Ime komponente
 */
export const preloadComponent = async (path: string, componentName: string): Promise<unknown> => {
  if (!preloadedComponents[componentName]) {
    try {
      // Dinamično uvozi komponento
      const component = lazy(() => import(/* @vite-ignore */ path));
      preloadedComponents[componentName] = component;
      return component;
    } catch (error) {
      console.error(`Error preloading component ${componentName}:`, error);
    }
  }
  return preloadedComponents[componentName];
};

/**
 * Predhodno naloži podatke za določeno stran
 * @param route Pot do strani
 * @param timeRange Časovno obdobje (za dashboard)
 */
export const preloadRouteData = async (route: string, timeRange: TimeRange = '24h'): Promise<unknown> => {
  // Če so podatki že naloženi, jih ne nalagaj ponovno
  if (preloadedData[route]) {
    return preloadedData[route];
  }

  try {
    let data: Record<string, unknown> = {};

    switch (route) {
      case '/':
      case '/dashboard': {
        // Predhodno naloži podatke za dashboard
        const [activityData, transactions] = await Promise.all([
          fetchActivityData(timeRange),
          fetchRecentTransactions(10)
        ]);
        data = { activityData, transactions };
        break;
      }

      case '/blockchain': {
        // Predhodno naloži podatke za blockchain stran
        const blockchainTransactions = await fetchRecentTransactions(10);
        data = { transactions: blockchainTransactions };
        break;
      }

      // Dodaj druge strani po potrebi
    }

    preloadedData[route] = data;
    return data;
  } catch (error) {
    console.error(`Error preloading data for route ${route}:`, error);
    return null;
  }
};

/**
 * Pridobi predhodno naložene podatke za določeno stran
 * @param route Pot do strani
 */
export const getPreloadedData = (route: string): unknown => {
  return preloadedData[route] || null;
};

/**
 * Pridobi predhodno naloženo komponento
 * @param componentName Ime komponente
 */
export const getPreloadedComponent = (componentName: string): unknown => {
  return preloadedComponents[componentName] || null;
};

/**
 * Predhodno naloži vse komponente in podatke za določeno stran
 * @param route Pot do strani
 */
export const preloadRoute = async (route: string) => {
  // Predhodno naloži podatke
  await preloadRouteData(route);

  // Predhodno naloži komponente glede na stran
  switch (route) {
    case '/':
    case '/dashboard':
      await Promise.all([
        preloadComponent('../components/dashboard/ActivityChart', 'ActivityChart'),
        preloadComponent('../components/dashboard/TopStrategiesChart', 'TopStrategiesChart'),
        preloadComponent('../components/dashboard/TransactionsTable', 'TransactionsTable'),
        preloadComponent('../components/dashboard/ChainStatusPanel', 'ChainStatusPanel'),
        preloadComponent('../components/dashboard/RpcWorldMap', 'RpcWorldMap')
      ]);
      break;

    case '/blockchain':
      await Promise.all([
        preloadComponent('../components/dashboard/ChainStatusPanel', 'ChainStatusPanel'),
        preloadComponent('../components/dashboard/SystemHealthPanel', 'SystemHealthPanel'),
        preloadComponent('../components/dashboard/RpcWorldMap', 'RpcWorldMap'),
        preloadComponent('../components/dashboard/TransactionsTable', 'TransactionsTable')
      ]);
      break;

    // Dodaj druge strani po potrebi
  }
};
