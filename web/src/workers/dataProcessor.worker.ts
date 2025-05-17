// WebWorker datoteka

// Definicije tipov za podatke
interface Transaction {
  id: string;
  timestamp: string;
  status: string;
  profitLoss: number;
  executionTime?: number;
  network?: string;
}

interface Strategy {
  id: string;
  name: string;
  type: string;
  status: string;
  profit: number;
  transactions: Transaction[];
  successRate: number;
}

interface TimeSeriesData {
  timestamp: string;
  profitLoss: number;
}

interface MovingAverageData {
  timestamp: string;
  value: number;
}

interface NetworkMetric {
  network: string;
  count: number;
  profit: number;
  successRate: number;
}

interface ProcessTransactionsResult {
  totalProfitLoss: number;
  avgProfitLoss: number;
  successRate: number;
  avgExecutionTime: number;
  topProfitable: Transaction[];
  recent: Transaction[];
  networkMetrics: NetworkMetric[];
}

interface MetricsData {
  values: number[];
}

interface MetricsResult {
  count: number;
  sum: number;
  avg: number;
  min: number;
  max: number;
  stdDev: number;
  median: number;
  percentile90: number;
  percentile95: number;
  percentile99: number;
}

interface StrategyAnalysisResult {
  name: string;
  profitLoss: number;
  successRate: number;
  transactionCount: number;
  timeSeriesData: TimeSeriesData[];
  movingAverage: MovingAverageData[];
}

interface FilterDataParams {
  items: Record<string, unknown>[];
  filters: Record<string, unknown>;
}

interface SortDataParams {
  items: Record<string, unknown>[];
  sortField: string;
  sortDirection: 'asc' | 'desc';
}

// Tipi sporočil, ki jih lahko prejme WebWorker
enum MessageType {
  PROCESS_TRANSACTIONS = 'PROCESS_TRANSACTIONS',
  CALCULATE_METRICS = 'CALCULATE_METRICS',
  ANALYZE_STRATEGY = 'ANALYZE_STRATEGY',
  FILTER_DATA = 'FILTER_DATA',
  SORT_DATA = 'SORT_DATA'
}

// Vmesnik za sporočilo z generičnim tipom za podatke
interface WorkerMessage<T = unknown> {
  type: MessageType;
  data: T;
  id: string;
}

// Vmesnik za odgovor z generičnim tipom za podatke
interface WorkerResponse<T = unknown> {
  type: MessageType;
  data: T | null;
  id: string;
  error?: string;
}

/**
 * Obdelava transakcij
 * Izračuna različne metrike za transakcije
 */
function processTransactions(transactions: Transaction[]): ProcessTransactionsResult {
  // Simulacija zahtevne operacije
  const startTime = Date.now();

  // Izračun skupnega dobička/izgube
  const totalProfitLoss = transactions.reduce((sum, tx) => sum + (tx.profitLoss || 0), 0);

  // Izračun povprečnega dobička/izgube
  const avgProfitLoss = transactions.length > 0 ? totalProfitLoss / transactions.length : 0;

  // Izračun uspešnosti transakcij
  const successCount = transactions.filter(tx => tx.status === 'success').length;
  const successRate = transactions.length > 0 ? (successCount / transactions.length) * 100 : 0;

  // Izračun povprečnega časa izvajanja
  const executionTimes = transactions
    .filter(tx => tx.executionTime)
    .map(tx => tx.executionTime);
  const avgExecutionTime = executionTimes.length > 0
    ? executionTimes.reduce((sum, time) => sum + time, 0) / executionTimes.length
    : 0;

  // Razvrščanje transakcij po dobičku/izgubi
  const sortedByProfit = [...transactions].sort((a, b) => (b.profitLoss || 0) - (a.profitLoss || 0));
  const topProfitable = sortedByProfit.slice(0, 5);

  // Razvrščanje transakcij po času
  const sortedByTime = [...transactions].sort((a, b) =>
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
  const recent = sortedByTime.slice(0, 10);

  // Grupiranje po omrežju
  const networkGroups = transactions.reduce((groups, tx) => {
    const network = tx.network || 'unknown';
    if (!groups[network]) {
      groups[network] = [];
    }
    groups[network].push(tx);
    return groups;
  }, {} as Record<string, Transaction[]>);

  // Izračun metrik po omrežju
  const networkMetrics = Object.entries(networkGroups).map(([network, txs]) => {
    const networkProfit = txs.reduce((sum, tx) => sum + (tx.profitLoss || 0), 0);
    const networkSuccessRate = txs.length > 0
      ? (txs.filter(tx => tx.status === 'success').length / txs.length) * 100
      : 0;

    return {
      network,
      count: txs.length,
      profit: networkProfit,
      successRate: networkSuccessRate
    };
  });

  // Simulacija trajanja zahtevne operacije
  while (Date.now() - startTime < 500) {
    // Simulacija zahtevne operacije
  }

  return {
    totalProfitLoss,
    avgProfitLoss,
    successRate,
    avgExecutionTime,
    topProfitable,
    recent,
    networkMetrics
  };
}

/**
 * Izračun metrik
 * Izračuna različne metrike za podatke
 */
function calculateMetrics(data: MetricsData): MetricsResult {
  // Simulacija zahtevne operacije
  const startTime = Date.now();

  // Izračun osnovnih statistik
  const values = data.values || [];
  const count = values.length;
  const sum = values.reduce((acc: number, val: number) => acc + val, 0);
  const avg = count > 0 ? sum / count : 0;
  const min = count > 0 ? Math.min(...values) : 0;
  const max = count > 0 ? Math.max(...values) : 0;

  // Izračun standardne deviacije
  const variance = count > 0
    ? values.reduce((acc: number, val: number) => acc + Math.pow(val - avg, 2), 0) / count
    : 0;
  const stdDev = Math.sqrt(variance);

  // Izračun mediane
  const sorted = [...values].sort((a, b) => a - b);
  const median = count > 0
    ? count % 2 === 0
      ? (sorted[count / 2 - 1] + sorted[count / 2]) / 2
      : sorted[Math.floor(count / 2)]
    : 0;

  // Izračun percentil
  const percentile90 = count > 0
    ? sorted[Math.floor(count * 0.9)]
    : 0;
  const percentile95 = count > 0
    ? sorted[Math.floor(count * 0.95)]
    : 0;
  const percentile99 = count > 0
    ? sorted[Math.floor(count * 0.99)]
    : 0;

  // Simulacija trajanja zahtevne operacije
  while (Date.now() - startTime < 300) {
    // Simulacija zahtevne operacije
  }

  return {
    count,
    sum,
    avg,
    min,
    max,
    stdDev,
    median,
    percentile90,
    percentile95,
    percentile99
  };
}

/**
 * Analiza strategije
 * Analizira uspešnost strategije
 */
function analyzeStrategy(strategy: Strategy): StrategyAnalysisResult {
  // Simulacija zahtevne operacije
  const startTime = Date.now();

  const transactions = strategy.transactions || [];
  const profitLoss = transactions.reduce((sum: number, tx: Transaction) => sum + (tx.profitLoss || 0), 0);
  const successRate = transactions.length > 0
    ? (transactions.filter((tx: Transaction) => tx.status === 'success').length / transactions.length) * 100
    : 0;

  // Izračun trendov
  const timeSeriesData = transactions
    .sort((a: Transaction, b: Transaction) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    .map((tx: Transaction) => ({
      timestamp: tx.timestamp,
      profitLoss: tx.profitLoss || 0
    }));

  // Izračun drsečega povprečja
  const windowSize = 5;
  const movingAverage = [];

  for (let i = 0; i < timeSeriesData.length; i++) {
    const window = timeSeriesData.slice(Math.max(0, i - windowSize + 1), i + 1);
    const sum = window.reduce((acc, val) => acc + val.profitLoss, 0);
    const avg = sum / window.length;

    movingAverage.push({
      timestamp: timeSeriesData[i].timestamp,
      value: avg
    });
  }

  // Simulacija trajanja zahtevne operacije
  while (Date.now() - startTime < 400) {
    // Simulacija zahtevne operacije
  }

  return {
    name: strategy.name,
    profitLoss,
    successRate,
    transactionCount: transactions.length,
    timeSeriesData,
    movingAverage
  };
}

/**
 * Filtriranje podatkov
 * Filtrira podatke glede na kriterije
 */
function filterData(data: Record<string, unknown>[], filters: Record<string, unknown>): Record<string, unknown>[] {
  // Simulacija zahtevne operacije
  const startTime = Date.now();

  const filteredData = data.filter(item => {
    // Preveri vse filtre
    for (const [key, value] of Object.entries(filters)) {
      if (value === undefined || value === null) continue;

      // Preveri, ali ima element to lastnost
      if (!(key in item)) continue;

      // Preveri, ali se vrednost ujema
      if (typeof value === 'string') {
        // Primerjava nizov
        if (!item[key].toString().toLowerCase().includes(value.toLowerCase())) {
          return false;
        }
      } else if (Array.isArray(value)) {
        // Preveri, ali je vrednost v seznamu
        if (!value.includes(item[key])) {
          return false;
        }
      } else if (typeof value === 'object') {
        // Preveri razpon
        if (value.min !== undefined && item[key] < value.min) {
          return false;
        }
        if (value.max !== undefined && item[key] > value.max) {
          return false;
        }
      } else {
        // Primerjava drugih tipov
        if (item[key] !== value) {
          return false;
        }
      }
    }

    return true;
  });

  // Simulacija trajanja zahtevne operacije
  while (Date.now() - startTime < 200) {
    // Simulacija zahtevne operacije
  }

  return filteredData;
}

/**
 * Razvrščanje podatkov
 * Razvrsti podatke glede na kriterije
 */
function sortData(data: Record<string, unknown>[], sortField: string, sortDirection: 'asc' | 'desc'): Record<string, unknown>[] {
  // Simulacija zahtevne operacije
  const startTime = Date.now();

  const sortedData = [...data].sort((a, b) => {
    const aValue = a[sortField];
    const bValue = b[sortField];

    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
    } else if (typeof aValue === 'string' && typeof bValue === 'string') {
      return sortDirection === 'asc'
        ? aValue.localeCompare(bValue)
        : bValue.localeCompare(aValue);
    } else if (aValue instanceof Date && bValue instanceof Date) {
      return sortDirection === 'asc'
        ? aValue.getTime() - bValue.getTime()
        : bValue.getTime() - aValue.getTime();
    } else {
      // Poskusi pretvoriti v nize
      const aStr = String(aValue);
      const bStr = String(bValue);
      return sortDirection === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
    }
  });

  // Simulacija trajanja zahtevne operacije
  while (Date.now() - startTime < 100) {
    // Simulacija zahtevne operacije
  }

  return sortedData;
}

// Poslušalec za sporočila
self.addEventListener('message', (event: MessageEvent<WorkerMessage<unknown>>) => {
  const { type, data, id } = event.data;

  try {
    let result;

    switch (type) {
      case MessageType.PROCESS_TRANSACTIONS:
        result = processTransactions(data as Transaction[]);
        break;
      case MessageType.CALCULATE_METRICS:
        result = calculateMetrics(data as MetricsData);
        break;
      case MessageType.ANALYZE_STRATEGY:
        result = analyzeStrategy(data as Strategy);
        break;
      case MessageType.FILTER_DATA: {
        const filterParams = data as FilterDataParams;
        result = filterData(filterParams.items, filterParams.filters);
        break;
      }
      case MessageType.SORT_DATA: {
        const sortParams = data as SortDataParams;
        result = sortData(sortParams.items, sortParams.sortField, sortParams.sortDirection);
        break;
      }
      default:
        throw new Error(`Neznani tip sporočila: ${type}`);
    }

    const response: WorkerResponse = {
      type,
      data: result,
      id
    };

    self.postMessage(response);
  } catch (error) {
    const response: WorkerResponse = {
      type,
      data: null,
      id,
      error: error instanceof Error ? error.message : String(error)
    };

    self.postMessage(response);
  }
});

export {};
