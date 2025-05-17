/**
 * Tipi za strategije in analize
 */

// Transakcija za analizo
export interface StrategyTransaction {
  id: string;
  timestamp: string;
  status: string;
  profitLoss: number;
  executionTime: number;
}

// Strategija za analizo
export interface AnalysisStrategy {
  id: string;
  name: string;
  type: string;
  status: string;
  profit: number;
  transactions: StrategyTransaction[];
  successRate: number;
}

// Podatki časovne vrste
export interface TimeSeriesData {
  timestamp: string;
  profitLoss: number;
}

// Podatki drsečega povprečja
export interface MovingAverageData {
  timestamp: string;
  value: number;
}

// Rezultat analize strategije
export interface StrategyAnalysisResult {
  name: string;
  profitLoss: number;
  successRate: number;
  transactionCount: number;
  timeSeriesData: TimeSeriesData[];
  movingAverage: MovingAverageData[];
}
