import React from 'react';
import { render } from '@testing-library/react';
import StrategyAnalysis from '../StrategyAnalysis';
import { AnalysisStrategy } from '../../../types/strategy';
import * as useWorkerModule from '../../../hooks/useWorker';

// Mock za useWorker hook
jest.mock('../../../hooks/useWorker', () => {
  return {
    __esModule: true,
    MessageType: {
      ANALYZE_STRATEGY: 'ANALYZE_STRATEGY',
    },
    default: jest.fn(),
  };
});

// Mock za Card komponento
jest.mock('../../ui/Card', () => {
  return {
    __esModule: true,
    default: ({ children, title }: { children: React.ReactNode, title: string }) => (
      <div data-testid="card">
        <div data-testid="card-title">{title}</div>
        <div data-testid="card-content">{children}</div>
      </div>
    )
  };
});

// Mock za recharts komponente
jest.mock('recharts', () => {
  return {
    LineChart: () => <div data-testid="line-chart" />,
    Line: () => <div data-testid="line" />,
    XAxis: () => <div data-testid="x-axis" />,
    YAxis: () => <div data-testid="y-axis" />,
    CartesianGrid: () => <div data-testid="cartesian-grid" />,
    Tooltip: () => <div data-testid="tooltip" />,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

describe('StrategyAnalysis', () => {
  const mockStrategy: AnalysisStrategy = {
    id: '1',
    name: 'Test Strategy',
    type: 'Arbitrage',
    status: 'active',
    profit: 1.5,
    transactions: [
      {
        id: '1',
        timestamp: '2023-01-01T12:00:00Z',
        status: 'success',
        profitLoss: 0.5,
        executionTime: 100,
      }
    ],
    successRate: 75,
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock useWorker hook
    (useWorkerModule.default as jest.Mock).mockReturnValue({
      loading: false,
      error: null,
      execute: jest.fn().mockResolvedValue({
        name: 'Test Strategy',
        profitLoss: 0.5,
        successRate: 75,
        transactionCount: 1,
        timeSeriesData: [],
        movingAverage: []
      }),
    });
  });

  it('renders without crashing', () => {
    // Preveri, da se komponenta upodobi brez napak
    const { container } = render(<StrategyAnalysis strategy={mockStrategy} />);
    expect(container).toBeTruthy();
  });
});
