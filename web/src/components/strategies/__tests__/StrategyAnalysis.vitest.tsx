import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import StrategyAnalysis from '../StrategyAnalysis';
import { AnalysisStrategy } from '../../../types/strategy';
import * as useWorkerModule from '../../../hooks/useWorker';

// Mock za useWorker hook
vi.mock('../../../hooks/useWorker', () => {
  return {
    MessageType: {
      ANALYZE_STRATEGY: 'ANALYZE_STRATEGY',
    },
    default: vi.fn(),
  };
});

// Mock za recharts komponente
vi.mock('recharts', () => {
  return {
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
    LineChart: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="line-chart">{children}</div>
    ),
    Line: () => <div data-testid="chart-line" />,
    CartesianGrid: () => <div data-testid="cartesian-grid" />,
    XAxis: () => <div data-testid="x-axis" />,
    YAxis: () => <div data-testid="y-axis" />,
    Tooltip: () => <div data-testid="tooltip" />,
  };
});

// Mock za framer-motion
vi.mock('framer-motion', () => {
  return {
    motion: {
      div: ({ children, ...props }: { children: React.ReactNode }) => (
        <div data-testid="motion-div" {...props}>{children}</div>
      ),
    },
    AnimatePresence: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="animate-presence">{children}</div>
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
      },
      {
        id: '2',
        timestamp: '2023-01-02T12:00:00Z',
        status: 'failed',
        profitLoss: -0.2,
        executionTime: 150,
      },
    ],
    successRate: 75,
  };

  const mockAnalysisData = {
    name: 'Test Strategy',
    profitLoss: 0.3,
    successRate: 75,
    transactionCount: 2,
    timeSeriesData: [
      { timestamp: '2023-01-01T12:00:00Z', profitLoss: 0.5 },
      { timestamp: '2023-01-02T12:00:00Z', profitLoss: -0.2 },
    ],
    movingAverage: [
      { timestamp: '2023-01-01T12:00:00Z', value: 0.5 },
      { timestamp: '2023-01-02T12:00:00Z', value: 0.15 },
    ],
  };

  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders with strategy name', () => {
    // Mock useWorker hook
    vi.mocked(useWorkerModule.default).mockReturnValue({
      loading: false,
      error: null,
      execute: vi.fn().mockResolvedValue(mockAnalysisData),
    });

    render(<StrategyAnalysis strategy={mockStrategy} />);
    expect(screen.getByText(/Strategy Analysis: Test Strategy/i)).toBeInTheDocument();
  });

  it('renders error state', () => {
    // Mock useWorker hook za prikaz stanja napake
    vi.mocked(useWorkerModule.default).mockReturnValue({
      loading: false,
      error: 'Test error',
      execute: vi.fn().mockRejectedValue(new Error('Test error')),
    });

    render(<StrategyAnalysis strategy={mockStrategy} />);
    expect(screen.getByText(/Error: Test error/i)).toBeInTheDocument();
  });

  it('calls execute with strategy', async () => {
    // Mock useWorker hook za prikaz podatkov analize
    const mockExecute = vi.fn().mockResolvedValue(mockAnalysisData);
    vi.mocked(useWorkerModule.default).mockReturnValue({
      loading: false,
      error: null,
      execute: mockExecute,
    });

    render(<StrategyAnalysis strategy={mockStrategy} />);

    // Počakaj, da se podatki naložijo
    await waitFor(() => {
      expect(mockExecute).toHaveBeenCalledWith(mockStrategy);
    });
  });

  it('calls execute when strategy changes', () => {
    // Mock useWorker hook
    const mockExecute = vi.fn().mockResolvedValue(mockAnalysisData);
    vi.mocked(useWorkerModule.default).mockReturnValue({
      loading: false,
      error: null,
      execute: mockExecute,
    });

    const { rerender } = render(<StrategyAnalysis strategy={mockStrategy} />);
    expect(mockExecute).toHaveBeenCalledWith(mockStrategy);

    // Spremeni strategijo
    const updatedStrategy = { ...mockStrategy, id: '2', name: 'Updated Strategy' };
    rerender(<StrategyAnalysis strategy={updatedStrategy} />);
    expect(mockExecute).toHaveBeenCalledWith(updatedStrategy);
  });
});
