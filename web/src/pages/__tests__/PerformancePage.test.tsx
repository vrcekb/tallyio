import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import PerformancePage from '../PerformancePage';

// Mock za recharts komponente
jest.mock('recharts', () => {
  const OriginalModule = jest.requireActual('recharts');
  
  return {
    ...OriginalModule,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
    LineChart: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="line-chart">{children}</div>
    ),
    Line: () => <div data-testid="chart-line" />,
    XAxis: () => <div data-testid="x-axis" />,
    YAxis: () => <div data-testid="y-axis" />,
    CartesianGrid: () => <div data-testid="cartesian-grid" />,
    Tooltip: () => <div data-testid="chart-tooltip" />
  };
});

// Mock za Layout komponento
jest.mock('../../components/layout/Layout', () => {
  return {
    __esModule: true,
    default: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="layout">{children}</div>
    )
  };
});

// Mock za generateResourceUsage funkcijo
jest.mock('../../mockData/performance', () => {
  const originalModule = jest.requireActual('../../mockData/performance');
  
  return {
    ...originalModule,
    generateResourceUsage: jest.fn().mockReturnValue([
      { timestamp: '00:00', cpu: 10, memory: 20, network: 30 },
      { timestamp: '01:00', cpu: 15, memory: 25, network: 35 }
    ])
  };
});

// Pomožna funkcija za renderiranje komponente z vsemi potrebnimi wrapperji
const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <BrowserRouter>
      {ui}
    </BrowserRouter>
  );
};

describe('PerformancePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders performance page with correct title', () => {
    renderWithProviders(<PerformancePage />);
    
    expect(screen.getByText('System Performance')).toBeInTheDocument();
  });

  it('renders time range selector', () => {
    renderWithProviders(<PerformancePage />);
    
    expect(screen.getByText('Last 24 Hours')).toBeInTheDocument();
  });

  it('renders export report button', () => {
    renderWithProviders(<PerformancePage />);
    
    expect(screen.getByText('Export Report')).toBeInTheDocument();
  });

  it('renders tabs with correct labels', () => {
    renderWithProviders(<PerformancePage />);
    
    expect(screen.getByText('Overview')).toBeInTheDocument();
    expect(screen.getByText('CPU')).toBeInTheDocument();
    expect(screen.getByText('Memory')).toBeInTheDocument();
    expect(screen.getByText('Network')).toBeInTheDocument();
  });

  it('renders performance metrics in overview tab', () => {
    renderWithProviders(<PerformancePage />);
    
    // Preveri, ali so prikazane metrike
    expect(screen.getByText('Resource Usage')).toBeInTheDocument();
    expect(screen.getByText('Service Health')).toBeInTheDocument();
  });

  it('changes tab when tab is clicked', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<PerformancePage />);
    
    // Klikni na zavihek CPU
    await user.click(screen.getByText('CPU'));
    
    // Preveri, ali je prikazan CPU zavihek
    expect(screen.getByText('CPU Performance')).toBeInTheDocument();
    expect(screen.getByText('Detailed CPU metrics and utilization')).toBeInTheDocument();
  });

  it('renders charts with correct data', () => {
    renderWithProviders(<PerformancePage />);
    
    // Preveri, ali so prikazani grafi
    const charts = screen.getAllByTestId('line-chart');
    expect(charts.length).toBeGreaterThan(0);
    
    // Preveri, ali so prikazane osi
    const xAxes = screen.getAllByTestId('x-axis');
    const yAxes = screen.getAllByTestId('y-axis');
    expect(xAxes.length).toBeGreaterThan(0);
    expect(yAxes.length).toBeGreaterThan(0);
    
    // Preveri, ali so prikazane črte
    const lines = screen.getAllByTestId('chart-line');
    expect(lines.length).toBeGreaterThan(0);
  });

  it('renders service health status correctly', () => {
    renderWithProviders(<PerformancePage />);
    
    // Preveri, ali so prikazani statusi storitev
    expect(screen.getByText('Uptime:')).toBeInTheDocument();
    expect(screen.getByText('Response:')).toBeInTheDocument();
  });
});
