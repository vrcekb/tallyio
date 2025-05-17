import React from 'react';
import { render } from '@testing-library/react';
import VirtualizedTransactionsTable from '../VirtualizedTransactionsTable';
import { Transaction } from '../../../types';

// Mock za VirtualizedList komponento
jest.mock('../../ui/VirtualizedList', () => {
  return {
    __esModule: true,
    default: () => <div data-testid="virtualized-list" />
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

describe('VirtualizedTransactionsTable', () => {
  const mockTransactions: Transaction[] = [
    {
      id: '1',
      hash: '0x123',
      timestamp: '2023-01-01T12:00:00Z',
      strategy: 'Arbitrage',
      network: 'Ethereum',
      status: 'success',
      amount: 1.5,
      profit: 0.05,
      chain: 'ETH',
      type: 'swap',
      profitLoss: 0.05,
    }
  ];

  it('renders without crashing', () => {
    // Preveri, da se komponenta upodobi brez napak
    const { container } = render(<VirtualizedTransactionsTable data={mockTransactions} />);
    expect(container).toBeTruthy();
  });
});
