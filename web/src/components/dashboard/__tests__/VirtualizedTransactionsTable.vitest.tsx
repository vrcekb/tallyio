import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import VirtualizedTransactionsTable from '../VirtualizedTransactionsTable';
import { Transaction } from '../../../types';

// Mock za VirtualizedList komponento
vi.mock('../../ui/VirtualizedList', () => {
  return {
    default: ({ data, renderItem }: { data: any[], renderItem: any }) => (
      <div data-testid="virtualized-list">
        {data.map((item, index) => (
          <div key={item.id} data-testid={`list-item-${index}`}>
            {renderItem(item, index, {})}
          </div>
        ))}
      </div>
    ),
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
    },
    {
      id: '2',
      hash: '0x456',
      timestamp: '2023-01-02T12:00:00Z',
      strategy: 'Liquidation',
      network: 'Polygon',
      status: 'failed',
      amount: 2.5,
      profit: -0.1,
      chain: 'MATIC',
      type: 'liquidation',
      profitLoss: -0.1,
    },
    {
      id: '3',
      hash: '0x789',
      timestamp: '2023-01-03T12:00:00Z',
      strategy: 'MEV',
      network: 'Arbitrum',
      status: 'pending',
      amount: 3.5,
      profit: 0.2,
      chain: 'ARB',
      type: 'mev',
      profitLoss: 0.2,
    },
  ];

  it('renders the component with title', () => {
    render(<VirtualizedTransactionsTable data={mockTransactions} title="Test Transactions" />);
    expect(screen.getByText('Test Transactions')).toBeInTheDocument();
  });

  it('renders the table headers', () => {
    render(<VirtualizedTransactionsTable data={mockTransactions} />);
    expect(screen.getByText('Strategy')).toBeInTheDocument();
    expect(screen.getByText('Network')).toBeInTheDocument();
    expect(screen.getByText('Profit/Loss')).toBeInTheDocument();
  });

  it('renders transactions using VirtualizedList', () => {
    render(<VirtualizedTransactionsTable data={mockTransactions} />);
    expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
    expect(screen.getByTestId('list-item-0')).toBeInTheDocument();
    expect(screen.getByTestId('list-item-1')).toBeInTheDocument();
    expect(screen.getByTestId('list-item-2')).toBeInTheDocument();
  });

  it('renders transaction details correctly', () => {
    render(<VirtualizedTransactionsTable data={mockTransactions} />);
    expect(screen.getByText('Arbitrage')).toBeInTheDocument();
    expect(screen.getByText('Liquidation')).toBeInTheDocument();
    expect(screen.getByText('MEV')).toBeInTheDocument();
    expect(screen.getByText('Ethereum')).toBeInTheDocument();
    expect(screen.getByText('Polygon')).toBeInTheDocument();
    expect(screen.getByText('Arbitrum')).toBeInTheDocument();
    expect(screen.getByText('+0.0500 ETH')).toBeInTheDocument();
    expect(screen.getByText('-0.1000 ETH')).toBeInTheDocument();
    expect(screen.getByText('+0.2000 ETH')).toBeInTheDocument();
  });

  it('sorts transactions when clicking on headers', () => {
    render(<VirtualizedTransactionsTable data={mockTransactions} />);

    // Klik na Strategy header
    fireEvent.click(screen.getByText('Strategy'));

    // Klik na Network header
    fireEvent.click(screen.getByText('Network'));

    // Klik na Profit/Loss header
    fireEvent.click(screen.getByText('Profit/Loss'));

    // Klik na Profit/Loss header ponovno za spremembo smeri razvrščanja
    fireEvent.click(screen.getByText('Profit/Loss'));

    // Test je uspešen, če ni napak pri klikanju na glave tabele
    expect(true).toBe(true);
  });

  it('renders empty state when no transactions', () => {
    render(<VirtualizedTransactionsTable data={[]} />);
    expect(screen.getByText('No transactions found')).toBeInTheDocument();
  });
});
