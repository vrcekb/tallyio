import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import VirtualizedTransactionList from '../VirtualizedTransactionList';

// Mock za react-window
vi.mock('react-window', () => ({
  default: {
    FixedSizeList: ({ children, itemCount }: { children: any; itemCount: number }) => {
      const items = [];
      for (let i = 0; i < itemCount; i++) {
        items.push(children({ index: i, style: {} }));
      }
      return <div data-testid="virtualized-list">{items}</div>;
    }
  },
  FixedSizeList: ({ children, itemCount }: { children: any; itemCount: number }) => {
    const items = [];
    for (let i = 0; i < itemCount; i++) {
      items.push(children({ index: i, style: {} }));
    }
    return <div data-testid="virtualized-list">{items}</div>;
  }
}));

// Mock za react-virtualized-auto-sizer
vi.mock('react-virtualized-auto-sizer', () => ({
  default: ({ children }: { children: any }) => {
    return children({ height: 500, width: 500 });
  }
}));

describe('VirtualizedTransactionList', () => {
  const mockTransactions = [
    {
      id: '1',
      type: 'send',
      amount: '0.5',
      token: 'ETH',
      address: '0x123',
      timestamp: '2023-05-01T12:00:00Z',
      status: 'completed' as const
    },
    {
      id: '2',
      type: 'receive',
      amount: '1.0',
      token: 'ETH',
      address: '0x456',
      timestamp: '2023-05-02T12:00:00Z',
      status: 'pending' as const
    },
    {
      id: '3',
      type: 'send',
      amount: '0.1',
      token: 'ETH',
      address: '0x789',
      timestamp: '2023-05-03T12:00:00Z',
      status: 'failed' as const
    }
  ];

  const mockOnViewDetails = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders empty message when no transactions are provided', () => {
    render(
      <VirtualizedTransactionList
        transactions={[]}
        onViewDetails={mockOnViewDetails}
        emptyMessage="No transactions"
      />
    );

    expect(screen.getByText('No transactions')).toBeInTheDocument();
  });

  it('renders transactions correctly', () => {
    render(
      <VirtualizedTransactionList
        transactions={mockTransactions}
        onViewDetails={mockOnViewDetails}
      />
    );

    // Preveri, ali so prikazane vse transakcije
    expect(screen.getByText('Sent 0.5 ETH')).toBeInTheDocument();
    expect(screen.getByText('Received 1.0 ETH')).toBeInTheDocument();
    expect(screen.getByText('Sent 0.1 ETH')).toBeInTheDocument();

    // Preveri, ali so prikazani vsi statusi
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getByText('Pending')).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
  });

  it('calls onViewDetails when view details button is clicked', () => {
    render(
      <VirtualizedTransactionList
        transactions={mockTransactions}
        onViewDetails={mockOnViewDetails}
      />
    );

    // Klikni na gumb za prikaz podrobnosti
    const viewDetailsButtons = screen.getAllByRole('button');
    viewDetailsButtons[0].click();

    // Preveri, ali je bila funkcija onViewDetails klicana s pravilnim ID-jem
    expect(mockOnViewDetails).toHaveBeenCalledWith('1');
  });

  it('applies custom className', () => {
    render(
      <VirtualizedTransactionList
        transactions={mockTransactions}
        onViewDetails={mockOnViewDetails}
        className="custom-class"
      />
    );

    const listContainer = screen.getByTestId('virtualized-list').parentElement;
    expect(listContainer).toHaveClass('custom-class');
  });
});
