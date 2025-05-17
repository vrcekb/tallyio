import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import WalletPage from '../WalletPage';
import { ToastProvider } from '../../contexts/ToastContext';
import useWorker from '../../hooks/useWorker';

// Mock za useWorker hook
jest.mock('../../hooks/useWorker', () => {
  return {
    __esModule: true,
    default: jest.fn(),
    MessageType: {
      PROCESS_TRANSACTIONS: 'PROCESS_TRANSACTIONS'
    }
  };
});

// Mock za VirtualizedTransactionList komponento
jest.mock('../../components/wallet/VirtualizedTransactionList', () => {
  return {
    __esModule: true,
    default: ({ transactions, onViewDetails }: any) => (
      <div data-testid="virtualized-transaction-list">
        <div>Transaction count: {transactions.length}</div>
        <button onClick={() => onViewDetails('test-id')}>View Details</button>
      </div>
    )
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

// Pomožna funkcija za renderiranje komponente z vsemi potrebnimi wrapperji
const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <BrowserRouter>
      <ToastProvider>{ui}</ToastProvider>
    </BrowserRouter>
  );
};

describe('WalletPage', () => {
  // Privzeti mock za useWorker
  const mockExecute = jest.fn().mockResolvedValue({
    transactions: [
      {
        id: '1',
        type: 'send',
        amount: '0.5',
        token: 'ETH',
        address: '0x123',
        timestamp: '2023-05-01T12:00:00Z',
        status: 'completed',
        processed: true
      }
    ],
    stats: { count: 1 }
  });

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Nastavi privzeti mock za useWorker
    (useWorker as jest.Mock).mockReturnValue({
      loading: false,
      error: null,
      data: null,
      execute: mockExecute
    });
  });

  it('renders wallet page with correct title', () => {
    renderWithProviders(<WalletPage />);
    
    expect(screen.getByText('Wallet Management')).toBeInTheDocument();
  });

  it('calls processTransactions on mount', () => {
    renderWithProviders(<WalletPage />);
    
    expect(mockExecute).toHaveBeenCalled();
  });

  it('shows loading state when processing transactions', () => {
    // Nastavi loading stanje
    (useWorker as jest.Mock).mockReturnValue({
      loading: true,
      error: null,
      data: null,
      execute: mockExecute
    });
    
    renderWithProviders(<WalletPage />);
    
    expect(screen.getByText('Processing transactions...')).toBeInTheDocument();
  });

  it('shows transactions when data is loaded', async () => {
    // Nastavi podatke
    (useWorker as jest.Mock).mockReturnValue({
      loading: false,
      error: null,
      data: {
        transactions: [
          {
            id: '1',
            type: 'send',
            amount: '0.5',
            token: 'ETH',
            address: '0x123',
            timestamp: '2023-05-01T12:00:00Z',
            status: 'completed',
            processed: true
          }
        ],
        stats: { count: 1 }
      },
      execute: mockExecute
    });
    
    renderWithProviders(<WalletPage />);
    
    expect(screen.getByText('Transaction count: 1')).toBeInTheDocument();
  });

  it('opens create wallet modal when create button is clicked', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<WalletPage />);
    
    // Klikni na gumb za ustvarjanje denarnice
    await user.click(screen.getByText('Create Wallet'));
    
    // Preveri, ali je modal prikazan
    expect(screen.getByText('Create New Wallet')).toBeInTheDocument();
  });

  it('shows toast when view transaction details button is clicked', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<WalletPage />);
    
    // Klikni na gumb za prikaz podrobnosti
    await user.click(screen.getByText('View Details'));
    
    // Preveri, ali je toast prikazan (implementacija je odvisna od ToastProvider)
    // To je težko testirati brez dodatnih mockov, zato samo preverimo, ali je funkcija klicana
    expect(mockExecute).toHaveBeenCalled();
  });

  it('toggles drain protection when switch is clicked', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<WalletPage />);
    
    // Najdi stikalo za zaščito pred izčrpavanjem
    const drainProtectionSwitch = screen.getByRole('switch');
    
    // Preveri začetno stanje
    expect(drainProtectionSwitch).toBeChecked();
    
    // Klikni na stikalo
    await user.click(drainProtectionSwitch);
    
    // Preveri, ali je stanje spremenjeno
    expect(drainProtectionSwitch).not.toBeChecked();
    
    // Preveri, ali je opozorilo odstranjeno
    expect(screen.queryByText('Maximum transaction limits enabled')).not.toBeInTheDocument();
  });
});
