import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import SettingsPage from '../SettingsPage';
import { ToastProvider } from '../../contexts/ToastContext';

// Mock za Layout komponento
jest.mock('../../components/layout/Layout', () => {
  return {
    __esModule: true,
    default: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="layout">{children}</div>
    )
  };
});

// Mock za mockData
jest.mock('../../mockData/settings', () => {
  return {
    userProfile: {
      name: 'Test User',
      email: 'test@example.com',
      lastLogin: '2023-05-01T12:00:00Z',
      twoFactorEnabled: true
    },
    apiKeys: [
      {
        id: 'key1',
        name: 'Test API Key',
        key: 'test-api-key-123',
        created: '2023-05-01T12:00:00Z',
        permissions: ['read', 'write']
      }
    ],
    notificationSettings: [
      {
        type: 'System Alerts',
        email: true,
        push: false,
        slack: true,
        telegram: false
      }
    ],
    integrationConfigs: [
      {
        id: 'integration1',
        name: 'Test Integration',
        status: 'connected',
        lastSync: '2023-05-01T12:00:00Z'
      }
    ]
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

describe('SettingsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock za navigator.clipboard.writeText
    Object.assign(navigator, {
      clipboard: {
        writeText: jest.fn()
      }
    });
  });

  it('renders settings page with correct title', () => {
    renderWithProviders(<SettingsPage />);
    
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('renders tabs with correct labels', () => {
    renderWithProviders(<SettingsPage />);
    
    expect(screen.getByText('User Profile')).toBeInTheDocument();
    expect(screen.getByText('API Keys')).toBeInTheDocument();
    expect(screen.getByText('Notifications')).toBeInTheDocument();
    expect(screen.getByText('Integrations')).toBeInTheDocument();
  });

  it('renders user profile information', () => {
    renderWithProviders(<SettingsPage />);
    
    expect(screen.getByText('Test User')).toBeInTheDocument();
    expect(screen.getByText('test@example.com')).toBeInTheDocument();
  });

  it('changes tab when tab is clicked', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<SettingsPage />);
    
    // Klikni na zavihek API Keys
    await user.click(screen.getByText('API Keys'));
    
    // Preveri, ali je prikazan API Keys zavihek
    expect(screen.getByText('Manage your API keys for external integrations')).toBeInTheDocument();
    expect(screen.getByText('Test API Key')).toBeInTheDocument();
  });

  it('copies API key to clipboard when copy button is clicked', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<SettingsPage />);
    
    // Klikni na zavihek API Keys
    await user.click(screen.getByText('API Keys'));
    
    // Najdi gumb za kopiranje
    const copyButtons = screen.getAllByRole('button');
    const copyButton = copyButtons.find(button => button.innerHTML.includes('Copy'));
    
    // Klikni na gumb za kopiranje
    if (copyButton) {
      await user.click(copyButton);
    }
    
    // Preveri, ali je bila funkcija writeText klicana s pravilnim ključem
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('test-api-key-123');
  });

  it('renders notification settings', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<SettingsPage />);
    
    // Klikni na zavihek Notifications
    await user.click(screen.getByText('Notifications'));
    
    // Preveri, ali so prikazane nastavitve obvestil
    expect(screen.getByText('System Alerts')).toBeInTheDocument();
  });

  it('renders integration settings', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<SettingsPage />);
    
    // Klikni na zavihek Integrations
    await user.click(screen.getByText('Integrations'));
    
    // Preveri, ali so prikazane nastavitve integracij
    expect(screen.getByText('Test Integration')).toBeInTheDocument();
    expect(screen.getByText('connected')).toBeInTheDocument();
  });

  it('shows save button in profile tab', () => {
    renderWithProviders(<SettingsPage />);
    
    expect(screen.getByText('Save Changes')).toBeInTheDocument();
  });

  it('shows create API key button in API keys tab', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(<SettingsPage />);
    
    // Klikni na zavihek API Keys
    await user.click(screen.getByText('API Keys'));
    
    expect(screen.getByText('New API Key')).toBeInTheDocument();
  });
});
