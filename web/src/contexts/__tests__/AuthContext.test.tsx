import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AuthProvider, useAuth } from '../AuthContext';
import mockApiService from '../../services/mockApi';

// Mock za mockApiService
jest.mock('../../services/mockApi', () => ({
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
}));

// Mock za localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
};
Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// Komponenta za testiranje
const TestComponent = () => {
  const { user, isAuthenticated, isLoading, error, login, logout, register, resetPassword, clearError } = useAuth();

  return (
    <div>
      <div data-testid="loading">{isLoading ? 'Loading' : 'Not Loading'}</div>
      <div data-testid="authenticated">{isAuthenticated ? 'Authenticated' : 'Not Authenticated'}</div>
      <div data-testid="user">{user ? JSON.stringify(user) : 'No User'}</div>
      <div data-testid="error">{error || 'No Error'}</div>
      <button data-testid="login" onClick={() => login('test@example.com', 'password')}>
        Login
      </button>
      <button data-testid="logout" onClick={() => logout()}>
        Logout
      </button>
      <button data-testid="register" onClick={() => register('Test User', 'test@example.com', 'password')}>
        Register
      </button>
      <button data-testid="reset-password" onClick={() => resetPassword('test@example.com')}>
        Reset Password
      </button>
      <button data-testid="clear-error" onClick={() => clearError()}>
        Clear Error
      </button>
    </div>
  );
};

describe('AuthContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockImplementation((key) => {
      if (key === 'auth_token') return null;
      if (key === 'auth_user_id') return null;
      return null;
    });
  });

  it('should initialize with default values', async () => {
    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );

    expect(screen.getByTestId('loading')).toHaveTextContent('Loading');
    
    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    });
    
    expect(screen.getByTestId('authenticated')).toHaveTextContent('Not Authenticated');
    expect(screen.getByTestId('user')).toHaveTextContent('No User');
    expect(screen.getByTestId('error')).toHaveTextContent('No Error');
  });

  it('should check for existing token on initialization', async () => {
    localStorageMock.getItem.mockImplementation((key) => {
      if (key === 'auth_token') return 'mock-token';
      if (key === 'auth_user_id') return '1';
      return null;
    });

    mockApiService.get.mockResolvedValueOnce({
      data: {
        id: '1',
        name: 'Test User',
        email: 'test@example.com',
        role: 'user',
      },
      error: null,
      status: 200,
    });

    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );

    expect(screen.getByTestId('loading')).toHaveTextContent('Loading');

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    });

    expect(mockApiService.get).toHaveBeenCalledWith('auth/me');
    expect(screen.getByTestId('authenticated')).toHaveTextContent('Authenticated');
    expect(screen.getByTestId('user')).not.toHaveTextContent('No User');
  });

  it('should handle login', async () => {
    mockApiService.post.mockResolvedValueOnce({
      data: {
        user: {
          id: '1',
          name: 'Test User',
          email: 'test@example.com',
          role: 'user',
        },
        token: 'mock-token',
      },
      error: null,
      status: 200,
    });

    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    });

    const loginButton = screen.getByTestId('login');
    await act(async () => {
      userEvent.click(loginButton);
    });

    expect(mockApiService.post).toHaveBeenCalledWith('auth/login', {
      email: 'test@example.com',
      password: 'password',
    });

    await waitFor(() => {
      expect(screen.getByTestId('authenticated')).toHaveTextContent('Authenticated');
    });

    expect(localStorageMock.setItem).toHaveBeenCalledWith('auth_token', 'mock-token');
    expect(localStorageMock.setItem).toHaveBeenCalledWith('auth_user_id', '1');
    expect(screen.getByTestId('user')).not.toHaveTextContent('No User');
  });

  it('should handle login error', async () => {
    mockApiService.post.mockResolvedValueOnce({
      data: null,
      error: 'Invalid credentials',
      status: 401,
    });

    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    });

    const loginButton = screen.getByTestId('login');
    await act(async () => {
      userEvent.click(loginButton);
    });

    expect(mockApiService.post).toHaveBeenCalledWith('auth/login', {
      email: 'test@example.com',
      password: 'password',
    });

    await waitFor(() => {
      expect(screen.getByTestId('error')).toHaveTextContent('Invalid credentials');
    });

    expect(screen.getByTestId('authenticated')).toHaveTextContent('Not Authenticated');
    expect(screen.getByTestId('user')).toHaveTextContent('No User');
  });

  it('should handle logout', async () => {
    // Najprej se prijavimo
    mockApiService.post.mockResolvedValueOnce({
      data: {
        user: {
          id: '1',
          name: 'Test User',
          email: 'test@example.com',
          role: 'user',
        },
        token: 'mock-token',
      },
      error: null,
      status: 200,
    });

    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    });

    const loginButton = screen.getByTestId('login');
    await act(async () => {
      userEvent.click(loginButton);
    });

    await waitFor(() => {
      expect(screen.getByTestId('authenticated')).toHaveTextContent('Authenticated');
    });

    // Nato se odjavimo
    const logoutButton = screen.getByTestId('logout');
    await act(async () => {
      userEvent.click(logoutButton);
    });

    await waitFor(() => {
      expect(screen.getByTestId('authenticated')).toHaveTextContent('Not Authenticated');
    });

    expect(localStorageMock.removeItem).toHaveBeenCalledWith('auth_token');
    expect(localStorageMock.removeItem).toHaveBeenCalledWith('auth_user_id');
    expect(screen.getByTestId('user')).toHaveTextContent('No User');
  });

  it('should clear error', async () => {
    mockApiService.post.mockResolvedValueOnce({
      data: null,
      error: 'Invalid credentials',
      status: 401,
    });

    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    });

    const loginButton = screen.getByTestId('login');
    await act(async () => {
      userEvent.click(loginButton);
    });

    await waitFor(() => {
      expect(screen.getByTestId('error')).toHaveTextContent('Invalid credentials');
    });

    const clearErrorButton = screen.getByTestId('clear-error');
    await act(async () => {
      userEvent.click(clearErrorButton);
    });

    expect(screen.getByTestId('error')).toHaveTextContent('No Error');
  });
});
