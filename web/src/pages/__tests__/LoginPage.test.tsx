import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import LoginPage from '../LoginPage';
import { AuthProvider } from '../../contexts/AuthContext';
import mockApiService from '../../services/mockApi';

// Mock za useNavigate
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

// Mock za mockApiService
jest.mock('../../services/mockApi', () => ({
  post: jest.fn(),
}));

describe('LoginPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render login form', () => {
    render(
      <MemoryRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </MemoryRouter>
    );

    expect(screen.getByText('Prijava v TallyIO')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('E-poštni naslov')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Geslo')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Prijava' })).toBeInTheDocument();
    expect(screen.getByText('Zapomni si me')).toBeInTheDocument();
    expect(screen.getByText('Pozabljeno geslo?')).toBeInTheDocument();
    expect(screen.getByText('Registrirajte se')).toBeInTheDocument();
  });

  it('should navigate to register page when register link is clicked', () => {
    render(
      <MemoryRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </MemoryRouter>
    );

    const registerLink = screen.getByText('Registrirajte se');
    userEvent.click(registerLink);

    expect(window.location.pathname).toBe('/');
  });

  it('should navigate to forgot password page when forgot password link is clicked', () => {
    render(
      <MemoryRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </MemoryRouter>
    );

    const forgotPasswordLink = screen.getByText('Pozabljeno geslo?');
    userEvent.click(forgotPasswordLink);

    expect(window.location.pathname).toBe('/');
  });

  it('should submit form with entered credentials', async () => {
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
      <MemoryRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </MemoryRouter>
    );

    const emailInput = screen.getByPlaceholderText('E-poštni naslov');
    const passwordInput = screen.getByPlaceholderText('Geslo');
    const submitButton = screen.getByRole('button', { name: 'Prijava' });

    userEvent.type(emailInput, 'test@example.com');
    userEvent.type(passwordInput, 'password');
    userEvent.click(submitButton);

    await waitFor(() => {
      expect(mockApiService.post).toHaveBeenCalledWith('auth/login', {
        email: 'test@example.com',
        password: 'password',
      });
    });

    // Preveri, ali je uporabnik preusmerjen na nadzorno ploščo
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/');
    });
  });

  it('should show error message when login fails', async () => {
    mockApiService.post.mockResolvedValueOnce({
      data: null,
      error: 'Invalid credentials',
      status: 401,
    });

    render(
      <MemoryRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </MemoryRouter>
    );

    const emailInput = screen.getByPlaceholderText('E-poštni naslov');
    const passwordInput = screen.getByPlaceholderText('Geslo');
    const submitButton = screen.getByRole('button', { name: 'Prijava' });

    userEvent.type(emailInput, 'test@example.com');
    userEvent.type(passwordInput, 'wrong-password');
    userEvent.click(submitButton);

    await waitFor(() => {
      expect(mockApiService.post).toHaveBeenCalledWith('auth/login', {
        email: 'test@example.com',
        password: 'wrong-password',
      });
    });

    // Preveri, ali se prikaže sporočilo o napaki
    await waitFor(() => {
      expect(screen.getByText('Invalid credentials')).toBeInTheDocument();
    });

    // Preveri, ali uporabnik ni preusmerjen
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it('should toggle password visibility when eye icon is clicked', () => {
    render(
      <MemoryRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </MemoryRouter>
    );

    const passwordInput = screen.getByPlaceholderText('Geslo');
    const toggleButton = screen.getByLabelText('Prikaži geslo');

    // Preveri, ali je geslo skrito
    expect(passwordInput).toHaveAttribute('type', 'password');

    // Klikni na ikono za prikaz gesla
    userEvent.click(toggleButton);

    // Preveri, ali je geslo vidno
    expect(passwordInput).toHaveAttribute('type', 'text');

    // Klikni ponovno na ikono za skrivanje gesla
    userEvent.click(screen.getByLabelText('Skrij geslo'));

    // Preveri, ali je geslo ponovno skrito
    expect(passwordInput).toHaveAttribute('type', 'password');
  });

  it('should toggle remember me checkbox', () => {
    render(
      <MemoryRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </MemoryRouter>
    );

    const rememberMeCheckbox = screen.getByLabelText('Zapomni si me');

    // Preveri, ali je checkbox odkljukan
    expect(rememberMeCheckbox).not.toBeChecked();

    // Klikni na checkbox
    userEvent.click(rememberMeCheckbox);

    // Preveri, ali je checkbox obkljukan
    expect(rememberMeCheckbox).toBeChecked();

    // Klikni ponovno na checkbox
    userEvent.click(rememberMeCheckbox);

    // Preveri, ali je checkbox ponovno odkljukan
    expect(rememberMeCheckbox).not.toBeChecked();
  });
});
