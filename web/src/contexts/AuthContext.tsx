import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import apiService from '../services/api';
import mockApiService from '../services/mockApi';

// Vmesnik za uporabnika
export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user';
  avatar?: string;
}

// Vmesnik za stanje avtentikacije
interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

// Vmesnik za kontekst avtentikacije
interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  resetPassword: (email: string) => Promise<void>;
  updateProfile: (data: Partial<User>) => Promise<void>;
  clearError: () => void;
}

// Privzeto stanje
const defaultState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,
};

// Ustvarjanje konteksta
const AuthContext = createContext<AuthContextType>({
  ...defaultState,
  login: async () => {},
  logout: async () => {},
  register: async () => {},
  resetPassword: async () => {},
  updateProfile: async () => {},
  clearError: () => {},
});

// Hook za uporabo konteksta
export const useAuth = () => useContext(AuthContext);

// Props za AuthProvider
interface AuthProviderProps {
  children: ReactNode;
}

// Določi, ali naj se uporablja mock API
const useMockApi = import.meta.env.DEV;

// Mock uporabniki za razvojno okolje
const mockUsers: User[] = [
  {
    id: '1',
    name: 'Admin User',
    email: 'admin@example.com',
    role: 'admin',
    avatar: 'https://i.pravatar.cc/150?u=admin',
  },
  {
    id: '2',
    name: 'Test User',
    email: 'user@example.com',
    role: 'user',
    avatar: 'https://i.pravatar.cc/150?u=user',
  },
];

// Registriraj mock končne točke
if (useMockApi) {
  mockApiService.registerEndpoint('auth/me', (_token: string) => {
    const userId = localStorage.getItem('auth_user_id');
    return mockUsers.find(u => u.id === userId);
  });

  mockApiService.registerEndpoint('auth/login', (credentials: { email: string, password: string }) => {
    const user = mockUsers.find(u => u.email === credentials.email);
    if (!user || credentials.password !== 'password') {
      throw new Error('Invalid credentials');
    }
    return { user, token: `mock-token-${Date.now()}` };
  });

  mockApiService.registerEndpoint('auth/register', (data: { name: string, email: string, password: string }) => {
    const existingUser = mockUsers.find(u => u.email === data.email);
    if (existingUser) {
      throw new Error('User already exists');
    }

    const newUser: User = {
      id: `${mockUsers.length + 1}`,
      name: data.name,
      email: data.email,
      role: 'user',
      avatar: `https://i.pravatar.cc/150?u=${data.email}`,
    };

    mockUsers.push(newUser);

    return { user: newUser, token: `mock-token-${Date.now()}` };
  });

  mockApiService.registerEndpoint('auth/reset-password', (data: { email: string }) => {
    const user = mockUsers.find(u => u.email === data.email);
    if (!user) {
      throw new Error('User not found');
    }
    return { success: true };
  });

  mockApiService.registerEndpoint('auth/update-profile', (data: Partial<User>) => {
    const userId = localStorage.getItem('auth_user_id');
    const userIndex = mockUsers.findIndex(u => u.id === userId);

    if (userIndex === -1) {
      throw new Error('User not found');
    }

    mockUsers[userIndex] = { ...mockUsers[userIndex], ...data };

    return mockUsers[userIndex];
  });
}

/**
 * AuthProvider komponenta za zagotavljanje avtentikacije vsem komponentam v aplikaciji
 */
export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [state, setState] = useState<AuthState>(defaultState);

  // Nastavi avtentikacijski token za API
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      apiService.setAuthToken(token);
    }
  }, []);

  // Preveri, ali je uporabnik že prijavljen
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Preveri, ali obstaja token v localStorage
        const token = localStorage.getItem('auth_token');

        // Avtomatska prijava v razvojnem okolju
        if (import.meta.env.DEV && !token) {
          console.log('DEV MODE: Auto login enabled');

          // Izberi prvega uporabnika iz mock uporabnikov
          const autoUser = mockUsers[0];
          const autoToken = `mock-token-${Date.now()}`;

          // Shrani podatke v localStorage
          localStorage.setItem('auth_token', autoToken);
          localStorage.setItem('auth_user_id', autoUser.id);

          // Nastavi token za API
          apiService.setAuthToken(autoToken);

          setState({
            user: autoUser,
            isAuthenticated: true,
            isLoading: false,
            error: null,
          });

          console.log('DEV MODE: Auto logged in as', autoUser.name);
          return;
        }

        if (!token) {
          setState({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,
          });
          return;
        }

        // Nastavi token za API
        apiService.setAuthToken(token);

        // Pridobi podatke o uporabniku
        const api = useMockApi ? mockApiService : apiService;
        const response = await api.get<User>('auth/me');

        if (response.error || !response.data) {
          throw new Error(response.error || 'Failed to get user data');
        }

        setState({
          user: response.data,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });
      } catch (error) {
        // Odstrani neveljaven token
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user_id');
        apiService.setAuthToken(null);

        setState({
          user: null,
          isAuthenticated: false,
          isLoading: false,
          error: error instanceof Error ? error.message : 'An error occurred',
        });
      }
    };

    checkAuth();
  }, []);

  // Funkcija za prijavo
  const login = async (email: string, password: string) => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));

      // Izvedi API klic za prijavo
      const api = useMockApi ? mockApiService : apiService;
      const response = await api.post<{ user: User; token: string }>('auth/login', { email, password });

      if (response.error || !response.data) {
        throw new Error(response.error || 'Login failed');
      }

      const { user, token } = response.data;

      // Shrani token v localStorage
      localStorage.setItem('auth_token', token);
      localStorage.setItem('auth_user_id', user.id);

      // Nastavi token za API
      apiService.setAuthToken(token);

      setState({
        user,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Funkcija za odjavo
  const logout = async () => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));

      // Izvedi API klic za odjavo (če je potrebno)
      // const api = useMockApi ? mockApiService : apiService;
      // await api.post('auth/logout');

      // Odstrani token iz localStorage
      localStorage.removeItem('auth_token');
      localStorage.removeItem('auth_user_id');

      // Odstrani token iz API
      apiService.setAuthToken(null);

      setState({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Funkcija za registracijo
  const register = async (name: string, email: string, password: string) => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));

      // Izvedi API klic za registracijo
      const api = useMockApi ? mockApiService : apiService;
      const response = await api.post<{ user: User; token: string }>('auth/register', { name, email, password });

      if (response.error || !response.data) {
        throw new Error(response.error || 'Registration failed');
      }

      const { user, token } = response.data;

      // Shrani token v localStorage
      localStorage.setItem('auth_token', token);
      localStorage.setItem('auth_user_id', user.id);

      // Nastavi token za API
      apiService.setAuthToken(token);

      setState({
        user,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Funkcija za ponastavitev gesla
  const resetPassword = async (email: string) => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));

      // Izvedi API klic za ponastavitev gesla
      const api = useMockApi ? mockApiService : apiService;
      const response = await api.post<{ success: boolean }>('auth/reset-password', { email });

      if (response.error || !response.data) {
        throw new Error(response.error || 'Password reset failed');
      }

      setState(prev => ({
        ...prev,
        isLoading: false,
        error: null,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Funkcija za posodobitev profila
  const updateProfile = async (data: Partial<User>) => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));

      // Preveri, ali je uporabnik prijavljen
      if (!state.user) {
        throw new Error('User not authenticated');
      }

      // Izvedi API klic za posodobitev profila
      const api = useMockApi ? mockApiService : apiService;
      const response = await api.put<User>('auth/update-profile', data);

      if (response.error || !response.data) {
        throw new Error(response.error || 'Profile update failed');
      }

      setState(prev => ({
        ...prev,
        user: response.data,
        isLoading: false,
        error: null,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      }));
    }
  };

  // Funkcija za brisanje napake
  const clearError = () => {
    setState(prev => ({ ...prev, error: null }));
  };

  // Vrednost konteksta
  const contextValue: AuthContextType = {
    ...state,
    login,
    logout,
    register,
    resetPassword,
    updateProfile,
    clearError,
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
