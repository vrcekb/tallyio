import { useState, useEffect, useCallback } from 'react';
import apiService from '../services/api';
import mockApiService from '../services/mockApi';

// Vmesnik za stanje API klica
interface ApiState<T> {
  data: T | null;
  isLoading: boolean;
  error: string | null;
  status: number | null;
}

// Vmesnik za možnosti API klica
interface ApiOptions<T> {
  skip?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (error: string, status: number) => void;
  useMock?: boolean;
}

/**
 * Hook za uporabo API-ja
 * @param method HTTP metoda
 * @param endpoint Končna točka API-ja
 * @param body Telo zahteve (za POST, PUT, PATCH)
 * @param params Parametri poizvedbe (za GET)
 * @param options Dodatne možnosti
 * @returns Stanje API klica in funkcija za ponovno izvajanje klica
 */
function useApi<T = unknown>(
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  endpoint: string,
  body?: unknown,
  params?: Record<string, string | number | boolean>,
  options: ApiOptions<T> = {}
) {
  const { skip = false, onSuccess, onError, useMock = import.meta.env.DEV } = options;

  const [state, setState] = useState<ApiState<T>>({
    data: null,
    isLoading: !skip,
    error: null,
    status: null,
  });

  // Funkcija za izvajanje API klica
  const execute = useCallback(
    async (newBody?: unknown, newParams?: Record<string, string | number | boolean>) => {
      setState(prev => ({ ...prev, isLoading: true, error: null }));

      try {
        const api = useMock ? mockApiService : apiService;
        let response;

        switch (method) {
          case 'GET':
            response = await api.get<T>(endpoint, newParams || params);
            break;
          case 'POST':
            response = await api.post<T>(endpoint, newBody || body);
            break;
          case 'PUT':
            response = await api.put<T>(endpoint, newBody || body);
            break;
          case 'PATCH':
            response = await api.patch<T>(endpoint, newBody || body);
            break;
          case 'DELETE':
            response = await api.delete<T>(endpoint);
            break;
          default:
            throw new Error(`Unsupported method: ${method}`);
        }

        if (response.error) {
          setState({
            data: null,
            isLoading: false,
            error: response.error,
            status: response.status,
          });

          if (onError) {
            onError(response.error, response.status);
          }
        } else {
          setState({
            data: response.data,
            isLoading: false,
            error: null,
            status: response.status,
          });

          if (onSuccess && response.data) {
            onSuccess(response.data);
          }
        }

        return response;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';

        setState({
          data: null,
          isLoading: false,
          error: errorMessage,
          status: 500,
        });

        if (onError) {
          onError(errorMessage, 500);
        }

        return {
          data: null,
          error: errorMessage,
          status: 500,
        };
      }
    },
    [method, endpoint, body, params, useMock, onSuccess, onError]
  );

  // Izvedi API klic ob montiranju komponente
  useEffect(() => {
    if (!skip) {
      execute();
    }
  }, [skip, execute]);

  return { ...state, execute, setState };
}

// Specializirane funkcije za različne HTTP metode
export function useGet<T = unknown>(
  endpoint: string,
  params?: Record<string, string | number | boolean>,
  options?: ApiOptions<T>
) {
  return useApi<T>('GET', endpoint, undefined, params, options);
}

export function usePost<T = unknown>(
  endpoint: string,
  body?: unknown,
  options?: ApiOptions<T>
) {
  return useApi<T>('POST', endpoint, body, undefined, { ...options, skip: true });
}

export function usePut<T = unknown>(
  endpoint: string,
  body?: unknown,
  options?: ApiOptions<T>
) {
  return useApi<T>('PUT', endpoint, body, undefined, { ...options, skip: true });
}

export function usePatch<T = unknown>(
  endpoint: string,
  body?: unknown,
  options?: ApiOptions<T>
) {
  return useApi<T>('PATCH', endpoint, body, undefined, { ...options, skip: true });
}

export function useDelete<T = unknown>(
  endpoint: string,
  options?: ApiOptions<T>
) {
  return useApi<T>('DELETE', endpoint, undefined, undefined, { ...options, skip: true });
}

export default useApi;
