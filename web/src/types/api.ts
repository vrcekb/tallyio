/**
 * Tipi za API komunikacijo
 */

// Odgovor API-ja
export interface ApiResponse<T = unknown> {
  data: T | null;
  error: string | null;
  status: number;
}

// Možnosti zahteve
export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  body?: unknown;
  params?: Record<string, string | number | boolean>;
  timeout?: number;
  cache?: RequestCache;
  shouldFail?: boolean;
  errorMessage?: string;
  errorStatus?: number;
  delay?: number;
}

// Možnosti API klica
export interface ApiOptions<T> {
  skip?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (error: string, status: number) => void;
  useMock?: boolean;
}
