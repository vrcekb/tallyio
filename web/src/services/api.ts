/**
 * API servis za komunikacijo z backend API-ji
 */

// Vmesnik za možnosti zahteve
interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  body?: unknown;
  params?: Record<string, string | number | boolean>;
  timeout?: number;
  cache?: RequestCache;
}

// Vmesnik za odgovor
interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  status: number;
}

// Privzete možnosti
const DEFAULT_OPTIONS: RequestOptions = {
  method: 'GET',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 sekund
  cache: 'default',
};

// Razred za API servis
class ApiService {
  private baseUrl: string;
  private defaultOptions: RequestOptions;

  constructor(baseUrl: string, defaultOptions: RequestOptions = {}) {
    this.baseUrl = baseUrl;
    this.defaultOptions = { ...DEFAULT_OPTIONS, ...defaultOptions };
  }

  /**
   * Nastavi avtentikacijski token
   * @param token Avtentikacijski token
   */
  public setAuthToken(token: string | null): void {
    if (token) {
      this.defaultOptions.headers = {
        ...this.defaultOptions.headers,
        Authorization: `Bearer ${token}`,
      };
    } else if (this.defaultOptions.headers?.Authorization) {
      // Uporabimo destrukturiranje, da odstranimo Authorization iz headers
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { Authorization, ...headers } = this.defaultOptions.headers;
      this.defaultOptions.headers = headers;
    }
  }

  /**
   * Izvede zahtevo na API
   * @param endpoint Končna točka API-ja
   * @param options Možnosti zahteve
   * @returns Odgovor API-ja
   */
  public async request<T>(endpoint: string, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    const url = this.buildUrl(endpoint, options.params);
    const mergedOptions = this.mergeOptions(options);

    try {
      // Ustvari AbortController za timeout
      const controller = new AbortController();
      const { signal } = controller;

      // Nastavi timeout
      const timeoutId = setTimeout(() => {
        controller.abort();
      }, mergedOptions.timeout);

      // Izvedi zahtevo
      const response = await fetch(url, {
        ...mergedOptions,
        signal,
        body: mergedOptions.body ? JSON.stringify(mergedOptions.body) : undefined,
      });

      // Počisti timeout
      clearTimeout(timeoutId);

      // Preveri, ali je zahteva uspela
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return {
          data: null,
          error: errorData.message || `Request failed with status ${response.status}`,
          status: response.status,
        };
      }

      // Preveri, ali je odgovor prazen
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        return {
          data: null,
          error: null,
          status: response.status,
        };
      }

      // Preberi odgovor
      const data = await response.json();

      return {
        data,
        error: null,
        status: response.status,
      };
    } catch (error) {
      // Obravnavaj napake
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          return {
            data: null,
            error: 'Request timed out',
            status: 408,
          };
        }

        return {
          data: null,
          error: error.message,
          status: 500,
        };
      }

      return {
        data: null,
        error: 'Unknown error',
        status: 500,
      };
    }
  }

  /**
   * Izvede GET zahtevo
   * @param endpoint Končna točka API-ja
   * @param params Parametri poizvedbe
   * @param options Dodatne možnosti
   * @returns Odgovor API-ja
   */
  public async get<T>(
    endpoint: string,
    params?: Record<string, string | number | boolean>,
    options?: Omit<RequestOptions, 'method' | 'params'>
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'GET',
      params,
    });
  }

  /**
   * Izvede POST zahtevo
   * @param endpoint Končna točka API-ja
   * @param body Telo zahteve
   * @param options Dodatne možnosti
   * @returns Odgovor API-ja
   */
  public async post<T>(
    endpoint: string,
    body?: unknown,
    options?: Omit<RequestOptions, 'method' | 'body'>
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body,
    });
  }

  /**
   * Izvede PUT zahtevo
   * @param endpoint Končna točka API-ja
   * @param body Telo zahteve
   * @param options Dodatne možnosti
   * @returns Odgovor API-ja
   */
  public async put<T>(
    endpoint: string,
    body?: unknown,
    options?: Omit<RequestOptions, 'method' | 'body'>
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body,
    });
  }

  /**
   * Izvede PATCH zahtevo
   * @param endpoint Končna točka API-ja
   * @param body Telo zahteve
   * @param options Dodatne možnosti
   * @returns Odgovor API-ja
   */
  public async patch<T>(
    endpoint: string,
    body?: unknown,
    options?: Omit<RequestOptions, 'method' | 'body'>
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PATCH',
      body,
    });
  }

  /**
   * Izvede DELETE zahtevo
   * @param endpoint Končna točka API-ja
   * @param options Dodatne možnosti
   * @returns Odgovor API-ja
   */
  public async delete<T>(
    endpoint: string,
    options?: Omit<RequestOptions, 'method'>
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'DELETE',
    });
  }

  /**
   * Zgradi URL za zahtevo
   * @param endpoint Končna točka API-ja
   * @param params Parametri poizvedbe
   * @returns URL za zahtevo
   */
  private buildUrl(endpoint: string, params?: Record<string, string | number | boolean>): string {
    const url = new URL(`${this.baseUrl}/${endpoint.startsWith('/') ? endpoint.slice(1) : endpoint}`);

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value));
        }
      });
    }

    return url.toString();
  }

  /**
   * Združi privzete možnosti z možnostmi zahteve
   * @param options Možnosti zahteve
   * @returns Združene možnosti
   */
  private mergeOptions(options: RequestOptions): RequestOptions {
    return {
      ...this.defaultOptions,
      ...options,
      headers: {
        ...this.defaultOptions.headers,
        ...options.headers,
      },
    };
  }
}

// Ustvari singleton instanco
const apiService = new ApiService(
  import.meta.env.VITE_API_URL || 'http://localhost:8080/api'
);

export default apiService;
