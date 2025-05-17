/**
 * Mock API servis za simulacijo API klicev v razvojnem okolju
 */

import {
  systemHealthStatus,
  chainStatus,
  strategies,
  generateRecentTransactions,
  keyMetrics,
  rpcLocations
} from '../mockData';

// Vmesnik za odgovor
interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  status: number;
}

// Vmesnik za možnosti zahteve
interface RequestOptions {
  delay?: number;
  shouldFail?: boolean;
  errorMessage?: string;
  errorStatus?: number;
}

// Privzete možnosti
const DEFAULT_OPTIONS: RequestOptions = {
  delay: 500,
  shouldFail: false,
  errorMessage: 'An error occurred',
  errorStatus: 500,
};

// Razred za mock API servis
class MockApiService {
  private options: RequestOptions;
  private endpoints: Record<string, unknown>;

  constructor(options: RequestOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };

    // Definiraj končne točke in njihove odgovore
    this.endpoints = {
      // Sistemski status
      'system/health': systemHealthStatus,

      // Blockchain status
      'blockchain/status': chainStatus,

      // Strategije
      'strategies': strategies,
      'strategies/active': strategies.filter(s => s.status === 'active'),

      // Transakcije
      'transactions/recent': generateRecentTransactions(20),

      // Metrike
      'metrics/key': keyMetrics,

      // RPC lokacije
      'rpc/locations': rpcLocations,
    };
  }

  /**
   * Simulira API klic
   * @param endpoint Končna točka API-ja
   * @param options Možnosti zahteve
   * @returns Odgovor API-ja
   */
  public async request<T>(endpoint: string, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    const mergedOptions = { ...this.options, ...options };

    return new Promise((resolve) => {
      setTimeout(() => {
        // Če naj zahteva ne uspe
        if (mergedOptions.shouldFail) {
          resolve({
            data: null,
            error: mergedOptions.errorMessage || 'An error occurred',
            status: mergedOptions.errorStatus || 500,
          });
          return;
        }

        // Preveri, ali končna točka obstaja
        const normalizedEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
        const data = this.endpoints[normalizedEndpoint];

        if (data === undefined) {
          resolve({
            data: null,
            error: `Endpoint ${endpoint} not found`,
            status: 404,
          });
          return;
        }

        // Vrni podatke
        resolve({
          data: typeof data === 'function' ? data() : data as T,
          error: null,
          status: 200,
        });
      }, mergedOptions.delay);
    });
  }

  /**
   * Simulira GET zahtevo
   * @param endpoint Končna točka API-ja
   * @param options Možnosti zahteve
   * @returns Odgovor API-ja
   */
  public async get<T>(endpoint: string, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, options);
  }

  /**
   * Simulira POST zahtevo
   * @param endpoint Končna točka API-ja
   * @param body Telo zahteve
   * @param options Možnosti zahteve
   * @returns Odgovor API-ja
   */
  public async post<T>(endpoint: string, body: unknown = {}, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    console.log(`POST ${endpoint}`, body);
    return this.request<T>(endpoint, options);
  }

  /**
   * Simulira PUT zahtevo
   * @param endpoint Končna točka API-ja
   * @param body Telo zahteve
   * @param options Možnosti zahteve
   * @returns Odgovor API-ja
   */
  public async put<T>(endpoint: string, body: unknown = {}, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    console.log(`PUT ${endpoint}`, body);
    return this.request<T>(endpoint, options);
  }

  /**
   * Simulira PATCH zahtevo
   * @param endpoint Končna točka API-ja
   * @param body Telo zahteve
   * @param options Možnosti zahteve
   * @returns Odgovor API-ja
   */
  public async patch<T>(endpoint: string, body: unknown = {}, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    console.log(`PATCH ${endpoint}`, body);
    return this.request<T>(endpoint, options);
  }

  /**
   * Simulira DELETE zahtevo
   * @param endpoint Končna točka API-ja
   * @param options Možnosti zahteve
   * @returns Odgovor API-ja
   */
  public async delete<T>(endpoint: string, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    console.log(`DELETE ${endpoint}`);
    return this.request<T>(endpoint, options);
  }

  /**
   * Registrira novo končno točko
   * @param endpoint Končna točka API-ja
   * @param data Podatki za končno točko
   */
  public registerEndpoint(endpoint: string, data: unknown): void {
    const normalizedEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
    this.endpoints[normalizedEndpoint] = data;
  }

  /**
   * Odstrani končno točko
   * @param endpoint Končna točka API-ja
   */
  public removeEndpoint(endpoint: string): void {
    const normalizedEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
    delete this.endpoints[normalizedEndpoint];
  }

  /**
   * Nastavi možnosti za vse zahteve
   * @param options Možnosti zahteve
   */
  public setOptions(options: RequestOptions): void {
    this.options = { ...this.options, ...options };
  }
}

// Ustvari singleton instanco
const mockApiService = new MockApiService();

export default mockApiService;
