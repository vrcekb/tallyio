import { renderHook } from '@testing-library/react-hooks';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import useWorker, { MessageType } from '../useWorker';

// Mock za Worker
class MockWorker {
  private onMessageCallback: ((event: MessageEvent) => void) | null = null;

  constructor() {
    // Simuliraj, da je Worker pripravljen
    setTimeout(() => {
      this.postMessageToMain({ type: 'READY', data: null });
    }, 0);
  }

  addEventListener(event: string, callback: (event: MessageEvent) => void) {
    if (event === 'message') {
      this.onMessageCallback = callback;
    }
  }

  postMessage(message: any) {
    // Simuliraj obdelavo sporočila v Worker-ju
    setTimeout(() => {
      const { type, data, id } = message;

      // Simuliraj različne odgovore glede na tip sporočila
      switch (type) {
        case MessageType.PROCESS_TRANSACTIONS:
          this.postMessageToMain({
            type,
            data: {
              transactions: data.map((tx: any) => ({ ...tx, processed: true })),
              stats: { count: data.length }
            },
            id
          });
          break;
        case MessageType.CALCULATE_METRICS:
          this.postMessageToMain({
            type,
            data: { metrics: { cpu: 50, memory: 60, network: 70 } },
            id
          });
          break;
        case MessageType.FILTER_DATA:
          this.postMessageToMain({
            type,
            data: data.items.filter((item: any) => item.value > data.filters.minValue),
            id
          });
          break;
        case 'ERROR_TEST':
          this.postMessageToMain({
            type,
            data: null,
            error: 'Test error',
            id
          });
          break;
        default:
          this.postMessageToMain({
            type,
            data: { received: data },
            id
          });
      }
    }, 10);
  }

  terminate() {
    // Počisti
  }

  private postMessageToMain(data: any) {
    if (this.onMessageCallback) {
      this.onMessageCallback(new MessageEvent('message', { data }));
    }
  }
}

// Mock za globalni Worker konstruktor
// @ts-ignore
global.Worker = MockWorker;

describe('useWorker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize with default state', () => {
    const { result } = renderHook(() =>
      useWorker('/mock-worker.js', MessageType.PROCESS_TRANSACTIONS)
    );

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.data).toBeNull();
    expect(typeof result.current.execute).toBe('function');
  });

  it('should process data and update state', async () => {
    const { result, waitForNextUpdate } = renderHook(() =>
      useWorker('/mock-worker.js', MessageType.PROCESS_TRANSACTIONS)
    );

    const mockData = [
      { id: '1', value: 100 },
      { id: '2', value: 200 }
    ];

    // Izvedi operacijo
    let promise;
    promise = result.current.execute(mockData);

    // Preveri, ali je stanje nalaganja nastavljeno
    expect(result.current.loading).toBe(true);

    // Počakaj na posodobitev
    await waitForNextUpdate();

    // Preveri, ali je stanje posodobljeno
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.data).toEqual({
      transactions: [
        { id: '1', value: 100, processed: true },
        { id: '2', value: 200, processed: true }
      ],
      stats: { count: 2 }
    });

    // Preveri, ali je Promise razrešen s pravilnimi podatki
    const result2 = await promise;
    expect(result2).toEqual({
      transactions: [
        { id: '1', value: 100, processed: true },
        { id: '2', value: 200, processed: true }
      ],
      stats: { count: 2 }
    });
  });

  it('should handle errors', async () => {
    const { result, waitForNextUpdate } = renderHook(() =>
      useWorker('/mock-worker.js', 'ERROR_TEST' as MessageType)
    );

    // Izvedi operacijo, ki bo povzročila napako
    let promise;
    promise = result.current.execute({}).catch(error => {
      // Ujemi napako, da ne bo neobravnavana zavrnitev
      expect(error.message).toBe('Test error');
    });

    // Preveri, ali je stanje nalaganja nastavljeno
    expect(result.current.loading).toBe(true);

    // Počakaj na posodobitev
    await waitForNextUpdate();

    // Preveri, ali je stanje napake posodobljeno
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe('Test error');
    expect(result.current.data).toBeNull();

    // Počakaj, da se obljuba razreši
    await promise;
  });

  it('should handle different message types', async () => {
    const { result, waitForNextUpdate } = renderHook(() =>
      useWorker('/mock-worker.js', MessageType.CALCULATE_METRICS)
    );

    // Izvedi operacijo
    result.current.execute({});

    // Počakaj na posodobitev
    await waitForNextUpdate();

    // Preveri, ali je stanje posodobljeno
    expect(result.current.data).toEqual({
      metrics: { cpu: 50, memory: 60, network: 70 }
    });
  });
});
