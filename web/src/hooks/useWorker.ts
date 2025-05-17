import { useState, useEffect, useRef, useCallback } from 'react';

// Tipi sporočil, ki jih lahko prejme WebWorker
export enum MessageType {
  PROCESS_TRANSACTIONS = 'PROCESS_TRANSACTIONS',
  CALCULATE_METRICS = 'CALCULATE_METRICS',
  ANALYZE_STRATEGY = 'ANALYZE_STRATEGY',
  FILTER_DATA = 'FILTER_DATA',
  SORT_DATA = 'SORT_DATA'
}

// Vmesnik za sporočilo
interface WorkerMessage<TData = unknown> {
  type: MessageType;
  data: TData;
  id: string;
}

// Vmesnik za odgovor
interface WorkerResponse<TResult = unknown> {
  type: MessageType;
  data: TResult;
  id: string;
  error?: string;
}

// Vmesnik za stanje
interface WorkerState<TResult = unknown> {
  loading: boolean;
  error: string | null;
  data: TResult | null;
}

// Vmesnik za hook
interface UseWorkerReturn<TResult, TData = unknown> {
  loading: boolean;
  error: string | null;
  data: TResult | null;
  execute: (data: TData) => Promise<TResult>;
}

/**
 * Hook za uporabo WebWorker-jev
 * @param workerPath Pot do WebWorker datoteke
 * @param messageType Tip sporočila
 * @returns Objekt z loading, error, data in execute funkcijo
 */
function useWorker<TResult, TData = unknown>(
  workerPath: string,
  messageType: MessageType
): UseWorkerReturn<TResult, TData> {
  const [state, setState] = useState<WorkerState<TResult>>({
    loading: false,
    error: null,
    data: null
  });

  const workerRef = useRef<Worker | null>(null);
  const callbacksRef = useRef<Map<string, (response: WorkerResponse<TResult>) => void>>(new Map());

  // Ustvari WebWorker
  useEffect(() => {
    // Ustvari WebWorker
    workerRef.current = new Worker(workerPath, { type: 'module' });

    // Nastavi poslušalca za sporočila
    workerRef.current.addEventListener('message', (event: MessageEvent<WorkerResponse<TResult>>) => {
      const response = event.data;
      const callback = callbacksRef.current.get(response.id);

      if (callback) {
        callback(response);
        callbacksRef.current.delete(response.id);
      }
    });

    // Počisti WebWorker ob odmontiranju komponente
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [workerPath]);

  // Funkcija za pošiljanje sporočila WebWorker-ju
  const execute = useCallback(
    (data: TData): Promise<TResult> => {
      return new Promise((resolve, reject) => {
        if (!workerRef.current) {
          reject(new Error('WebWorker ni na voljo'));
          return;
        }

        setState(prev => ({ ...prev, loading: true, error: null }));

        // Ustvari ID za sporočilo
        const id = Math.random().toString(36).substring(2, 9);

        // Nastavi callback za odgovor
        callbacksRef.current.set(id, (response: WorkerResponse<TResult>) => {
          if (response.error) {
            setState(prev => ({
              ...prev,
              loading: false,
              error: response.error || 'Neznana napaka'
            }));
            reject(new Error(response.error));
          } else {
            setState(prev => ({
              ...prev,
              loading: false,
              data: response.data
            }));
            resolve(response.data);
          }
        });

        // Pošlji sporočilo WebWorker-ju
        const message: WorkerMessage<TData> = {
          type: messageType,
          data,
          id
        };

        workerRef.current.postMessage(message);
      });
    },
    [messageType]
  );

  return {
    loading: state.loading,
    error: state.error,
    data: state.data,
    execute
  };
}

export default useWorker;
