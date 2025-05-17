import { renderHook, act } from '@testing-library/react-hooks';
import useApi, { useGet, usePost, usePut, usePatch, useDelete } from '../useApi';
// apiService je importiran, vendar ni uporabljen, ker je mockiran
import _apiService from '../../services/api';
import mockApiService from '../../services/mockApi';

// Mock za apiService in mockApiService
jest.mock('../../services/api', () => ({
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  patch: jest.fn(),
  delete: jest.fn(),
}));

jest.mock('../../services/mockApi', () => ({
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  patch: jest.fn(),
  delete: jest.fn(),
}));

describe('useApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should call the API with the correct parameters', async () => {
    mockApiService.get.mockResolvedValueOnce({
      data: { test: 'data' },
      error: null,
      status: 200,
    });

    const { result, waitForNextUpdate } = renderHook(() =>
      useApi('GET', 'test-endpoint', undefined, { param1: 'value1' }, { useMock: true })
    );

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBe(null);
    expect(result.current.error).toBe(null);
    expect(result.current.status).toBe(null);

    await waitForNextUpdate();

    expect(mockApiService.get).toHaveBeenCalledWith('test-endpoint', { param1: 'value1' });
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toEqual({ test: 'data' });
    expect(result.current.error).toBe(null);
    expect(result.current.status).toBe(200);
  });

  it('should handle API errors', async () => {
    mockApiService.get.mockResolvedValueOnce({
      data: null,
      error: 'API Error',
      status: 500,
    });

    const { result, waitForNextUpdate } = renderHook(() =>
      useApi('GET', 'test-endpoint', undefined, undefined, { useMock: true })
    );

    expect(result.current.isLoading).toBe(true);

    await waitForNextUpdate();

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBe(null);
    expect(result.current.error).toBe('API Error');
    expect(result.current.status).toBe(500);
  });

  it('should handle exceptions', async () => {
    mockApiService.get.mockRejectedValueOnce(new Error('Network Error'));

    const { result, waitForNextUpdate } = renderHook(() =>
      useApi('GET', 'test-endpoint', undefined, undefined, { useMock: true })
    );

    expect(result.current.isLoading).toBe(true);

    await waitForNextUpdate();

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBe(null);
    expect(result.current.error).toBe('Network Error');
    expect(result.current.status).toBe(500);
  });

  it('should call onSuccess callback when API call succeeds', async () => {
    const onSuccess = jest.fn();
    mockApiService.get.mockResolvedValueOnce({
      data: { test: 'data' },
      error: null,
      status: 200,
    });

    const { waitForNextUpdate } = renderHook(() =>
      useApi('GET', 'test-endpoint', undefined, undefined, { useMock: true, onSuccess })
    );

    await waitForNextUpdate();

    expect(onSuccess).toHaveBeenCalledWith({ test: 'data' });
  });

  it('should call onError callback when API call fails', async () => {
    const onError = jest.fn();
    mockApiService.get.mockResolvedValueOnce({
      data: null,
      error: 'API Error',
      status: 500,
    });

    const { waitForNextUpdate } = renderHook(() =>
      useApi('GET', 'test-endpoint', undefined, undefined, { useMock: true, onError })
    );

    await waitForNextUpdate();

    expect(onError).toHaveBeenCalledWith('API Error', 500);
  });

  it('should not call API when skip is true', async () => {
    const { result } = renderHook(() =>
      useApi('GET', 'test-endpoint', undefined, undefined, { useMock: true, skip: true })
    );

    expect(result.current.isLoading).toBe(false);
    expect(mockApiService.get).not.toHaveBeenCalled();
  });

  it('should execute API call when execute is called', async () => {
    mockApiService.post.mockResolvedValueOnce({
      data: { test: 'data' },
      error: null,
      status: 200,
    });

    const { result, waitForNextUpdate } = renderHook(() =>
      useApi('POST', 'test-endpoint', { initialBody: 'value' }, undefined, { useMock: true, skip: true })
    );

    expect(result.current.isLoading).toBe(false);
    expect(mockApiService.post).not.toHaveBeenCalled();

    act(() => {
      result.current.execute({ newBody: 'value' });
    });

    expect(result.current.isLoading).toBe(true);

    await waitForNextUpdate();

    expect(mockApiService.post).toHaveBeenCalledWith('test-endpoint', { newBody: 'value' });
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toEqual({ test: 'data' });
  });
});

describe('useGet', () => {
  it('should call useApi with GET method', () => {
    mockApiService.get.mockResolvedValueOnce({
      data: { test: 'data' },
      error: null,
      status: 200,
    });

    const { result } = renderHook(() =>
      useGet('test-endpoint', { param1: 'value1' }, { useMock: true })
    );

    expect(result.current.isLoading).toBe(true);
    expect(mockApiService.get).toHaveBeenCalledWith('test-endpoint', { param1: 'value1' });
  });
});

describe('usePost', () => {
  it('should call useApi with POST method and skip=true', () => {
    const { result } = renderHook(() =>
      usePost('test-endpoint', { body: 'value' }, { useMock: true })
    );

    expect(result.current.isLoading).toBe(false);
    expect(mockApiService.post).not.toHaveBeenCalled();
  });
});

describe('usePut', () => {
  it('should call useApi with PUT method and skip=true', () => {
    const { result } = renderHook(() =>
      usePut('test-endpoint', { body: 'value' }, { useMock: true })
    );

    expect(result.current.isLoading).toBe(false);
    expect(mockApiService.put).not.toHaveBeenCalled();
  });
});

describe('usePatch', () => {
  it('should call useApi with PATCH method and skip=true', () => {
    const { result } = renderHook(() =>
      usePatch('test-endpoint', { body: 'value' }, { useMock: true })
    );

    expect(result.current.isLoading).toBe(false);
    expect(mockApiService.patch).not.toHaveBeenCalled();
  });
});

describe('useDelete', () => {
  it('should call useApi with DELETE method and skip=true', () => {
    const { result } = renderHook(() =>
      useDelete('test-endpoint', { useMock: true })
    );

    expect(result.current.isLoading).toBe(false);
    expect(mockApiService.delete).not.toHaveBeenCalled();
  });
});
