import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock za URL.createObjectURL
window.URL.createObjectURL = vi.fn(() => 'mock-url');

// Mock za URL.revokeObjectURL
window.URL.revokeObjectURL = vi.fn();

// Mock za matchMedia
window.matchMedia = vi.fn().mockImplementation((query) => ({
  matches: false,
  media: query,
  onchange: null,
  addListener: vi.fn(),
  removeListener: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  dispatchEvent: vi.fn(),
}));

// Mock za ResizeObserver
class ResizeObserverMock {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}

window.ResizeObserver = ResizeObserverMock;

// Mock za IntersectionObserver
class IntersectionObserverMock {
  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback;
  }

  callback: IntersectionObserverCallback;
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}

window.IntersectionObserver = IntersectionObserverMock;

// Mock za Web Worker
class WorkerMock {
  constructor(stringUrl: string) {
    this.url = stringUrl;
  }

  url: string;
  addEventListener = vi.fn();
  removeEventListener = vi.fn();
  postMessage = vi.fn();
  terminate = vi.fn();
}

window.Worker = WorkerMock;

// Mock za WebSocket
class WebSocketMock {
  constructor(url: string) {
    this.url = url;
  }

  url: string;
  CONNECTING = 0;
  OPEN = 1;
  CLOSING = 2;
  CLOSED = 3;
  readyState = this.CONNECTING;
  addEventListener = vi.fn();
  removeEventListener = vi.fn();
  send = vi.fn();
  close = vi.fn();
}

window.WebSocket = WebSocketMock;

// Mock za localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value.toString();
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Mock za sessionStorage
const sessionStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value.toString();
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'sessionStorage', {
  value: sessionStorageMock,
});

// Mock za navigator.clipboard
Object.defineProperty(navigator, 'clipboard', {
  value: {
    writeText: vi.fn(),
    readText: vi.fn(),
  },
});

// Mock za navigator.geolocation
Object.defineProperty(navigator, 'geolocation', {
  value: {
    getCurrentPosition: vi.fn(),
    watchPosition: vi.fn(),
    clearWatch: vi.fn(),
  },
});

// Mock za navigator.mediaDevices
Object.defineProperty(navigator, 'mediaDevices', {
  value: {
    getUserMedia: vi.fn(),
    enumerateDevices: vi.fn(),
    getDisplayMedia: vi.fn(),
  },
});

// Mock za navigator.serviceWorker
Object.defineProperty(navigator, 'serviceWorker', {
  value: {
    register: vi.fn(),
    getRegistration: vi.fn(),
    getRegistrations: vi.fn(),
    ready: Promise.resolve({
      active: {
        postMessage: vi.fn(),
      },
    }),
  },
});

// Mock za window.scrollTo
window.scrollTo = vi.fn();

// Mock za window.fetch
window.fetch = vi.fn();

// Mock za console.error
console.error = vi.fn();

// Mock za console.warn
console.warn = vi.fn();

// Mock za console.info
console.info = vi.fn();

// Mock za console.debug
console.debug = vi.fn();
