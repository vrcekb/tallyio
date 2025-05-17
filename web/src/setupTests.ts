// Dodaj React Testing Library
import '@testing-library/jest-dom';

// Polnila za Web API-je, ki jih JSDOM ne implementira
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // Deprecated
    removeListener: jest.fn(), // Deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Polnilo za IntersectionObserver
class MockIntersectionObserver {
  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback;
  }

  observe = jest.fn();
  unobserve = jest.fn();
  disconnect = jest.fn();
  callback: IntersectionObserverCallback;
}

Object.defineProperty(window, 'IntersectionObserver', {
  writable: true,
  value: MockIntersectionObserver,
});

// Polnilo za localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Polnilo za sessionStorage
const sessionStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'sessionStorage', {
  value: sessionStorageMock,
});

// Polnilo za URL
class MockURL {
  constructor(url: string, _base?: string) {
    this.href = url;
    this.pathname = url.split('?')[0];
    this.searchParams = {
      append: jest.fn(),
      get: jest.fn(),
      has: jest.fn(),
      set: jest.fn(),
      delete: jest.fn(),
      toString: jest.fn().mockReturnValue(''),
    };
  }

  href: string;
  pathname: string;
  searchParams: {
    append: jest.fn as any,
    get: jest.fn as any,
    has: jest.fn as any,
    set: jest.fn as any,
    delete: jest.fn as any,
    toString: jest.fn as any
  };

  toString() {
    return this.href;
  }
}

Object.defineProperty(global, 'URL', {
  value: MockURL,
});

// Polnilo za fetch
global.fetch = jest.fn().mockImplementation(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
    blob: () => Promise.resolve(new Blob()),
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
    formData: () => Promise.resolve(new FormData()),
    headers: {
      get: jest.fn(),
    },
    status: 200,
    statusText: 'OK',
  })
);

// Polnilo za WebSocket
class MockWebSocket {
  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;
    this.readyState = MockWebSocket.CONNECTING;
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  protocols?: string | string[];
  readyState: number;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  send = jest.fn();
  close = jest.fn().mockImplementation(() => {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  });
}

Object.defineProperty(global, 'WebSocket', {
  value: MockWebSocket,
});

// Polnilo za ResizeObserver
class MockResizeObserver {
  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
  }

  observe = jest.fn();
  unobserve = jest.fn();
  disconnect = jest.fn();
  callback: ResizeObserverCallback;
}

Object.defineProperty(global, 'ResizeObserver', {
  value: MockResizeObserver,
});

// Polnilo za Worker
class MockWorker {
  constructor(stringUrl: string) {
    this.url = stringUrl;
  }

  url: string;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  postMessage = jest.fn().mockImplementation((data) => {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }));
    }
  });
  terminate = jest.fn();
}

Object.defineProperty(global, 'Worker', {
  value: MockWorker,
});

// Polnilo za console.error, da ne onesnažuje izpisa testov
const originalConsoleError = console.error;
console.error = (...args: unknown[]) => {
  if (
    typeof args[0] === 'string' &&
    args[0].includes('Warning: ReactDOM.render is no longer supported')
  ) {
    return;
  }
  originalConsoleError(...args);
};
