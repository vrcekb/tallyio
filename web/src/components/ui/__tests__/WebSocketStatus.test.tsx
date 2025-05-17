import React from 'react';
import { render } from '@testing-library/react';
import WebSocketStatus from '../WebSocketStatus';
import * as WebSocketContext from '../../../contexts/WebSocketContext';

// Preprost test za WebSocketStatus komponento
describe('WebSocketStatus', () => {
  // Mock za useWebSocket hook
  const mockUseWebSocket = jest.spyOn(WebSocketContext, 'useWebSocket');

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders without crashing', () => {
    // Mock useWebSocket hook
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      isAuthenticated: true,
      connect: jest.fn(),
      disconnect: jest.fn(),
      send: jest.fn(),
      lastMessage: null,
      systemStatus: null,
      chainStatus: null,
      transactions: [],
      strategies: [],
      rpcStatus: null,
      mempoolData: null
    });

    // Preveri, da se komponenta upodobi brez napak
    const { container } = render(<WebSocketStatus />);
    expect(container).toBeTruthy();
  });
});
