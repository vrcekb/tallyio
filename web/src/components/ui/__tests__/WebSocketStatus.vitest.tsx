import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import WebSocketStatus from '../WebSocketStatus';
import * as WebSocketContext from '../../../contexts/WebSocketContext';

describe('WebSocketStatus', () => {
  // Mock za useWebSocket hook
  const mockUseWebSocket = vi.spyOn(WebSocketContext, 'useWebSocket');
  
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders connected state with text', () => {
    // Mock useWebSocket hook za prikaz povezanega stanja
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      isAuthenticated: true,
      connect: vi.fn(),
      disconnect: vi.fn(),
      send: vi.fn(),
      lastMessage: null,
      systemStatus: null,
      chainStatus: null,
      transactions: [],
      strategies: [],
      rpcStatus: null,
      mempoolData: null
    });

    render(<WebSocketStatus />);
    
    // Preveri, da je prikazana ikona in besedilo za povezano stanje
    expect(screen.getByTitle('WebSocket povezava je aktivna')).toBeInTheDocument();
    expect(screen.getByText('Povezano')).toBeInTheDocument();
  });

  it('renders disconnected state with text', () => {
    // Mock useWebSocket hook za prikaz nepovezanega stanja
    mockUseWebSocket.mockReturnValue({
      isConnected: false,
      isAuthenticated: false,
      connect: vi.fn(),
      disconnect: vi.fn(),
      send: vi.fn(),
      lastMessage: null,
      systemStatus: null,
      chainStatus: null,
      transactions: [],
      strategies: [],
      rpcStatus: null,
      mempoolData: null
    });

    render(<WebSocketStatus />);
    
    // Preveri, da je prikazana ikona in besedilo za nepovezano stanje
    expect(screen.getByTitle('WebSocket povezava ni aktivna')).toBeInTheDocument();
    expect(screen.getByText('Ni povezave')).toBeInTheDocument();
  });

  it('renders connected state without text', () => {
    // Mock useWebSocket hook za prikaz povezanega stanja
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      isAuthenticated: true,
      connect: vi.fn(),
      disconnect: vi.fn(),
      send: vi.fn(),
      lastMessage: null,
      systemStatus: null,
      chainStatus: null,
      transactions: [],
      strategies: [],
      rpcStatus: null,
      mempoolData: null
    });

    render(<WebSocketStatus showText={false} />);
    
    // Preveri, da je prikazana ikona, vendar ne besedilo za povezano stanje
    expect(screen.getByTitle('WebSocket povezava je aktivna')).toBeInTheDocument();
    expect(screen.queryByText('Povezano')).not.toBeInTheDocument();
  });

  it('renders disconnected state without text', () => {
    // Mock useWebSocket hook za prikaz nepovezanega stanja
    mockUseWebSocket.mockReturnValue({
      isConnected: false,
      isAuthenticated: false,
      connect: vi.fn(),
      disconnect: vi.fn(),
      send: vi.fn(),
      lastMessage: null,
      systemStatus: null,
      chainStatus: null,
      transactions: [],
      strategies: [],
      rpcStatus: null,
      mempoolData: null
    });

    render(<WebSocketStatus showText={false} />);
    
    // Preveri, da je prikazana ikona, vendar ne besedilo za nepovezano stanje
    expect(screen.getByTitle('WebSocket povezava ni aktivna')).toBeInTheDocument();
    expect(screen.queryByText('Ni povezave')).not.toBeInTheDocument();
  });

  it('applies custom className', () => {
    // Mock useWebSocket hook
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      isAuthenticated: true,
      connect: vi.fn(),
      disconnect: vi.fn(),
      send: vi.fn(),
      lastMessage: null,
      systemStatus: null,
      chainStatus: null,
      transactions: [],
      strategies: [],
      rpcStatus: null,
      mempoolData: null
    });

    render(<WebSocketStatus className="custom-class" />);
    
    // Preveri, da je uporabljen razred
    expect(screen.getByTitle('WebSocket povezava je aktivna').className).toContain('custom-class');
  });
});
