import React, { useState, useEffect, lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import SkipToContent from './components/ui/SkipToContent';
import ProtectedRoute from './components/auth/ProtectedRoute';
import { TimeRange } from './types';
import { preloadRoute } from './utils/preloadRoutes';
import { WebSocketProvider } from './contexts/WebSocketContext';
import mockWebSocketServer from './services/mockWebSocketServer';
import { ThemeProvider, useTheme } from './theme/ThemeContext';
import ThemeVariables from './theme/ThemeVariables';
import ThemeNameSelector from './theme/ThemeNameSelector';
import { AuthProvider } from './contexts/AuthContext';
import ToastProvider from './contexts/ToastContext';
import { initAccessibility } from './utils/accessibility';
import './styles/accessibility.css';

// Lazy loading za vse strani
const OverviewDashboard = lazy(() => import('./components/dashboard/OverviewDashboard'));
const BlockchainPage = lazy(() => import('./pages/BlockchainPage'));
const BusinessPage = lazy(() => import('./pages/BusinessPage'));
const PerformancePage = lazy(() => import('./pages/PerformancePage'));
const StrategiesPage = lazy(() => import('./pages/StrategiesPage'));
const RiskManagementPage = lazy(() => import('./pages/RiskManagementPage'));
const MLPage = lazy(() => import('./pages/MLPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const DexManagementPage = lazy(() => import('./pages/DexManagementPage'));
const ProtocolManagementPage = lazy(() => import('./pages/ProtocolManagementPage'));
const WalletPage = lazy(() => import('./pages/WalletPage'));
const SmartContractsPage = lazy(() => import('./pages/SmartContractsPage'));

// Avtentikacijske strani
const LoginPage = lazy(() => import('./pages/LoginPage'));
const RegisterPage = lazy(() => import('./pages/RegisterPage'));
const ForgotPasswordPage = lazy(() => import('./pages/ForgotPasswordPage'));
const ProfilePage = lazy(() => import('./pages/ProfilePage'));

// Komponenta za prikaz nalaganja
const LoadingPage = () => (
  <div className="flex justify-center items-center h-[80vh]">
    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
  </div>
);

// Komponenta za predhodno nalaganje strani
const RoutePreloader = () => {
  const location = useLocation();

  // Predhodno naloži naslednjo stran, ko se uporabnik premakne z miško nad povezavo
  useEffect(() => {
    const handleLinkHover = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'A' && target.getAttribute('href')) {
        const href = target.getAttribute('href') as string;
        if (href.startsWith('/')) {
          preloadRoute(href);
        }
      }
    };

    document.addEventListener('mouseover', handleLinkHover);
    return () => {
      document.removeEventListener('mouseover', handleLinkHover);
    };
  }, []);

  // Predhodno naloži trenutno stran
  useEffect(() => {
    preloadRoute(location.pathname);
  }, [location.pathname]);

  return null;
};

function App() {
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Initialize dark mode based on user preference
  useEffect(() => {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setIsDarkMode(false); // Set to false to make light mode default
    }

    // Listen for changes in color scheme preference
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      setIsDarkMode(false); // Keep light mode as default even when system preference changes
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, []);

  // Apply dark mode class to document
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  // Predhodno naloži začetno stran
  useEffect(() => {
    preloadRoute('/');
    preloadRoute('/blockchain');
  }, []);

  // Inicializacija mock WebSocket strežnika v razvojnem okolju
  useEffect(() => {
    if (import.meta.env.DEV) {
      // Zaženi mock WebSocket strežnik
      mockWebSocketServer.start();

      return () => {
        // Ustavi mock WebSocket strežnik ob odmontiranju komponente
        mockWebSocketServer.stop();
      };
    }
  }, []);

  // Inicializacija dostopnosti
  useEffect(() => {
    initAccessibility();
  }, []);

  return (
    <ThemeProvider defaultMode="light" defaultTheme="winter-morning">
      <ThemeVariables />
      <AuthProvider>
        <ToastProvider>
          <WebSocketProvider autoConnect={true} url="ws://localhost:8080/ws">
            <Router>
              <div className="min-h-screen bg-gray-50 dark:bg-dark-background">
                <RoutePreloader />
              <Routes>
                {/* Javne poti */}
                <Route path="/login" element={
                  <Suspense fallback={<LoadingPage />}>
                    <LoginPage />
                  </Suspense>
                } />
                <Route path="/register" element={
                  <Suspense fallback={<LoadingPage />}>
                    <RegisterPage />
                  </Suspense>
                } />
                <Route path="/forgot-password" element={
                  <Suspense fallback={<LoadingPage />}>
                    <ForgotPasswordPage />
                  </Suspense>
                } />

                {/* Zaščitene poti */}
                <Route path="*" element={
                  <ProtectedRoute>
                    <>
                      <Sidebar
                        isOpen={isSidebarOpen}
                        onClose={() => setIsSidebarOpen(false)}
                      />

                      <SkipToContent />
                      <div className={`min-h-screen transition-all duration-300 ${isSidebarOpen ? 'lg:ml-64' : 'lg:ml-0'}`}>
                        <Header
                          timeRange={timeRange}
                          onTimeRangeChange={setTimeRange}
                          isDarkMode={isDarkMode}
                          toggleDarkMode={toggleDarkMode}
                          toggleSidebar={toggleSidebar}
                        />

                        <main id="main-content" className="p-6" tabIndex={-1}>
                          <Suspense fallback={<LoadingPage />}>
                            <Routes>
                              <Route path="/" element={<OverviewDashboard timeRange={timeRange} onTimeRangeChange={setTimeRange} />} />
                              <Route path="/business" element={<BusinessPage />} />
                              <Route path="/performance" element={<PerformancePage />} />
                              <Route path="/blockchain" element={<BlockchainPage />} />
                              <Route path="/strategies" element={<StrategiesPage />} />
                              <Route path="/risk" element={<RiskManagementPage />} />
                              <Route path="/ml" element={<MLPage />} />
                              <Route path="/settings" element={<SettingsPage />} />
                              <Route path="/dex" element={<DexManagementPage />} />
                              <Route path="/protocols" element={<ProtocolManagementPage />} />
                              <Route path="/wallet" element={<WalletPage />} />
                              <Route path="/contracts" element={<SmartContractsPage />} />
                              <Route path="/profile" element={<ProfilePage />} />
                            </Routes>
                          </Suspense>
                        </main>
                      </div>
                    </>
                  </ProtectedRoute>
                } />
              </Routes>
            </div>
          </Router>
        </WebSocketProvider>
        </ToastProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;