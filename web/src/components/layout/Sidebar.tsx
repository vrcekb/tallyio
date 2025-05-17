import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Home,
  BarChart2,
  Terminal,
  Database,
  Zap,
  AlertTriangle,
  Brain,
  Repeat,
  BookOpen,
  Wallet,
  Code2
} from 'lucide-react';
import { preloadRoute } from '../../utils/preloadRoutes';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const navigationItems = [
    { name: 'Overview', icon: <Home size={20} />, path: '/' },
    { name: 'Business', icon: <BarChart2 size={20} />, path: '/business' },
    { name: 'Performance', icon: <Terminal size={20} />, path: '/performance' },
    { name: 'Blockchain', icon: <Database size={20} />, path: '/blockchain' },
    { name: 'Strategies', icon: <Zap size={20} />, path: '/strategies' },
    { name: 'Risk Management', icon: <AlertTriangle size={20} />, path: '/risk' },
    { name: 'Machine Learning', icon: <Brain size={20} />, path: '/ml' },
    { name: 'DEX Management', icon: <Repeat size={20} />, path: '/dex' },
    { name: 'Protocol Management', icon: <BookOpen size={20} />, path: '/protocols' },
    { name: 'Wallet', icon: <Wallet size={20} />, path: '/wallet' },
    { name: 'Smart Contracts', icon: <Code2 size={20} />, path: '/contracts' },
  ];

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-40 lg:hidden"
          onClick={onClose}
        ></div>
      )}

      <aside
        className={`fixed top-0 left-0 h-full w-64 bg-white dark:bg-dark-card border-r border-gray-200 dark:border-dark-border transition-transform duration-300 ease-in-out z-50 lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="p-4 border-b border-gray-200 dark:border-dark-border">
          <div className="flex items-center">
            <div className="w-8 h-8 rounded-md bg-primary-500 flex items-center justify-center">
              <Zap size={18} className="text-white" />
            </div>
            <div className="ml-3">
              <h1 className="text-lg font-bold text-gray-900 dark:text-white">TallyIO</h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">Autonomous Trading</p>
            </div>
          </div>
        </div>

        <nav className="p-4">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-4 uppercase tracking-wider">
            Main Navigation
          </p>

          <div className="space-y-1">
            {navigationItems.map((item) => (
              <NavLink
                key={item.name}
                to={item.path}
                className={({ isActive }) => `
                  flex items-center p-3 rounded-md cursor-pointer transition-colors
                  ${isActive
                    ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-background'
                  }
                `}
                onMouseEnter={() => preloadRoute(item.path)}
                onClick={() => {
                  // Predhodno naloži stran pred navigacijo
                  const handleClick = async () => {
                    await preloadRoute(item.path);
                  };
                  handleClick();
                }}
              >
                <span className="text-current">{item.icon}</span>
                <span className="ml-3 font-medium">{item.name}</span>
              </NavLink>
            ))}
          </div>
        </nav>
      </aside>
    </>
  );
};

export default Sidebar;