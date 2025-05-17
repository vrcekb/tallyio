import React from 'react';
import { Moon, Sun, Monitor } from 'lucide-react';
import { useTheme } from '../../theme/ThemeContext';
import { ThemeMode } from '../../theme/types';

interface ThemeSelectorProps {
  className?: string;
}

/**
 * Komponenta za izbiro teme (svetla, temna, sistemska)
 */
const ThemeSelector: React.FC<ThemeSelectorProps> = ({ className = '' }) => {
  const { mode, setMode } = useTheme();

  // Funkcija za nastavitev načina teme
  const handleModeChange = (newMode: ThemeMode) => {
    setMode(newMode);
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <button
        onClick={() => handleModeChange('light')}
        className={`p-2 rounded-lg transition-colors ${
          mode === 'light'
            ? 'bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400'
            : 'hover:bg-gray-100 dark:hover:bg-dark-background text-gray-700 dark:text-gray-300'
        }`}
        title="Svetla tema"
        aria-label="Svetla tema"
      >
        <Sun size={18} />
      </button>

      <button
        onClick={() => handleModeChange('dark')}
        className={`p-2 rounded-lg transition-colors ${
          mode === 'dark'
            ? 'bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400'
            : 'hover:bg-gray-100 dark:hover:bg-dark-background text-gray-700 dark:text-gray-300'
        }`}
        title="Temna tema"
        aria-label="Temna tema"
      >
        <Moon size={18} />
      </button>

      <button
        onClick={() => handleModeChange('system')}
        className={`p-2 rounded-lg transition-colors ${
          mode === 'system'
            ? 'bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400'
            : 'hover:bg-gray-100 dark:hover:bg-dark-background text-gray-700 dark:text-gray-300'
        }`}
        title="Sistemska tema"
        aria-label="Sistemska tema"
      >
        <Monitor size={18} />
      </button>
    </div>
  );
};

export default ThemeSelector;
