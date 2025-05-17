import React from 'react';
import { Palette } from 'lucide-react';
import { useTheme } from '../../theme/ThemeContext';
import { ThemeName } from '../../theme/types';

interface ThemeNameSelectorProps {
  className?: string;
}

/**
 * Komponenta za izbiro teme (default, dark, nordic)
 */
const ThemeNameSelector: React.FC<ThemeNameSelectorProps> = ({ className = '' }) => {
  const { themeName, setThemeName } = useTheme();

  // Funkcija za nastavitev imena teme
  const handleThemeChange = (newTheme: ThemeName) => {
    setThemeName(newTheme);
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className="relative group">
        <button
          className="p-2 rounded-lg transition-colors hover:bg-gray-100 dark:hover:bg-dark-background text-gray-700 dark:text-gray-300"
          title="Izberi temo"
          aria-label="Izberi temo"
        >
          <Palette size={18} />
        </button>
        <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-dark-card rounded-lg shadow-lg border border-gray-200 dark:border-dark-border opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 z-50">
          <div className="p-2">
            <button
              onClick={() => handleThemeChange('default')}
              className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                themeName === 'default'
                  ? 'bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400'
                  : 'hover:bg-gray-100 dark:hover:bg-dark-background text-gray-700 dark:text-gray-300'
              }`}
            >
              <div className="flex items-center">
                <div className="w-4 h-4 rounded-full bg-blue-500 mr-2"></div>
                <span>Privzeta tema</span>
              </div>
            </button>
            <button
              onClick={() => handleThemeChange('dark')}
              className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                themeName === 'dark'
                  ? 'bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400'
                  : 'hover:bg-gray-100 dark:hover:bg-dark-background text-gray-700 dark:text-gray-300'
              }`}
            >
              <div className="flex items-center">
                <div className="w-4 h-4 rounded-full bg-gray-800 mr-2"></div>
                <span>Temna tema</span>
              </div>
            </button>
            <button
              onClick={() => handleThemeChange('nordic')}
              className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                themeName === 'nordic'
                  ? 'bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400'
                  : 'hover:bg-gray-100 dark:hover:bg-dark-background text-gray-700 dark:text-gray-300'
              }`}
            >
              <div className="flex items-center">
                <div className="w-4 h-4 rounded-full bg-[#526D82] mr-2"></div>
                <span>Nordijska tema</span>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ThemeNameSelector;
