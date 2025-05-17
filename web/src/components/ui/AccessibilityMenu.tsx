import React, { useState } from 'react';
import { Accessibility, X, Type, MousePointer2, Eye } from 'lucide-react';

interface AccessibilityMenuProps {
  className?: string;
}

/**
 * Komponenta za dostopnost
 * Omogoča prilagajanje uporabniškega vmesnika za boljšo dostopnost
 */
const AccessibilityMenu: React.FC<AccessibilityMenuProps> = ({ className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [fontSize, setFontSize] = useState(100);
  const [contrast, setContrast] = useState('normal');
  const [cursorSize, setCursorSize] = useState('normal');

  // Funkcija za povečanje velikosti pisave
  const increaseFontSize = () => {
    if (fontSize < 150) {
      const newSize = fontSize + 10;
      setFontSize(newSize);
      document.documentElement.style.fontSize = `${newSize}%`;
    }
  };

  // Funkcija za zmanjšanje velikosti pisave
  const decreaseFontSize = () => {
    if (fontSize > 80) {
      const newSize = fontSize - 10;
      setFontSize(newSize);
      document.documentElement.style.fontSize = `${newSize}%`;
    }
  };

  // Funkcija za ponastavitev velikosti pisave
  const resetFontSize = () => {
    setFontSize(100);
    document.documentElement.style.fontSize = '100%';
  };

  // Funkcija za nastavitev kontrasta
  const setHighContrast = () => {
    if (contrast === 'high') {
      setContrast('normal');
      document.documentElement.classList.remove('high-contrast');
    } else {
      setContrast('high');
      document.documentElement.classList.add('high-contrast');
    }
  };

  // Funkcija za nastavitev velikosti kazalca
  const toggleCursorSize = () => {
    if (cursorSize === 'large') {
      setCursorSize('normal');
      document.documentElement.classList.remove('large-cursor');
    } else {
      setCursorSize('large');
      document.documentElement.classList.add('large-cursor');
    }
  };

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-background"
        aria-label="Accessibility Menu"
        aria-expanded={isOpen}
      >
        <Accessibility size={20} className="text-gray-700 dark:text-gray-300" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-dark-card rounded-lg shadow-lg z-50 p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              Accessibility Options
            </h3>
            <button
              onClick={() => setIsOpen(false)}
              className="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-background"
              aria-label="Close Accessibility Menu"
            >
              <X size={16} className="text-gray-700 dark:text-gray-300" />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex items-center mb-2">
                <Type size={16} className="text-gray-700 dark:text-gray-300 mr-2" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Font Size: {fontSize}%
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={decreaseFontSize}
                  className="p-1 rounded-lg bg-gray-100 dark:bg-dark-background hover:bg-gray-200 dark:hover:bg-gray-700"
                  aria-label="Decrease Font Size"
                  disabled={fontSize <= 80}
                >
                  A-
                </button>
                <button
                  onClick={resetFontSize}
                  className="p-1 rounded-lg bg-gray-100 dark:bg-dark-background hover:bg-gray-200 dark:hover:bg-gray-700"
                  aria-label="Reset Font Size"
                >
                  Reset
                </button>
                <button
                  onClick={increaseFontSize}
                  className="p-1 rounded-lg bg-gray-100 dark:bg-dark-background hover:bg-gray-200 dark:hover:bg-gray-700"
                  aria-label="Increase Font Size"
                  disabled={fontSize >= 150}
                >
                  A+
                </button>
              </div>
            </div>

            <div>
              <button
                onClick={setHighContrast}
                className={`flex items-center w-full p-2 rounded-lg ${
                  contrast === 'high'
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : 'bg-gray-100 dark:bg-dark-background hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
                aria-pressed={contrast === 'high'}
              >
                <Eye size={16} className="mr-2" />
                <span className="text-sm font-medium">
                  {contrast === 'high' ? 'Disable' : 'Enable'} High Contrast
                </span>
              </button>
            </div>

            <div>
              <button
                onClick={toggleCursorSize}
                className={`flex items-center w-full p-2 rounded-lg ${
                  cursorSize === 'large'
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : 'bg-gray-100 dark:bg-dark-background hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
                aria-pressed={cursorSize === 'large'}
              >
                <MousePointer2 size={16} className="mr-2" />
                <span className="text-sm font-medium">
                  {cursorSize === 'large' ? 'Normal' : 'Large'} Cursor
                </span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AccessibilityMenu;
