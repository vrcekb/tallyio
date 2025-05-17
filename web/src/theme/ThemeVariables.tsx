import React, { useEffect } from 'react';
import { useTheme } from './ThemeContext';

/**
 * Komponenta za nastavitev CSS spremenljivk za temo
 * Te spremenljivke so nato na voljo v celotni aplikaciji
 */
const ThemeVariables: React.FC = () => {
  const { theme, isDark } = useTheme();

  // Učinek za nastavitev CSS spremenljivk
  useEffect(() => {
    const root = document.documentElement;

    // Nastavi barve
    // Primarna barva
    root.style.setProperty('--color-primary-50', theme.colors.primary[50]);
    root.style.setProperty('--color-primary-100', theme.colors.primary[100]);
    root.style.setProperty('--color-primary-200', theme.colors.primary[200]);
    root.style.setProperty('--color-primary-300', theme.colors.primary[300]);
    root.style.setProperty('--color-primary-400', theme.colors.primary[400]);
    root.style.setProperty('--color-primary-500', theme.colors.primary[500]);
    root.style.setProperty('--color-primary-600', theme.colors.primary[600]);
    root.style.setProperty('--color-primary-700', theme.colors.primary[700]);
    root.style.setProperty('--color-primary-800', theme.colors.primary[800]);
    root.style.setProperty('--color-primary-900', theme.colors.primary[900]);

    // Uspeh barva
    root.style.setProperty('--color-success-50', theme.colors.success[50]);
    root.style.setProperty('--color-success-100', theme.colors.success[100]);
    root.style.setProperty('--color-success-200', theme.colors.success[200]);
    root.style.setProperty('--color-success-300', theme.colors.success[300]);
    root.style.setProperty('--color-success-400', theme.colors.success[400]);
    root.style.setProperty('--color-success-500', theme.colors.success[500]);
    root.style.setProperty('--color-success-600', theme.colors.success[600]);
    root.style.setProperty('--color-success-700', theme.colors.success[700]);
    root.style.setProperty('--color-success-800', theme.colors.success[800]);
    root.style.setProperty('--color-success-900', theme.colors.success[900]);

    // Opozorilo barva
    root.style.setProperty('--color-warning-50', theme.colors.warning[50]);
    root.style.setProperty('--color-warning-100', theme.colors.warning[100]);
    root.style.setProperty('--color-warning-200', theme.colors.warning[200]);
    root.style.setProperty('--color-warning-300', theme.colors.warning[300]);
    root.style.setProperty('--color-warning-400', theme.colors.warning[400]);
    root.style.setProperty('--color-warning-500', theme.colors.warning[500]);
    root.style.setProperty('--color-warning-600', theme.colors.warning[600]);
    root.style.setProperty('--color-warning-700', theme.colors.warning[700]);
    root.style.setProperty('--color-warning-800', theme.colors.warning[800]);
    root.style.setProperty('--color-warning-900', theme.colors.warning[900]);

    // Napaka barva
    root.style.setProperty('--color-error-50', theme.colors.error[50]);
    root.style.setProperty('--color-error-100', theme.colors.error[100]);
    root.style.setProperty('--color-error-200', theme.colors.error[200]);
    root.style.setProperty('--color-error-300', theme.colors.error[300]);
    root.style.setProperty('--color-error-400', theme.colors.error[400]);
    root.style.setProperty('--color-error-500', theme.colors.error[500]);
    root.style.setProperty('--color-error-600', theme.colors.error[600]);
    root.style.setProperty('--color-error-700', theme.colors.error[700]);
    root.style.setProperty('--color-error-800', theme.colors.error[800]);
    root.style.setProperty('--color-error-900', theme.colors.error[900]);

    // Siva barva
    root.style.setProperty('--color-gray-50', theme.colors.gray[50]);
    root.style.setProperty('--color-gray-100', theme.colors.gray[100]);
    root.style.setProperty('--color-gray-200', theme.colors.gray[200]);
    root.style.setProperty('--color-gray-300', theme.colors.gray[300]);
    root.style.setProperty('--color-gray-400', theme.colors.gray[400]);
    root.style.setProperty('--color-gray-500', theme.colors.gray[500]);
    root.style.setProperty('--color-gray-600', theme.colors.gray[600]);
    root.style.setProperty('--color-gray-700', theme.colors.gray[700]);
    root.style.setProperty('--color-gray-800', theme.colors.gray[800]);
    root.style.setProperty('--color-gray-900', theme.colors.gray[900]);

    // Ozadje
    root.style.setProperty('--color-background', isDark ? theme.colors.background.dark : theme.colors.background.light);
    
    // Kartica
    root.style.setProperty('--color-card', isDark ? theme.colors.card.dark : theme.colors.card.light);
    
    // Besedilo
    root.style.setProperty('--color-text-primary', isDark ? theme.colors.text.primary.dark : theme.colors.text.primary.light);
    root.style.setProperty('--color-text-secondary', isDark ? theme.colors.text.secondary.dark : theme.colors.text.secondary.light);
    root.style.setProperty('--color-text-muted', isDark ? theme.colors.text.muted.dark : theme.colors.text.muted.light);
    
    // Rob
    root.style.setProperty('--color-border', isDark ? theme.colors.border.dark : theme.colors.border.light);

    // Zaobljenost robov
    root.style.setProperty('--border-radius-none', theme.borderRadius.none);
    root.style.setProperty('--border-radius-sm', theme.borderRadius.sm);
    root.style.setProperty('--border-radius-md', theme.borderRadius.md);
    root.style.setProperty('--border-radius-lg', theme.borderRadius.lg);
    root.style.setProperty('--border-radius-xl', theme.borderRadius.xl);
    root.style.setProperty('--border-radius-full', theme.borderRadius.full);

    // Senčenje
    root.style.setProperty('--shadow-none', theme.shadows.none);
    root.style.setProperty('--shadow-sm', theme.shadows.sm);
    root.style.setProperty('--shadow-md', theme.shadows.md);
    root.style.setProperty('--shadow-lg', theme.shadows.lg);
    root.style.setProperty('--shadow-xl', theme.shadows.xl);

    // Tipografija
    root.style.setProperty('--font-family-sans', theme.typography.fontFamily.sans);
    root.style.setProperty('--font-family-mono', theme.typography.fontFamily.mono);

    // Velikost pisave
    root.style.setProperty('--font-size-xs', theme.typography.fontSize.xs);
    root.style.setProperty('--font-size-sm', theme.typography.fontSize.sm);
    root.style.setProperty('--font-size-base', theme.typography.fontSize.base);
    root.style.setProperty('--font-size-lg', theme.typography.fontSize.lg);
    root.style.setProperty('--font-size-xl', theme.typography.fontSize.xl);
    root.style.setProperty('--font-size-2xl', theme.typography.fontSize['2xl']);
    root.style.setProperty('--font-size-3xl', theme.typography.fontSize['3xl']);
    root.style.setProperty('--font-size-4xl', theme.typography.fontSize['4xl']);
    root.style.setProperty('--font-size-5xl', theme.typography.fontSize['5xl']);

    // Debelina pisave
    root.style.setProperty('--font-weight-light', theme.typography.fontWeight.light);
    root.style.setProperty('--font-weight-normal', theme.typography.fontWeight.normal);
    root.style.setProperty('--font-weight-medium', theme.typography.fontWeight.medium);
    root.style.setProperty('--font-weight-semibold', theme.typography.fontWeight.semibold);
    root.style.setProperty('--font-weight-bold', theme.typography.fontWeight.bold);

    // Višina vrstice
    root.style.setProperty('--line-height-none', theme.typography.lineHeight.none);
    root.style.setProperty('--line-height-tight', theme.typography.lineHeight.tight);
    root.style.setProperty('--line-height-normal', theme.typography.lineHeight.normal);
    root.style.setProperty('--line-height-relaxed', theme.typography.lineHeight.relaxed);
    root.style.setProperty('--line-height-loose', theme.typography.lineHeight.loose);

    // Animacije
    root.style.setProperty('--transition-fast', theme.animations.transition.fast);
    root.style.setProperty('--transition-normal', theme.animations.transition.normal);
    root.style.setProperty('--transition-slow', theme.animations.transition.slow);

    // Easing
    root.style.setProperty('--easing-ease-in-out', theme.animations.easing.easeInOut);
    root.style.setProperty('--easing-ease-out', theme.animations.easing.easeOut);
    root.style.setProperty('--easing-ease-in', theme.animations.easing.easeIn);

    // Razmiki
    root.style.setProperty('--spacing-0', theme.spacing[0]);
    root.style.setProperty('--spacing-1', theme.spacing[1]);
    root.style.setProperty('--spacing-2', theme.spacing[2]);
    root.style.setProperty('--spacing-3', theme.spacing[3]);
    root.style.setProperty('--spacing-4', theme.spacing[4]);
    root.style.setProperty('--spacing-5', theme.spacing[5]);
    root.style.setProperty('--spacing-6', theme.spacing[6]);
    root.style.setProperty('--spacing-8', theme.spacing[8]);
    root.style.setProperty('--spacing-10', theme.spacing[10]);
    root.style.setProperty('--spacing-12', theme.spacing[12]);
    root.style.setProperty('--spacing-16', theme.spacing[16]);
    root.style.setProperty('--spacing-20', theme.spacing[20]);
    root.style.setProperty('--spacing-24', theme.spacing[24]);
    root.style.setProperty('--spacing-32', theme.spacing[32]);
    root.style.setProperty('--spacing-40', theme.spacing[40]);
    root.style.setProperty('--spacing-48', theme.spacing[48]);
    root.style.setProperty('--spacing-64', theme.spacing[64]);
  }, [theme, isDark]);

  return null;
};

export default ThemeVariables;
