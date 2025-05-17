/**
 * Funkcije za izboljšanje dostopnosti
 */

/**
 * Nastavi poslušalca za zaznavanje navigacije s tipkovnico
 * Ko uporabnik pritisne tipko Tab, doda razred 'user-is-tabbing' na body
 * Ko uporabnik klikne z miško, odstrani razred 'user-is-tabbing' z body
 */
export function setupKeyboardNavigation(): void {
  function handleFirstTab(e: KeyboardEvent): void {
    if (e.key === 'Tab') {
      document.body.classList.add('user-is-tabbing');
      
      window.removeEventListener('keydown', handleFirstTab);
      window.addEventListener('mousedown', handleMouseDownOnce);
    }
  }
  
  function handleMouseDownOnce(): void {
    document.body.classList.remove('user-is-tabbing');
    
    window.removeEventListener('mousedown', handleMouseDownOnce);
    window.addEventListener('keydown', handleFirstTab);
  }
  
  window.addEventListener('keydown', handleFirstTab);
}

/**
 * Nastavi poslušalca za zaznavanje preferenc za zmanjšano gibanje
 * Ko uporabnik nastavi preferenco za zmanjšano gibanje, doda razred 'reduced-motion' na body
 */
export function setupReducedMotionListener(): void {
  const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  
  function handleReducedMotionChange(e: MediaQueryListEvent): void {
    if (e.matches) {
      document.body.classList.add('reduced-motion');
    } else {
      document.body.classList.remove('reduced-motion');
    }
  }
  
  // Nastavi začetno stanje
  if (mediaQuery.matches) {
    document.body.classList.add('reduced-motion');
  }
  
  // Poslušaj za spremembe
  mediaQuery.addEventListener('change', handleReducedMotionChange);
}

/**
 * Nastavi poslušalca za zaznavanje preferenc za visok kontrast
 * Ko uporabnik nastavi preferenco za visok kontrast, doda razred 'high-contrast' na body
 */
export function setupHighContrastListener(): void {
  const mediaQuery = window.matchMedia('(prefers-contrast: more)');
  
  function handleHighContrastChange(e: MediaQueryListEvent): void {
    if (e.matches) {
      document.documentElement.classList.add('high-contrast');
    } else {
      document.documentElement.classList.remove('high-contrast');
    }
  }
  
  // Nastavi začetno stanje
  if (mediaQuery.matches) {
    document.documentElement.classList.add('high-contrast');
  }
  
  // Poslušaj za spremembe
  mediaQuery.addEventListener('change', handleHighContrastChange);
}

/**
 * Inicializira vse funkcije za dostopnost
 */
export function initAccessibility(): void {
  setupKeyboardNavigation();
  setupReducedMotionListener();
  setupHighContrastListener();
}
