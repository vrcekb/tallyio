import React from 'react';

interface SkipToContentProps {
  contentId?: string;
  className?: string;
}

/**
 * Komponenta za preskok na vsebino
 * Omogoča uporabnikom, ki uporabljajo tipkovnico, da preskočijo navigacijo
 * in se takoj premaknejo na glavno vsebino
 */
const SkipToContent: React.FC<SkipToContentProps> = ({
  contentId = 'main-content',
  className = '',
}) => {
  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    
    const contentElement = document.getElementById(contentId);
    
    if (contentElement) {
      // Nastavi fokus na element
      contentElement.tabIndex = -1;
      contentElement.focus();
      
      // Premakni se na element
      contentElement.scrollIntoView();
    }
  };
  
  return (
    <a
      href={`#${contentId}`}
      className={`skip-to-content ${className}`}
      onClick={handleClick}
    >
      Skip to content
    </a>
  );
};

export default SkipToContent;
