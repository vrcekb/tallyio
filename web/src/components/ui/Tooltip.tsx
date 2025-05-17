import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type TooltipPosition = 'top' | 'right' | 'bottom' | 'left';

interface TooltipProps {
  children: React.ReactNode;
  content: React.ReactNode;
  position?: TooltipPosition;
  delay?: number;
  className?: string;
  contentClassName?: string;
  arrow?: boolean;
}

/**
 * Komponenta za prikaz tooltipa
 */
const Tooltip: React.FC<TooltipProps> = ({
  children,
  content,
  position = 'top',
  delay = 300,
  className = '',
  contentClassName = '',
  arrow = true
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (isVisible && triggerRef.current && tooltipRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      
      let x = 0;
      let y = 0;
      
      switch (position) {
        case 'top':
          x = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
          y = triggerRect.top - tooltipRect.height - 8;
          break;
        case 'right':
          x = triggerRect.right + 8;
          y = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
          break;
        case 'bottom':
          x = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
          y = triggerRect.bottom + 8;
          break;
        case 'left':
          x = triggerRect.left - tooltipRect.width - 8;
          y = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
          break;
      }
      
      // Ensure tooltip stays within viewport
      const padding = 10;
      x = Math.max(padding, Math.min(x, window.innerWidth - tooltipRect.width - padding));
      y = Math.max(padding, Math.min(y, window.innerHeight - tooltipRect.height - padding));
      
      setCoords({ x, y });
    }
  }, [isVisible, position]);

  const getInitialAnimation = () => {
    switch (position) {
      case 'top':
        return { opacity: 0, y: 10 };
      case 'right':
        return { opacity: 0, x: -10 };
      case 'bottom':
        return { opacity: 0, y: -10 };
      case 'left':
        return { opacity: 0, x: 10 };
    }
  };

  const getAnimateAnimation = () => {
    switch (position) {
      case 'top':
      case 'bottom':
        return { opacity: 1, y: 0 };
      case 'right':
      case 'left':
        return { opacity: 1, x: 0 };
    }
  };

  const getArrowPosition = () => {
    switch (position) {
      case 'top':
        return 'bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 rotate-45';
      case 'right':
        return 'left-0 top-1/2 transform -translate-x-1/2 -translate-y-1/2 rotate-45';
      case 'bottom':
        return 'top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 rotate-45';
      case 'left':
        return 'right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2 rotate-45';
    }
  };

  return (
    <div
      ref={triggerRef}
      className={`inline-block ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      
      <AnimatePresence>
        {isVisible && (
          <motion.div
            ref={tooltipRef}
            initial={getInitialAnimation()}
            animate={getAnimateAnimation()}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed z-50 pointer-events-none"
            style={{ left: coords.x, top: coords.y }}
          >
            <div className={`
              relative bg-gray-900 text-white text-sm rounded-md py-1.5 px-3 shadow-lg
              dark:bg-gray-800 max-w-xs ${contentClassName}
            `}>
              {arrow && (
                <div className={`
                  absolute w-2 h-2 bg-gray-900 dark:bg-gray-800
                  ${getArrowPosition()}
                `}></div>
              )}
              {content}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Tooltip;
