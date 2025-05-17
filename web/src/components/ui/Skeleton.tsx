import React from 'react';

interface SkeletonProps {
  variant?: 'text' | 'circular' | 'rectangular' | 'card';
  width?: string | number;
  height?: string | number;
  className?: string;
  animation?: 'pulse' | 'wave' | 'none';
  repeat?: number;
  gap?: number;
}

/**
 * Komponenta za prikaz nalagalnika
 */
const Skeleton: React.FC<SkeletonProps> = ({
  variant = 'text',
  width,
  height,
  className = '',
  animation = 'pulse',
  repeat = 1,
  gap = 8
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'circular':
        return 'rounded-full';
      case 'rectangular':
        return 'rounded-md';
      case 'card':
        return 'rounded-lg';
      default:
        return 'rounded';
    }
  };

  const getAnimationClasses = () => {
    switch (animation) {
      case 'pulse':
        return 'animate-pulse';
      case 'wave':
        return 'skeleton-wave';
      default:
        return '';
    }
  };

  const getDefaultDimensions = () => {
    switch (variant) {
      case 'text':
        return { width: width || '100%', height: height || '1rem' };
      case 'circular':
        return { width: width || '2.5rem', height: height || '2.5rem' };
      case 'rectangular':
        return { width: width || '100%', height: height || '100px' };
      case 'card':
        return { width: width || '100%', height: height || '200px' };
      default:
        return { width, height };
    }
  };

  const dimensions = getDefaultDimensions();
  const variantClasses = getVariantClasses();
  const animationClasses = getAnimationClasses();

  const renderSkeleton = () => (
    <div
      className={`
        bg-gray-200 dark:bg-gray-700 ${variantClasses} ${animationClasses} ${className}
      `}
      style={{
        width: dimensions.width,
        height: dimensions.height
      }}
    />
  );

  if (repeat === 1) {
    return renderSkeleton();
  }

  return (
    <div className="flex flex-col" style={{ gap: `${gap}px` }}>
      {Array.from({ length: repeat }).map((_, index) => (
        <div key={index}>
          {renderSkeleton()}
        </div>
      ))}
    </div>
  );
};

export default Skeleton;
