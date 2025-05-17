import React from 'react';

interface AvatarProps {
  src?: string;
  alt?: string;
  name?: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  shape?: 'circle' | 'square';
  status?: 'online' | 'offline' | 'away' | 'busy';
  className?: string;
  imageClassName?: string;
  fallbackClassName?: string;
  statusClassName?: string;
}

/**
 * Komponenta za prikaz avatarja
 */
const Avatar: React.FC<AvatarProps> = ({
  src,
  alt = '',
  name,
  size = 'md',
  shape = 'circle',
  status,
  className = '',
  imageClassName = '',
  fallbackClassName = '',
  statusClassName = ''
}) => {
  const [imageError, setImageError] = React.useState(false);

  const handleImageError = () => {
    setImageError(true);
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'xs':
        return {
          container: 'w-6 h-6',
          text: 'text-xs',
          status: 'w-1.5 h-1.5'
        };
      case 'sm':
        return {
          container: 'w-8 h-8',
          text: 'text-sm',
          status: 'w-2 h-2'
        };
      case 'lg':
        return {
          container: 'w-12 h-12',
          text: 'text-lg',
          status: 'w-3 h-3'
        };
      case 'xl':
        return {
          container: 'w-16 h-16',
          text: 'text-xl',
          status: 'w-4 h-4'
        };
      default:
        return {
          container: 'w-10 h-10',
          text: 'text-base',
          status: 'w-2.5 h-2.5'
        };
    }
  };

  const getShapeClasses = () => {
    switch (shape) {
      case 'square':
        return 'rounded-md';
      default:
        return 'rounded-full';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'online':
        return 'bg-success-500';
      case 'offline':
        return 'bg-gray-400';
      case 'away':
        return 'bg-warning-500';
      case 'busy':
        return 'bg-error-500';
      default:
        return 'bg-success-500';
    }
  };

  const getInitials = () => {
    if (!name) return '';
    
    const nameParts = name.split(' ');
    if (nameParts.length === 1) {
      return nameParts[0].charAt(0).toUpperCase();
    }
    
    return (
      nameParts[0].charAt(0).toUpperCase() +
      nameParts[nameParts.length - 1].charAt(0).toUpperCase()
    );
  };

  const getRandomColor = () => {
    if (!name) return 'bg-primary-500';
    
    const colors = [
      'bg-primary-500',
      'bg-success-500',
      'bg-warning-500',
      'bg-error-500',
      'bg-purple-500',
      'bg-pink-500',
      'bg-indigo-500',
      'bg-blue-500',
      'bg-teal-500'
    ];
    
    // Use a hash function to get a consistent color for the same name
    const hash = name.split('').reduce((acc, char) => {
      return acc + char.charCodeAt(0);
    }, 0);
    
    return colors[hash % colors.length];
  };

  const sizeClasses = getSizeClasses();
  const shapeClasses = getShapeClasses();
  const statusColor = getStatusColor();
  const initials = getInitials();
  const bgColor = getRandomColor();

  return (
    <div className={`relative inline-flex ${className}`}>
      <div className={`${sizeClasses.container} ${shapeClasses} overflow-hidden flex-shrink-0`}>
        {src && !imageError ? (
          <img
            src={src}
            alt={alt}
            className={`w-full h-full object-cover ${imageClassName}`}
            onError={handleImageError}
          />
        ) : (
          <div className={`
            w-full h-full flex items-center justify-center text-white
            ${bgColor} ${sizeClasses.text} ${fallbackClassName}
          `}>
            {initials}
          </div>
        )}
      </div>
      
      {status && (
        <span className={`
          absolute bottom-0 right-0 block ${sizeClasses.status} ${shapeClasses}
          ${statusColor} ring-2 ring-white dark:ring-dark-card ${statusClassName}
        `} />
      )}
    </div>
  );
};

export default Avatar;
