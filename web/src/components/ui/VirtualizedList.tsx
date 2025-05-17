import React, { useRef, useState, useEffect, useCallback, memo } from 'react';
import { FixedSizeList as List, ListChildComponentProps } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';

interface VirtualizedListProps<T> {
  data: T[];
  height?: number | string;
  itemHeight?: number;
  renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode;
  className?: string;
  itemClassName?: string;
  overscanCount?: number;
  onScroll?: (scrollInfo: { scrollOffset: number; scrollDirection: 'forward' | 'backward' }) => void;
  scrollToIndex?: number;
}

/**
 * Komponenta za virtualiziran seznam
 * Uporablja react-window za učinkovit prikaz velikih seznamov
 */
function VirtualizedListComponent<T>({
  data,
  height = 400,
  itemHeight = 50,
  renderItem,
  className = '',
  itemClassName = '',
  overscanCount = 5,
  onScroll,
  scrollToIndex
}: VirtualizedListProps<T>) {
  const listRef = useRef<List>(null);
  // Spremenljivki sta uporabljeni v handleScroll funkciji
  const [_scrollDirection, setScrollDirection] = useState<'forward' | 'backward'>('forward');
  const [_lastScrollOffset, setLastScrollOffset] = useState(0);

  // Učinek za pomik na določen indeks
  useEffect(() => {
    if (scrollToIndex !== undefined && listRef.current) {
      listRef.current.scrollToItem(scrollToIndex, 'start');
    }
  }, [scrollToIndex]);

  // Funkcija za obravnavo dogodka pomikanja
  const handleScroll = useCallback(({ scrollOffset, scrollDirection }: { scrollOffset: number; scrollDirection: 'forward' | 'backward' }) => {
    setScrollDirection(scrollDirection);
    setLastScrollOffset(scrollOffset);

    if (onScroll) {
      onScroll({ scrollOffset, scrollDirection });
    }
  }, [onScroll]);

  // Funkcija za upodabljanje elementa
  const Row = memo(({ index, style }: ListChildComponentProps) => {
    const item = data[index];
    return (
      <div
        style={style}
        className={`${itemClassName} ${index % 2 === 0 ? 'bg-white dark:bg-dark-card' : 'bg-gray-50/50 dark:bg-dark-background/50'} hover:bg-primary-50/30 dark:hover:bg-primary-900/10 transition-all duration-300`}
      >
        {renderItem(item, index, style)}
      </div>
    );
  });

  return (
    <div className={`overflow-hidden rounded-b-lg ${className}`} style={{ height }}>
      <AutoSizer>
        {({ width, height }) => (
          <List
            ref={listRef}
            width={width}
            height={height}
            itemCount={data.length}
            itemSize={itemHeight}
            overscanCount={overscanCount}
            onScroll={handleScroll}
          >
            {Row}
          </List>
        )}
      </AutoSizer>
    </div>
  );
}

// Memorizirana verzija komponente za boljšo učinkovitost
const VirtualizedList = memo(VirtualizedListComponent) as typeof VirtualizedListComponent;

export default VirtualizedList;
