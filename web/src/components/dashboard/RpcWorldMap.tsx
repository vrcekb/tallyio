import React, { useRef, useMemo, memo } from 'react';
import { motion } from 'framer-motion';
import DottedMap from 'dotted-map';
import Card from '../ui/Card';
import { RpcLocation } from '../../types';

interface RpcWorldMapProps {
  data: RpcLocation[];
}

// Memoizirane pomožne funkcije
const projectPoint = (lat: number, lng: number) => {
  const x = (lng + 180) * (800 / 360);
  const y = (90 - lat) * (400 / 180);
  return { x, y };
};

const createCurvedPath = (
  start: { x: number; y: number },
  end: { x: number; y: number }
) => {
  const midX = (start.x + end.x) / 2;
  const midY = Math.min(start.y, end.y) - 50;
  return `M ${start.x} ${start.y} Q ${midX} ${midY} ${end.x} ${end.y}`;
};

// Memoizirana komponenta za točko na zemljevidu
const LocationPoint = memo(({ location, index }: { location: RpcLocation; index: number }) => {
  const point = projectPoint(location.coordinates[1], location.coordinates[0]);
  return (
    <g key={`point-group-${index}`}>
      <circle
        cx={point.x}
        cy={point.y}
        r="4"
        fill={
          location.status === 'online'
            ? '#22C55E'
            : location.status === 'degraded'
            ? '#F59E0B'
            : '#EF4444'
        }
        className="drop-shadow-md"
      />
      <circle
        cx={point.x}
        cy={point.y}
        r="4"
        fill={
          location.status === 'online'
            ? '#22C55E'
            : location.status === 'degraded'
            ? '#F59E0B'
            : '#EF4444'
        }
        opacity="0.5"
      >
        <animate
          attributeName="r"
          from="4"
          to="12"
          dur="1.5s"
          begin="0s"
          repeatCount="indefinite"
        />
        <animate
          attributeName="opacity"
          from="0.5"
          to="0"
          dur="1.5s"
          begin="0s"
          repeatCount="indefinite"
        />
      </circle>
      <title>{`${location.name}: ${location.latency}ms`}</title>
    </g>
  );
});

// Glavna komponenta
const RpcWorldMap: React.FC<RpcWorldMapProps> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  // Memoizacija zemljevida
  const svgMap = useMemo(() => {
    const map = new DottedMap({ height: 100, grid: 'diagonal' });
    return map.getSVG({
      radius: 0.22,
      color: '#64748B40',
      shape: 'circle',
      backgroundColor: 'transparent'
    });
  }, []);

  // Memoizacija povezav
  const connections = useMemo(() => {
    return data.reduce<Array<{ start: RpcLocation; end: RpcLocation }>>((acc, start, i) => {
      const nextNodes = data.slice(i + 1);
      const validConnections = nextNodes
        .filter(end => {
          const distance = Math.sqrt(
            Math.pow(end.coordinates[0] - start.coordinates[0], 2) +
            Math.pow(end.coordinates[1] - start.coordinates[1], 2)
          );
          return distance < 100; // Only connect relatively close nodes
        })
        .map(end => ({ start, end }));
      return [...acc, ...validConnections];
    }, []);
  }, [data]);

  // Memoizacija poti povezav
  const ConnectionPaths = memo(({ connections }: { connections: Array<{ start: RpcLocation; end: RpcLocation }> }) => {
    return (
      <>
        {connections.map((connection, i) => {
          const startPoint = projectPoint(connection.start.coordinates[1], connection.start.coordinates[0]);
          const endPoint = projectPoint(connection.end.coordinates[1], connection.end.coordinates[0]);
          return (
            <g key={`path-group-${i}`}>
              <motion.path
                d={createCurvedPath(startPoint, endPoint)}
                fill="none"
                stroke="url(#path-gradient)"
                strokeWidth="1"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{
                  duration: 1.5,
                  delay: 0.1 * Math.min(i, 5), // Omejimo zakasnitev na največ 5 povezav
                  ease: "easeOut"
                }}
              />
            </g>
          );
        })}
      </>
    );
  });

  // Memoizacija legende
  const Legend = memo(() => (
    <div className="absolute bottom-2 right-2 bg-white dark:bg-dark-card p-2 rounded-md shadow-md border border-gray-200 dark:border-dark-border">
      <div className="flex flex-col space-y-1">
        <div className="flex items-center text-xs">
          <div className="w-3 h-3 rounded-full bg-success-500 mr-1"></div>
          <span className="text-gray-700 dark:text-gray-300">Online</span>
        </div>
        <div className="flex items-center text-xs">
          <div className="w-3 h-3 rounded-full bg-warning-500 mr-1"></div>
          <span className="text-gray-700 dark:text-gray-300">Degraded</span>
        </div>
        <div className="flex items-center text-xs">
          <div className="w-3 h-3 rounded-full bg-error-500 mr-1"></div>
          <span className="text-gray-700 dark:text-gray-300">Offline</span>
        </div>
      </div>
    </div>
  ));

  return (
    <Card title="RPC Network">
      <div className="relative w-full h-[350px] bg-transparent">
        <img
          src={`data:image/svg+xml;utf8,${encodeURIComponent(svgMap)}`}
          className="h-full w-full [mask-image:linear-gradient(to_bottom,transparent,white_10%,white_90%,transparent)] pointer-events-none select-none"
          alt="world map"
          draggable={false}
          loading="lazy" // Dodamo lazy loading za sliko
        />
        <svg
          ref={svgRef}
          viewBox="0 0 800 400"
          className="w-full h-full absolute inset-0 pointer-events-none select-none"
        >
          <defs>
            <linearGradient id="path-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="white" stopOpacity="0" />
              <stop offset="5%" stopColor="#3575E3" stopOpacity="0.5" />
              <stop offset="95%" stopColor="#3575E3" stopOpacity="0.5" />
              <stop offset="100%" stopColor="white" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Uporabimo memorizirane komponente */}
          <ConnectionPaths connections={connections} />

          {data.map((location, i) => (
            <LocationPoint key={location.id || i} location={location} index={i} />
          ))}
        </svg>

        <div className="absolute top-2 left-2 flex flex-col space-y-1 bg-white dark:bg-dark-card p-2 rounded-md shadow-md border border-gray-200 dark:border-dark-border">
          <div className="flex items-center text-xs">
            <div className="w-3 h-3 rounded-full bg-success-500 mr-1"></div>
            <span className="text-gray-700 dark:text-gray-300">Online</span>
          </div>
          <div className="flex items-center text-xs">
            <div className="w-3 h-3 rounded-full bg-warning-500 mr-1"></div>
            <span className="text-gray-700 dark:text-gray-300">Degraded</span>
          </div>
          <div className="flex items-center text-xs">
            <div className="w-3 h-3 rounded-full bg-error-500 mr-1"></div>
            <span className="text-gray-700 dark:text-gray-300">Offline</span>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default RpcWorldMap;