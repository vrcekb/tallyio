import React from 'react';
import { motion } from 'framer-motion';

interface LayoutProps {
  children: React.ReactNode;
  animate?: boolean;
}

const Layout: React.FC<LayoutProps> = ({ children, animate = true }) => {
  if (!animate) {
    return <div className="space-y-6">{children}</div>;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 10 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
      className="space-y-6"
    >
      {children}
    </motion.div>
  );
};

export default Layout;