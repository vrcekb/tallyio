import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: 'admin' | 'user';
}

/**
 * Komponenta za zaščitene poti
 * Preusmeri na prijavo, če uporabnik ni prijavljen
 * Preusmeri na nadzorno ploščo, če uporabnik nima ustrezne vloge
 */
const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRole,
}) => {
  const { isAuthenticated, isLoading, user } = useAuth();
  const location = useLocation();

  // Če se še nalagajo podatki, prikaži nalaganje
  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  // Če uporabnik ni prijavljen, preusmeri na prijavo
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Če je zahtevana vloga in uporabnik nima ustrezne vloge, preusmeri na nadzorno ploščo
  if (requiredRole && user?.role !== requiredRole) {
    return <Navigate to="/" replace />;
  }

  // Če je uporabnik prijavljen in ima ustrezno vlogo, prikaži vsebino
  return <>{children}</>;
};

export default ProtectedRoute;
