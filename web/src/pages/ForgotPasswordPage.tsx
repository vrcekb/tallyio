import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Mail, AlertCircle, CheckCircle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

/**
 * Stran za ponastavitev gesla
 */
const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const { resetPassword, isAuthenticated, isLoading, error, clearError } = useAuth();
  const navigate = useNavigate();

  // Preusmeri na nadzorno ploščo, če je uporabnik že prijavljen
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, navigate]);

  // Funkcija za ponastavitev gesla
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await resetPassword(email);
    setIsSubmitted(true);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-dark-background py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
            Ponastavitev gesla
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
            Vnesite svoj e-poštni naslov in poslali vam bomo navodila za ponastavitev gesla.
          </p>
        </div>
        
        {error && (
          <div className="bg-error-50 dark:bg-error-900/30 border border-error-200 dark:border-error-800 rounded-md p-4 flex items-start">
            <AlertCircle className="h-5 w-5 text-error-500 mr-3 mt-0.5" />
            <div className="text-sm text-error-700 dark:text-error-300">
              {error}
            </div>
            <button
              className="ml-auto text-error-500 hover:text-error-600 dark:text-error-400 dark:hover:text-error-300"
              onClick={clearError}
              aria-label="Zapri opozorilo"
            >
              &times;
            </button>
          </div>
        )}
        
        {isSubmitted && !error && (
          <div className="bg-success-50 dark:bg-success-900/30 border border-success-200 dark:border-success-800 rounded-md p-4 flex items-start">
            <CheckCircle className="h-5 w-5 text-success-500 mr-3 mt-0.5" />
            <div className="text-sm text-success-700 dark:text-success-300">
              Navodila za ponastavitev gesla smo poslali na vaš e-poštni naslov. Preverite svoj e-poštni predal.
            </div>
          </div>
        )}
        
        {!isSubmitted && (
          <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
            <div className="rounded-md shadow-sm">
              <div className="relative">
                <label htmlFor="email-address" className="sr-only">
                  E-poštni naslov
                </label>
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  id="email-address"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="appearance-none rounded-md relative block w-full px-10 py-3 border border-gray-300 dark:border-gray-700 placeholder-gray-500 dark:placeholder-gray-400 text-gray-900 dark:text-white dark:bg-dark-card focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                  placeholder="E-poštni naslov"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  aria-required="true"
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={isLoading}
                className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                ) : null}
                {isLoading ? 'Pošiljanje...' : 'Pošlji navodila'}
              </button>
            </div>
          </form>
        )}
        
        <div className="text-center mt-4">
          <Link
            to="/login"
            className="font-medium text-primary-600 hover:text-primary-500 dark:text-primary-400 dark:hover:text-primary-300"
          >
            Nazaj na prijavo
          </Link>
        </div>
        
        {isSubmitted && (
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Niste prejeli e-pošte?{' '}
              <button
                onClick={() => setIsSubmitted(false)}
                className="font-medium text-primary-600 hover:text-primary-500 dark:text-primary-400 dark:hover:text-primary-300"
              >
                Poskusite ponovno
              </button>
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ForgotPasswordPage;
