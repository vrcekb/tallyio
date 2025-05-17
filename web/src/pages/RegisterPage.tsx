import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { User, Mail, Lock, AlertCircle, Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

/**
 * Stran za registracijo uporabnika
 */
const RegisterPage: React.FC = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [agreeTerms, setAgreeTerms] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const { register, isAuthenticated, isLoading, error, clearError } = useAuth();
  const navigate = useNavigate();

  // Preusmeri na nadzorno ploščo, če je uporabnik že prijavljen
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, navigate]);

  // Funkcija za registracijo
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Preveri, ali se gesli ujemata
    if (password !== confirmPassword) {
      setValidationError('Gesli se ne ujemata');
      return;
    }
    
    // Preveri, ali je geslo dovolj močno
    if (password.length < 8) {
      setValidationError('Geslo mora vsebovati vsaj 8 znakov');
      return;
    }
    
    // Preveri, ali se uporabnik strinja s pogoji uporabe
    if (!agreeTerms) {
      setValidationError('Strinjati se morate s pogoji uporabe');
      return;
    }
    
    // Počisti validacijske napake
    setValidationError(null);
    
    // Registriraj uporabnika
    await register(name, email, password);
  };

  // Funkcija za preklop prikaza gesla
  const toggleShowPassword = () => {
    setShowPassword(!showPassword);
  };

  // Funkcija za preklop prikaza potrditvenega gesla
  const toggleShowConfirmPassword = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-dark-background py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
            Ustvarite nov račun
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
            Že imate račun?{' '}
            <Link
              to="/login"
              className="font-medium text-primary-600 hover:text-primary-500 dark:text-primary-400 dark:hover:text-primary-300"
            >
              Prijavite se
            </Link>
          </p>
        </div>
        
        {(error || validationError) && (
          <div className="bg-error-50 dark:bg-error-900/30 border border-error-200 dark:border-error-800 rounded-md p-4 flex items-start">
            <AlertCircle className="h-5 w-5 text-error-500 mr-3 mt-0.5" />
            <div className="text-sm text-error-700 dark:text-error-300">
              {validationError || error}
            </div>
            <button
              className="ml-auto text-error-500 hover:text-error-600 dark:text-error-400 dark:hover:text-error-300"
              onClick={() => {
                clearError();
                setValidationError(null);
              }}
              aria-label="Zapri opozorilo"
            >
              &times;
            </button>
          </div>
        )}
        
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div className="relative">
              <label htmlFor="name" className="sr-only">
                Ime in priimek
              </label>
              <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                id="name"
                name="name"
                type="text"
                autoComplete="name"
                required
                className="appearance-none rounded-t-md relative block w-full px-10 py-3 border border-gray-300 dark:border-gray-700 placeholder-gray-500 dark:placeholder-gray-400 text-gray-900 dark:text-white dark:bg-dark-card focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                placeholder="Ime in priimek"
                value={name}
                onChange={(e) => setName(e.target.value)}
                aria-required="true"
              />
            </div>
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
                className="appearance-none relative block w-full px-10 py-3 border border-gray-300 dark:border-gray-700 placeholder-gray-500 dark:placeholder-gray-400 text-gray-900 dark:text-white dark:bg-dark-card focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                placeholder="E-poštni naslov"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                aria-required="true"
              />
            </div>
            <div className="relative">
              <label htmlFor="password" className="sr-only">
                Geslo
              </label>
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                id="password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete="new-password"
                required
                className="appearance-none relative block w-full px-10 py-3 border border-gray-300 dark:border-gray-700 placeholder-gray-500 dark:placeholder-gray-400 text-gray-900 dark:text-white dark:bg-dark-card focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                placeholder="Geslo"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                aria-required="true"
              />
              <button
                type="button"
                className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
                onClick={toggleShowPassword}
                aria-label={showPassword ? 'Skrij geslo' : 'Prikaži geslo'}
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
            <div className="relative">
              <label htmlFor="confirm-password" className="sr-only">
                Potrdi geslo
              </label>
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                id="confirm-password"
                name="confirm-password"
                type={showConfirmPassword ? 'text' : 'password'}
                autoComplete="new-password"
                required
                className="appearance-none rounded-b-md relative block w-full px-10 py-3 border border-gray-300 dark:border-gray-700 placeholder-gray-500 dark:placeholder-gray-400 text-gray-900 dark:text-white dark:bg-dark-card focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                placeholder="Potrdi geslo"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                aria-required="true"
              />
              <button
                type="button"
                className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
                onClick={toggleShowConfirmPassword}
                aria-label={showConfirmPassword ? 'Skrij geslo' : 'Prikaži geslo'}
              >
                {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
          </div>

          <div className="flex items-center">
            <input
              id="agree-terms"
              name="agree-terms"
              type="checkbox"
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 dark:border-gray-700 rounded"
              checked={agreeTerms}
              onChange={(e) => setAgreeTerms(e.target.checked)}
            />
            <label
              htmlFor="agree-terms"
              className="ml-2 block text-sm text-gray-900 dark:text-gray-300"
            >
              Strinjam se s{' '}
              <a
                href="#"
                className="font-medium text-primary-600 hover:text-primary-500 dark:text-primary-400 dark:hover:text-primary-300"
              >
                pogoji uporabe
              </a>{' '}
              in{' '}
              <a
                href="#"
                className="font-medium text-primary-600 hover:text-primary-500 dark:text-primary-400 dark:hover:text-primary-300"
              >
                politiko zasebnosti
              </a>
            </label>
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
              {isLoading ? 'Registracija...' : 'Registracija'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RegisterPage;
