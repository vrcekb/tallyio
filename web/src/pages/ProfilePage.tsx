import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, Mail, AlertCircle, CheckCircle, Camera } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import Layout from '../components/layout/Layout';

/**
 * Stran za urejanje uporabniškega profila
 */
const ProfilePage: React.FC = () => {
  const { user, isAuthenticated, isLoading, error, updateProfile, clearError } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [avatar, setAvatar] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const navigate = useNavigate();

  // Preusmeri na prijavo, če uporabnik ni prijavljen
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [isLoading, isAuthenticated, navigate]);

  // Nastavi začetne vrednosti
  useEffect(() => {
    if (user) {
      setName(user.name);
      setEmail(user.email);
      setAvatar(user.avatar || '');
    }
  }, [user]);

  // Funkcija za posodobitev profila
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsUpdating(true);
    setSuccessMessage(null);

    try {
      await updateProfile({ name, email, avatar });
      setSuccessMessage('Profil je bil uspešno posodobljen');

      // Počisti sporočilo o uspehu po 5 sekundah
      setTimeout(() => {
        setSuccessMessage(null);
      }, 5000);
    } catch (_err) {
      // Napaka se bo prikazala preko AuthContext
    } finally {
      setIsUpdating(false);
    }
  };

  // Funkcija za nalaganje slike
  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setAvatar(event.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  if (isLoading || !user) {
    return (
      <Layout>
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-2xl font-bold mb-6">Uporabniški profil</h1>

        {error && (
          <div className="bg-error-50 dark:bg-error-900/30 border border-error-200 dark:border-error-800 rounded-md p-4 flex items-start mb-6">
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

        {successMessage && (
          <div className="bg-success-50 dark:bg-success-900/30 border border-success-200 dark:border-success-800 rounded-md p-4 flex items-start mb-6">
            <CheckCircle className="h-5 w-5 text-success-500 mr-3 mt-0.5" />
            <div className="text-sm text-success-700 dark:text-success-300">
              {successMessage}
            </div>
            <button
              className="ml-auto text-success-500 hover:text-success-600 dark:text-success-400 dark:hover:text-success-300"
              onClick={() => setSuccessMessage(null)}
              aria-label="Zapri opozorilo"
            >
              &times;
            </button>
          </div>
        )}

        <div className="bg-white dark:bg-dark-card shadow rounded-lg overflow-hidden">
          <div className="p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6 mb-8">
              <div className="relative">
                <div className="w-24 h-24 rounded-full overflow-hidden bg-gray-200 dark:bg-gray-700">
                  {avatar ? (
                    <img
                      src={avatar}
                      alt={name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      <User size={48} />
                    </div>
                  )}
                </div>
                <label
                  htmlFor="avatar-upload"
                  className="absolute bottom-0 right-0 bg-primary-500 text-white p-1.5 rounded-full cursor-pointer hover:bg-primary-600 transition-colors"
                >
                  <Camera size={16} />
                  <input
                    id="avatar-upload"
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleAvatarChange}
                  />
                </label>
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {user.name}
                </h2>
                <p className="text-gray-500 dark:text-gray-400">{user.email}</p>
                <div className="mt-2">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900/30 dark:text-primary-300">
                    {user.role === 'admin' ? 'Administrator' : 'Uporabnik'}
                  </span>
                </div>
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="space-y-6">
                <div>
                  <label
                    htmlFor="name"
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                  >
                    Ime in priimek
                  </label>
                  <div className="mt-1 relative rounded-md shadow-sm">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <User className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                      type="text"
                      name="name"
                      id="name"
                      className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm dark:bg-dark-card dark:text-white"
                      placeholder="Ime in priimek"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <div>
                  <label
                    htmlFor="email"
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                  >
                    E-poštni naslov
                  </label>
                  <div className="mt-1 relative rounded-md shadow-sm">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Mail className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                      type="email"
                      name="email"
                      id="email"
                      className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm dark:bg-dark-card dark:text-white"
                      placeholder="E-poštni naslov"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <div className="flex justify-end">
                  <button
                    type="submit"
                    disabled={isUpdating}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isUpdating ? (
                      <>
                        <svg
                          className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
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
                        Shranjevanje...
                      </>
                    ) : (
                      'Shrani spremembe'
                    )}
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default ProfilePage;
