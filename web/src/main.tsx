import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import * as serviceWorkerRegistration from './serviceWorkerRegistration';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);

// Če želite, da vaša aplikacija deluje offline in se naloži hitreje, lahko spremenite
// unregister() v register() spodaj. Upoštevajte, da to prinaša nekaj pasti.
// Več o service workerjih: https://cra.link/PWA
serviceWorkerRegistration.register({
  onUpdate: (registration) => {
    // Prikaži obvestilo o posodobitvi
    serviceWorkerRegistration.showUpdateNotification(registration);
  },
  onSuccess: (registration) => {
    console.log('Service Worker uspešno registriran:', registration);
  },
});
