import React from 'react';
import { useToast } from '../contexts/ToastContext';
import Button from '../components/ui/Button';

/**
 * Hook za demonstracijo uporabe obvestil (toast)
 */
const useToastDemo = () => {
  const { showToast } = useToast();

  const ToastDemo = () => {
    return (
      <div className="flex flex-wrap gap-2">
        <Button
          color="success"
          onClick={() => showToast('success', 'Uspeh!', 'Operacija je bila uspešno izvedena.')}
        >
          Uspeh
        </Button>
        <Button
          color="error"
          onClick={() => showToast('error', 'Napaka!', 'Prišlo je do napake pri izvajanju operacije.')}
        >
          Napaka
        </Button>
        <Button
          color="warning"
          onClick={() => showToast('warning', 'Opozorilo!', 'Bodite pozorni na to opozorilo.')}
        >
          Opozorilo
        </Button>
        <Button
          color="primary"
          onClick={() => showToast('info', 'Informacija', 'To je informativno sporočilo.')}
        >
          Informacija
        </Button>
      </div>
    );
  };

  return { ToastDemo };
};

export default useToastDemo;
