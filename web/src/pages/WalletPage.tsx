import React, { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Wallet,
  Plus,
  Send,
  Download,
  Shield,
  AlertTriangle,
  ExternalLink,
  Copy,
  Check,
  Trash2,
  X,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Layout from '../components/layout/Layout';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import Tooltip from '../components/ui/Tooltip';
import Modal from '../components/ui/Modal';
import Input from '../components/ui/Input';
import Switch from '../components/ui/Switch';
import Alert from '../components/ui/Alert';
import VirtualizedTransactionList from '../components/wallet/VirtualizedTransactionList';
import useWorker, { MessageType } from '../hooks/useWorker';
import { useToast } from '../contexts/ToastContext';

interface WalletTransaction {
  id: string;
  type: 'send' | 'receive';
  amount: string;
  token: string;
  address: string;
  timestamp: string;
  status: 'completed' | 'pending' | 'failed';
}

interface ConnectedWallet {
  id: string;
  name: string;
  address: string;
  balance: string;
  tokens: Array<{
    symbol: string;
    balance: string;
    value: number;
  }>;
}

const mockTransactions: WalletTransaction[] = [
  {
    id: '1',
    type: 'send',
    amount: '0.5',
    token: 'ETH',
    address: '0x1234...5678',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    status: 'completed'
  },
  {
    id: '2',
    type: 'receive',
    amount: '1000',
    token: 'USDC',
    address: '0x8765...4321',
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    status: 'completed'
  }
];

const mockWallets: ConnectedWallet[] = [
  {
    id: '1',
    name: 'Main Wallet',
    address: '0x1234567890abcdef1234567890abcdef12345678',
    balance: '12.45',
    tokens: [
      { symbol: 'ETH', balance: '12.45', value: 24567.89 },
      { symbol: 'USDC', balance: '5000.00', value: 5000.00 },
      { symbol: 'WBTC', balance: '0.5', value: 15678.45 }
    ]
  }
];

const CreateWalletModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onCreateWallet: (name: string) => void;
}> = ({ isOpen, onClose, onCreateWallet }) => {
  const [walletName, setWalletName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!walletName.trim()) return;

    setIsCreating(true);
    try {
      // Simulate wallet creation
      await new Promise(resolve => setTimeout(resolve, 1500));
      onCreateWallet(walletName);
      onClose();
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Create New Wallet"
      size="md"
    >
      <form onSubmit={handleSubmit}>
        <Alert
          status="info"
          variant="subtle"
          className="mb-4"
        >
          Create a new wallet to manage your assets securely.
        </Alert>

        <div className="mb-6">
          <Input
            label="Wallet Name"
            value={walletName}
            onChange={(e) => setWalletName(e.target.value)}
            placeholder="Enter wallet name"
            fullWidth
            disabled={isCreating}
          />
        </div>

        <div className="flex justify-end space-x-3">
          <Button
            variant="ghost"
            onClick={onClose}
            disabled={isCreating}
          >
            Cancel
          </Button>
          <Button
            type="submit"
            color="primary"
            disabled={!walletName.trim() || isCreating}
            loading={isCreating}
            loadingText="Creating..."
          >
            Create Wallet
          </Button>
        </div>
      </form>
    </Modal>
  );
};

const WalletPage: React.FC = () => {
  const [selectedWallet, setSelectedWallet] = useState(mockWallets[0]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [drainProtection, setDrainProtection] = useState(true);
  const [copiedAddress, setCopiedAddress] = useState(false);
  const [wallets, setWallets] = useState(mockWallets);
  const { showToast } = useToast();

  const copyAddress = (address: string) => {
    navigator.clipboard.writeText(address);
    setCopiedAddress(true);
    setTimeout(() => setCopiedAddress(false), 2000);
    showToast('success', 'Address copied', 'Wallet address has been copied to clipboard');
  };

  const formatAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const handleCreateWallet = (name: string) => {
    const newWallet: ConnectedWallet = {
      id: `wallet-${Date.now()}`,
      name,
      address: `0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
      balance: '0.00',
      tokens: [
        { symbol: 'ETH', balance: '0.00', value: 0 }
      ]
    };

    setWallets(prev => [...prev, newWallet]);
    setSelectedWallet(newWallet);
    showToast('success', 'Wallet created', `${name} wallet has been successfully created`);
  };

  const handleDeleteWallet = (id: string, name: string) => {
    showToast('warning', 'Deleting wallet', `Are you sure you want to delete ${name}?`);
    // Simulirajmo zamik
    setTimeout(() => {
      setWallets(prev => prev.filter(wallet => wallet.id !== id));
      if (selectedWallet.id === id && wallets.length > 1) {
        setSelectedWallet(wallets.find(w => w.id !== id) || wallets[0]);
      }
      showToast('success', 'Wallet deleted', `${name} has been successfully deleted`);
    }, 1500);
  };

  const handleSend = () => {
    showToast('info', 'Send transaction', 'Preparing to send assets');
  };

  const handleReceive = () => {
    showToast('info', 'Receive assets', 'Preparing to receive assets');
  };

  // Uporaba Web Workerja za obdelavo transakcij
  const {
    loading: processingTransactions,
    data: processedTransactions,
    execute: processTransactions
  } = useWorker<{ transactions: WalletTransaction[], stats: any }, WalletTransaction[]>(
    new URL('../workers/dataProcessor.worker.ts', import.meta.url).href,
    MessageType.PROCESS_TRANSACTIONS
  );

  // Memoizacija seznama transakcij za izboljšanje zmogljivosti
  const memoizedTransactions = useMemo(() =>
    processedTransactions?.transactions || mockTransactions,
    [processedTransactions]
  );

  // Obdelava transakcij ob nalaganju komponente
  useEffect(() => {
    processTransactions(mockTransactions)
      .then(result => {
        console.log('Transactions processed:', result);
      })
      .catch(error => {
        console.error('Error processing transactions:', error);
      });
  }, [processTransactions]);

  // Funkcija za prikaz podrobnosti transakcije
  const handleViewTransactionDetails = useCallback((txId: string) => {
    const tx = memoizedTransactions.find(t => t.id === txId);
    if (tx) {
      showToast('info', 'Transaction details', `Viewing details for transaction ${txId}`);
    }
  }, [memoizedTransactions, showToast]);

  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          <Wallet className="w-8 h-8 text-primary-600 dark:text-primary-400 mr-3" />
          <h1 className="text-2xl font-bold text-primary-900 dark:text-primary-100">Wallet Management</h1>
        </div>
        <Button
          color="primary"
          leftIcon={<Plus size={20} />}
          onClick={() => setShowCreateModal(true)}
        >
          Create Wallet
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main Wallet Content */}
        <div className="lg:col-span-8">
          <Card>
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-lg font-medium text-primary-900 dark:text-primary-100">{selectedWallet.name}</h2>
                <div className="flex items-center mt-1">
                  <p className="text-sm text-primary-600 dark:text-primary-400">{formatAddress(selectedWallet.address)}</p>
                  <Tooltip content="Copy address">
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => copyAddress(selectedWallet.address)}
                    >
                      {copiedAddress ? (
                        <Check size={16} className="text-success-500" />
                      ) : (
                        <Copy size={16} className="text-primary-400" />
                      )}
                    </Button>
                  </Tooltip>
                  <Tooltip content="View on Etherscan">
                    <a
                      href={`https://etherscan.io/address/${selectedWallet.address}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-1 hover:bg-primary-50 dark:hover:bg-dark-background rounded-md"
                    >
                      <ExternalLink size={16} className="text-primary-500" />
                    </a>
                  </Tooltip>
                </div>
              </div>
              <div className="flex space-x-2">
                <Button
                  color="primary"
                  leftIcon={<Send size={18} />}
                  onClick={handleSend}
                >
                  Send
                </Button>
                <Button
                  variant="outline"
                  color="primary"
                  leftIcon={<Download size={18} />}
                  onClick={handleReceive}
                >
                  Receive
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {selectedWallet.tokens.map((token) => (
                <div
                  key={token.symbol}
                  className="p-4 bg-primary-50 dark:bg-dark-background rounded-lg transition-all duration-300 hover:shadow-md hover:translate-y-[-2px]"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-primary-600 dark:text-primary-400">{token.symbol}</span>
                    <Badge variant="subtle" color="primary" size="sm">
                      ${token.value.toLocaleString()}
                    </Badge>
                  </div>
                  <p className="text-lg font-medium text-primary-900 dark:text-primary-100">{token.balance}</p>
                </div>
              ))}
            </div>

            <div className="border-t border-primary-100 dark:border-dark-border pt-6">
              <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100 mb-4">Recent Transactions</h3>
              {processingTransactions ? (
                <div className="flex justify-center items-center h-40">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-500"></div>
                  <span className="ml-2 text-primary-500">Processing transactions...</span>
                </div>
              ) : (
                <VirtualizedTransactionList
                  transactions={memoizedTransactions}
                  onViewDetails={handleViewTransactionDetails}
                  emptyMessage="No transactions to display"
                />
              )}
            </div>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-4 space-y-6">
          <div className="grid grid-cols-1 gap-6">
            {/* Connected Wallets */}
            <Card title="Connected Wallets">
              <div className="space-y-4">
                {wallets.map((wallet) => (
                  <div
                    key={wallet.id}
                    className={`p-4 rounded-lg cursor-pointer transition-all duration-300 ${
                      selectedWallet.id === wallet.id
                        ? 'bg-primary-100 dark:bg-primary-900/20 shadow-md'
                        : 'bg-primary-50 dark:bg-dark-background hover:bg-primary-100 dark:hover:bg-primary-900/20 hover:shadow-md hover:translate-y-[-2px]'
                    }`}
                    onClick={() => setSelectedWallet(wallet)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-primary-900 dark:text-primary-100">
                        {wallet.name}
                      </span>
                      <Tooltip content="Delete wallet">
                        <Button
                          variant="ghost"
                          size="xs"
                          color="error"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteWallet(wallet.id, wallet.name);
                          }}
                        >
                          <Trash2 size={16} />
                        </Button>
                      </Tooltip>
                    </div>
                    <p className="text-sm text-primary-600 dark:text-primary-400">
                      {formatAddress(wallet.address)}
                    </p>
                    <p className="text-sm font-medium text-primary-900 dark:text-primary-100 mt-2">
                      {wallet.balance} ETH
                    </p>
                  </div>
                ))}
              </div>
            </Card>

            {/* Security Settings */}
            <Card title="Security Settings">
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Shield size={20} className="text-primary-500" />
                    <span className="ml-2 text-sm font-medium text-primary-900 dark:text-primary-100">
                      Drain Protection
                    </span>
                    <Tooltip content="Prevents large unauthorized withdrawals from your wallet">
                      <Info size={16} className="ml-1 text-primary-400" />
                    </Tooltip>
                  </div>
                  <Switch
                    checked={drainProtection}
                    onChange={(checked) => {
                      setDrainProtection(checked);
                      showToast(
                        checked ? 'success' : 'warning',
                        checked ? 'Protection enabled' : 'Protection disabled',
                        checked ? 'Drain protection has been enabled' : 'Drain protection has been disabled'
                      );
                    }}
                    size="md"
                    color="primary"
                  />
                </div>

                {drainProtection && (
                  <Alert
                    status="warning"
                    variant="subtle"
                    icon={<AlertTriangle size={20} />}
                    title="Maximum transaction limits enabled"
                  >
                    <div className="text-sm">
                      <p className="mb-1">The following limits are currently active:</p>
                      <ul className="list-disc pl-5 space-y-1">
                        <li>Single transaction: <strong>5 ETH</strong></li>
                        <li>Daily limit: <strong>20 ETH</strong></li>
                      </ul>
                      <Button
                        variant="link"
                        size="sm"
                        color="warning"
                        className="mt-2 p-0"
                        onClick={() => showToast('info', 'Limit settings', 'Opening limit configuration')}
                      >
                        Configure limits
                      </Button>
                    </div>
                  </Alert>
                )}

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-sm text-primary-600 dark:text-primary-400">Transaction Signing</span>
                      <Tooltip content="Requires confirmation for all transactions">
                        <Info size={14} className="ml-1 text-primary-400" />
                      </Tooltip>
                    </div>
                    <Badge variant="subtle" color="success">Enabled</Badge>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-sm text-primary-600 dark:text-primary-400">Hardware Wallet Support</span>
                      <Tooltip content="Connect hardware wallets like Ledger or Trezor">
                        <Info size={14} className="ml-1 text-primary-400" />
                      </Tooltip>
                    </div>
                    <Badge variant="subtle" color="success">Active</Badge>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-sm text-primary-600 dark:text-primary-400">Address Whitelist</span>
                      <Tooltip content="Limit transactions to approved addresses only">
                        <Info size={14} className="ml-1 text-primary-400" />
                      </Tooltip>
                    </div>
                    <Button
                      variant="link"
                      size="sm"
                      onClick={() => showToast('info', 'Address whitelist', 'Opening whitelist management')}
                    >
                      Manage
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>

      <CreateWalletModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreateWallet={handleCreateWallet}
      />
    </Layout>
  );
};

export default WalletPage;