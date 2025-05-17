import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Code2, Upload, AlertTriangle, CheckCircle, Copy, ExternalLink, Search, Plus, Shield, X, RefreshCw } from 'lucide-react';
import Layout from '../components/layout/Layout';
import Card from '../components/ui/Card';

interface SmartContract {
  id: string;
  name: string;
  address: string;
  network: string;
  status: 'verified' | 'unverified' | 'error';
  deployedAt: string;
  lastInteraction: string;
  version: string;
}

interface AuditResult {
  severity: 'high' | 'medium' | 'low' | 'info';
  message: string;
  line: number;
  description: string;
}

const mockContracts: SmartContract[] = [
  {
    id: '1',
    name: 'TokenVesting',
    address: '0x1234567890abcdef1234567890abcdef12345678',
    network: 'Ethereum',
    status: 'verified',
    deployedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    lastInteraction: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    version: '1.0.0'
  },
  {
    id: '2',
    name: 'StakingPool',
    address: '0xabcdef1234567890abcdef1234567890abcdef12',
    network: 'Arbitrum',
    status: 'verified',
    deployedAt: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
    lastInteraction: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
    version: '2.1.0'
  }
];

const mockAuditResults: AuditResult[] = [
  {
    severity: 'high',
    message: 'Reentrancy vulnerability detected',
    line: 45,
    description: 'External call is made before state update. Consider using ReentrancyGuard or updating state before external calls.'
  },
  {
    severity: 'medium',
    message: 'Unchecked return value',
    line: 78,
    description: 'The return value of an external call is not checked. This could lead to silent failures.'
  },
  {
    severity: 'low',
    message: 'Consider using SafeMath',
    line: 92,
    description: 'Although Solidity 0.8.x includes overflow checks, explicit SafeMath usage improves readability and maintainability.'
  }
];

const DeployContractModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
}> = ({ isOpen, onClose }) => {
  const [contractName, setContractName] = useState('');
  const [contractCode, setContractCode] = useState('');
  const [selectedNetwork, setSelectedNetwork] = useState('ethereum');
  const [isDeploying, setIsDeploying] = useState(false);
  const [currentStep, setCurrentStep] = useState<'edit' | 'audit' | 'deploy'>('edit');
  const [auditResults, setAuditResults] = useState<AuditResult[]>([]);

  const handleAudit = async () => {
    setIsDeploying(true);
    try {
      // Simulate contract auditing
      await new Promise(resolve => setTimeout(resolve, 2000));
      setAuditResults(mockAuditResults);
      setCurrentStep('audit');
    } finally {
      setIsDeploying(false);
    }
  };

  const handleDeploy = async () => {
    setIsDeploying(true);
    try {
      // Simulate contract deployment
      await new Promise(resolve => setTimeout(resolve, 3000));
      onClose();
    } finally {
      setIsDeploying(false);
    }
  };

  const getSeverityColor = (severity: AuditResult['severity']) => {
    switch (severity) {
      case 'high':
        return 'text-error-500 bg-error-50 dark:bg-error-900/20';
      case 'medium':
        return 'text-warning-500 bg-warning-50 dark:bg-warning-900/20';
      case 'low':
        return 'text-primary-500 bg-primary-50 dark:bg-primary-900/20';
      default:
        return 'text-primary-500 bg-primary-50 dark:bg-primary-900/20';
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/50"
            onClick={onClose}
          />
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="relative bg-white dark:bg-dark-card rounded-lg shadow-xl w-full max-w-4xl p-6 max-h-[90vh] overflow-y-auto"
          >
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100">
                {currentStep === 'edit' && 'Deploy New Smart Contract'}
                {currentStep === 'audit' && 'Contract Audit Results'}
                {currentStep === 'deploy' && 'Deploy Contract'}
              </h3>
              <button
                onClick={onClose}
                className="p-1 rounded-lg hover:bg-primary-50 dark:hover:bg-dark-background"
              >
                <X size={20} className="text-primary-500" />
              </button>
            </div>

            {currentStep === 'edit' && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-primary-700 dark:text-primary-300 mb-2">
                      Contract Name
                    </label>
                    <input
                      type="text"
                      value={contractName}
                      onChange={(e) => setContractName(e.target.value)}
                      placeholder="Enter contract name"
                      className="w-full px-3 py-2 border border-primary-200 dark:border-dark-border rounded-lg bg-white dark:bg-dark-card text-primary-900 dark:text-primary-100"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-primary-700 dark:text-primary-300 mb-2">
                      Network
                    </label>
                    <select
                      value={selectedNetwork}
                      onChange={(e) => setSelectedNetwork(e.target.value)}
                      className="w-full px-3 py-2 border border-primary-200 dark:border-dark-border rounded-lg bg-white dark:bg-dark-card text-primary-900 dark:text-primary-100"
                    >
                      <option value="ethereum">Ethereum</option>
                      <option value="arbitrum">Arbitrum</option>
                      <option value="optimism">Optimism</option>
                      <option value="base">Base</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-primary-700 dark:text-primary-300 mb-2">
                    Contract Code
                  </label>
                  <div className="relative">
                    <textarea
                      value={contractCode}
                      onChange={(e) => setContractCode(e.target.value)}
                      placeholder="Paste your Solidity contract code here..."
                      className="w-full h-96 px-3 py-2 border border-primary-200 dark:border-dark-border rounded-lg bg-white dark:bg-dark-card text-primary-900 dark:text-primary-100 font-mono"
                    />
                  </div>
                </div>

                <div className="flex justify-end space-x-3">
                  <button
                    onClick={onClose}
                    className="px-4 py-2 text-primary-700 dark:text-primary-300 hover:bg-primary-50 dark:hover:bg-dark-background rounded-lg"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleAudit}
                    disabled={!contractName || !contractCode || isDeploying}
                    className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                  >
                    {isDeploying ? (
                      <>
                        <motion.span
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                          className="w-4 h-4 border-2 border-white border-t-transparent rounded-full inline-block mr-2"
                        />
                        Auditing...
                      </>
                    ) : (
                      <>
                        <Shield size={18} className="mr-2" />
                        Audit Contract
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {currentStep === 'audit' && (
              <div className="space-y-6">
                <div className="space-y-4">
                  {auditResults.map((result, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg ${getSeverityColor(result.severity)}`}
                    >
                      <div className="flex items-start">
                        <AlertTriangle size={20} className="mt-0.5" />
                        <div className="ml-3">
                          <div className="flex items-center">
                            <p className="font-medium">{result.message}</p>
                            <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-white/20">
                              Line {result.line}
                            </span>
                          </div>
                          <p className="text-sm mt-1">{result.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex justify-end space-x-3">
                  <button
                    onClick={() => setCurrentStep('edit')}
                    className="px-4 py-2 text-primary-700 dark:text-primary-300 hover:bg-primary-50 dark:hover:bg-dark-background rounded-lg"
                  >
                    Back to Editor
                  </button>
                  <button
                    onClick={handleDeploy}
                    disabled={isDeploying}
                    className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                  >
                    {isDeploying ? (
                      <>
                        <motion.span
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                          className="w-4 h-4 border-2 border-white border-t-transparent rounded-full inline-block mr-2"
                        />
                        Deploying...
                      </>
                    ) : (
                      <>
                        <Upload size={18} className="mr-2" />
                        Deploy Contract
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

const SmartContractsPage: React.FC = () => {
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNetwork, setSelectedNetwork] = useState('all');

  const formatAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const getStatusColor = (status: SmartContract['status']) => {
    switch (status) {
      case 'verified':
        return 'text-success-500';
      case 'unverified':
        return 'text-warning-500';
      case 'error':
        return 'text-error-500';
    }
  };

  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          <Code2 className="w-8 h-8 text-primary-600 dark:text-primary-400 mr-3" />
          <h1 className="text-2xl font-bold text-primary-900 dark:text-primary-100">Smart Contracts</h1>
        </div>
        <button 
          className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
          onClick={() => setShowDeployModal(true)}
        >
          <Plus size={20} className="mr-2" />
          Deploy Contract
        </button>
      </div>

      <Card>
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-6">
          <div className="relative flex-1 max-w-lg mb-4 lg:mb-0">
            <Search size={20} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-primary-400" />
            <input
              type="text"
              placeholder="Search contracts by name or address..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-primary-200 dark:border-dark-border rounded-lg bg-white dark:bg-dark-card text-primary-900 dark:text-primary-100"
            />
          </div>
          <div className="flex space-x-4">
            <select
              value={selectedNetwork}
              onChange={(e) => setSelectedNetwork(e.target.value)}
              className="px-3 py-2 border border-primary-200 dark:border-dark-border rounded-lg bg-white dark:bg-dark-card text-primary-900 dark:text-primary-100"
            >
              <option value="all">All Networks</option>
              <option value="ethereum">Ethereum</option>
              <option value="arbitrum">Arbitrum</option>
              <option value="optimism">Optimism</option>
              <option value="base">Base</option>
            </select>
            <button className="p-2 hover:bg-primary-50 dark:hover:bg-dark-background rounded-lg">
              <RefreshCw size={20} className="text-primary-500" />
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-primary-200 dark:divide-dark-border">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-primary-500 dark:text-primary-400 uppercase tracking-wider">
                  Contract
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-primary-500 dark:text-primary-400 uppercase tracking-wider">
                  Network
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-primary-500 dark:text-primary-400 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-primary-500 dark:text-primary-400 uppercase tracking-wider">
                  Deployed
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-primary-500 dark:text-primary-400 uppercase tracking-wider">
                  Last Interaction
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-primary-500 dark:text-primary-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-primary-200 dark:divide-dark-border">
              {mockContracts.map((contract) => (
                <tr key={contract.id} className="hover:bg-primary-50 dark:hover:bg-dark-background">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div>
                      <p className="text-sm font-medium text-primary-900 dark:text-primary-100">
                        {contract.name}
                      </p>
                      <div className="flex items-center mt-1">
                        <p className="text-xs text-primary-500">{formatAddress(contract.address)}</p>
                        <button className="ml-2 text-primary-400 hover:text-primary-600">
                          <Copy size={14} />
                        </button>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm text-primary-900 dark:text-primary-100">
                      {contract.network}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <CheckCircle size={16} className={getStatusColor(contract.status)} />
                      <span className={`ml-2 text-sm ${getStatusColor(contract.status)}`}>
                        {contract.status.charAt(0).toUpperCase() + contract.status.slice(1)}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm text-primary-600 dark:text-primary-400">
                      {new Date(contract.deployedAt).toLocaleDateString()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm text-primary-600 dark:text-primary-400">
                      {new Date(contract.lastInteraction).toLocaleDateString()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <button className="p-2 hover:bg-primary-100 dark:hover:bg-dark-background rounded-lg">
                        <ExternalLink size={16} className="text-primary-500" />
                      </button>
                      <button className="p-2 hover:bg-primary-100 dark:hover:bg-dark-background rounded-lg">
                        <Code2 size={16} className="text-primary-500" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <DeployContractModal
        isOpen={showDeployModal}
        onClose={() => setShowDeployModal(false)}
      />
    </Layout>
  );
};

export default SmartContractsPage;