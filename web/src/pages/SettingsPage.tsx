import React, { useState } from 'react';
import {
  User,
  Key,
  Bell,
  Link,
  Shield,
  Copy,
  Check,
  RefreshCw,
  Trash,
  Plus,
  Info
} from 'lucide-react';
import Layout from '../components/layout/Layout';
import Card from '../components/ui/Card';
import Tabs from '../components/ui/Tabs';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import Switch from '../components/ui/Switch';
import Badge from '../components/ui/Badge';
import Tooltip from '../components/ui/Tooltip';
import Alert from '../components/ui/Alert';
import { useToast } from '../contexts/ToastContext';
import {
  userProfile,
  apiKeys,
  notificationSettings,
  integrationConfigs
} from '../mockData/settings';

type TabType = 'profile' | 'api' | 'notifications' | 'integrations';

const SettingsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('profile');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [showAlert, setShowAlert] = useState(true);
  const { showToast } = useToast();

  const copyToClipboard = (key: string, id: string) => {
    navigator.clipboard.writeText(key);
    setCopiedKey(id);
    setTimeout(() => setCopiedKey(null), 2000);
    showToast('success', 'Copied to clipboard', 'API key has been copied to clipboard');
  };

  const handleSaveProfile = () => {
    showToast('success', 'Profile updated', 'Your profile has been successfully updated');
  };

  const handleCreateApiKey = () => {
    showToast('info', 'Creating API key', 'Your new API key is being generated');
    // Simulirajmo zamik
    setTimeout(() => {
      showToast('success', 'API key created', 'Your new API key has been successfully created');
    }, 1500);
  };

  const handleDeleteApiKey = (keyName: string) => {
    showToast('warning', 'Deleting API key', `Are you sure you want to delete ${keyName}?`);
    // Simulirajmo zamik
    setTimeout(() => {
      showToast('success', 'API key deleted', `${keyName} has been successfully deleted`);
    }, 1500);
  };

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'profile', label: 'User Profile', icon: <User size={18} /> },
    { id: 'api', label: 'API Keys', icon: <Key size={18} /> },
    { id: 'notifications', label: 'Notifications', icon: <Bell size={18} /> },
    { id: 'integrations', label: 'Integrations', icon: <Link size={18} /> },
  ];

  const renderProfileTab = () => (
    <Card>
      {showAlert && (
        <Alert
          status="info"
          variant="subtle"
          title="Profile Information"
          onClose={() => setShowAlert(false)}
          className="mb-6"
        >
          Your profile information is used across the platform. Keep it up to date.
        </Alert>
      )}

      <div className="space-y-6">
        <div className="flex items-center">
          <div className="w-20 h-20 rounded-full bg-primary-100 dark:bg-primary-900/20 flex items-center justify-center">
            <User size={32} className="text-primary-600 dark:text-primary-400" />
          </div>
          <div className="ml-6">
            <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100">{userProfile.name}</h3>
            <p className="text-sm text-primary-500 dark:text-primary-400">{userProfile.email}</p>
            <p className="text-sm text-primary-500 dark:text-primary-400 mt-1">
              Last login: {new Date(userProfile.lastLogin).toLocaleString()}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <Input
              label="Full Name"
              defaultValue={userProfile.name}
              fullWidth
            />
          </div>
          <div>
            <Input
              label="Email"
              type="email"
              defaultValue={userProfile.email}
              fullWidth
            />
          </div>
        </div>

        <div className="flex items-center justify-between p-4 bg-primary-50 dark:bg-dark-background rounded-lg">
          <div className="flex items-center">
            <Shield size={20} className="text-primary-500" />
            <span className="ml-2 text-sm font-medium text-primary-900 dark:text-primary-100">
              Two-Factor Authentication
            </span>
            <Tooltip
              content="Two-factor authentication adds an extra layer of security to your account"
              position="top"
            >
              <Info size={16} className="ml-1 text-primary-400" />
            </Tooltip>
          </div>
          <Button
            variant={userProfile.twoFactorEnabled ? 'outline' : 'solid'}
            color={userProfile.twoFactorEnabled ? 'success' : 'primary'}
          >
            {userProfile.twoFactorEnabled ? 'Enabled' : 'Enable 2FA'}
          </Button>
        </div>

        <div className="flex justify-end">
          <Button
            color="primary"
            onClick={handleSaveProfile}
          >
            Save Changes
          </Button>
        </div>
      </div>
    </Card>
  );

  const renderApiTab = () => (
    <Card>
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <p className="text-sm text-primary-500 dark:text-primary-400">
            Manage your API keys for external integrations
          </p>
          <Button
            color="primary"
            leftIcon={<Plus size={16} />}
            onClick={handleCreateApiKey}
          >
            New API Key
          </Button>
        </div>

        <div className="space-y-4">
          {apiKeys.map((key) => (
            <div key={key.id} className="p-4 bg-primary-50 dark:bg-dark-background rounded-lg transition-all duration-300 hover:shadow-md">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h4 className="font-medium text-primary-900 dark:text-primary-100">{key.name}</h4>
                  <p className="text-sm text-primary-500 dark:text-primary-400">
                    Created: {new Date(key.created).toLocaleDateString()}
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <Tooltip content="Copy API key">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(key.key, key.id)}
                    >
                      {copiedKey === key.id ? (
                        <Check size={16} className="text-success-500" />
                      ) : (
                        <Copy size={16} className="text-primary-500" />
                      )}
                    </Button>
                  </Tooltip>
                  <Tooltip content="Regenerate API key">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => showToast('info', 'Regenerating key', 'Your API key is being regenerated')}
                    >
                      <RefreshCw size={16} className="text-primary-500" />
                    </Button>
                  </Tooltip>
                  <Tooltip content="Delete API key">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteApiKey(key.name)}
                    >
                      <Trash size={16} className="text-error-500" />
                    </Button>
                  </Tooltip>
                </div>
              </div>
              <div className="flex items-center space-x-2 text-sm">
                <code className="px-2 py-1 bg-primary-100 dark:bg-dark-card rounded text-primary-600 dark:text-primary-300">
                  {key.key}
                </code>
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                {key.permissions.map((permission) => (
                  <Badge
                    key={permission}
                    variant="subtle"
                    color="primary"
                    size="sm"
                  >
                    {permission}
                  </Badge>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );

  const renderNotificationsTab = () => (
    <Card>
      <div className="space-y-4">
        {notificationSettings.map((setting) => (
          <div key={setting.type} className="p-4 bg-primary-50 dark:bg-dark-background rounded-lg transition-all duration-300 hover:shadow-md">
            <h4 className="font-medium text-primary-900 dark:text-primary-100 mb-3">{setting.type}</h4>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-sm text-primary-700 dark:text-primary-300">Email</span>
                  <Tooltip content="Receive notifications via email">
                    <Info size={14} className="ml-1 text-primary-400" />
                  </Tooltip>
                </div>
                <Switch
                  checked={setting.email}
                  onChange={() => showToast('success', 'Settings updated', 'Your notification preferences have been updated')}
                  size="sm"
                  color="primary"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-sm text-primary-700 dark:text-primary-300">Push</span>
                  <Tooltip content="Receive push notifications in your browser">
                    <Info size={14} className="ml-1 text-primary-400" />
                  </Tooltip>
                </div>
                <Switch
                  checked={setting.push}
                  onChange={() => showToast('success', 'Settings updated', 'Your notification preferences have been updated')}
                  size="sm"
                  color="primary"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-sm text-primary-700 dark:text-primary-300">Slack</span>
                  <Tooltip content="Receive notifications in your Slack workspace">
                    <Info size={14} className="ml-1 text-primary-400" />
                  </Tooltip>
                </div>
                <Switch
                  checked={setting.slack}
                  onChange={() => showToast('success', 'Settings updated', 'Your notification preferences have been updated')}
                  size="sm"
                  color="primary"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-sm text-primary-700 dark:text-primary-300">Telegram</span>
                  <Tooltip content="Receive notifications in your Telegram">
                    <Info size={14} className="ml-1 text-primary-400" />
                  </Tooltip>
                </div>
                <Switch
                  checked={setting.telegram}
                  onChange={() => showToast('success', 'Settings updated', 'Your notification preferences have been updated')}
                  size="sm"
                  color="primary"
                />
              </div>
            </div>
          </div>
        ))}

        <div className="flex justify-end mt-4">
          <Button
            color="primary"
            onClick={() => showToast('success', 'Notification settings saved', 'Your notification preferences have been updated')}
          >
            Save Preferences
          </Button>
        </div>
      </div>
    </Card>
  );

  const renderIntegrationsTab = () => (
    <Card>
      <div className="space-y-4">
        {integrationConfigs.map((integration) => (
          <div key={integration.id} className="p-4 bg-primary-50 dark:bg-dark-background rounded-lg transition-all duration-300 hover:shadow-md">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-primary-900 dark:text-primary-100">{integration.name}</h4>
              <Badge
                variant="subtle"
                color={
                  integration.status === 'connected'
                    ? 'success'
                    : integration.status === 'error'
                    ? 'error'
                    : 'primary'
                }
              >
                {integration.status}
              </Badge>
            </div>
            <p className="text-sm text-primary-500 dark:text-primary-400">
              Last sync: {new Date(integration.lastSync).toLocaleString()}
            </p>
            <div className="mt-4 flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => showToast('info', 'Configuring integration', `Opening configuration for ${integration.name}`)}
              >
                Configure
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => showToast('info', 'Syncing', `Syncing data with ${integration.name}`)}
              >
                Sync Now
              </Button>

              {integration.status === 'connected' ? (
                <Button
                  variant="ghost"
                  size="sm"
                  color="error"
                  onClick={() => showToast('warning', 'Disconnecting', `Disconnecting from ${integration.name}`)}
                >
                  Disconnect
                </Button>
              ) : (
                <Button
                  variant="ghost"
                  size="sm"
                  color="success"
                  onClick={() => showToast('info', 'Connecting', `Connecting to ${integration.name}`)}
                >
                  Connect
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'profile':
        return renderProfileTab();
      case 'api':
        return renderApiTab();
      case 'notifications':
        return renderNotificationsTab();
      case 'integrations':
        return renderIntegrationsTab();
      default:
        return null;
    }
  };

  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
      </div>

      <div className="max-w-6xl mx-auto">
        <Tabs
          tabs={tabs.map(tab => ({
            id: tab.id,
            label: tab.label,
            icon: tab.icon,
            content: renderActiveTab()
          }))}
          defaultTab="profile"
          onChange={(tabId) => setActiveTab(tabId as TabType)}
          variant="enclosed"
          color="primary"
        />
      </div>
    </Layout>
  );
};

export default SettingsPage;