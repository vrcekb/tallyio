export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'operator' | 'viewer';
  lastLogin: string;
  twoFactorEnabled: boolean;
}

export interface APIKey {
  id: string;
  name: string;
  key: string;
  created: string;
  lastUsed: string;
  permissions: string[];
}

export interface NotificationSetting {
  type: string;
  email: boolean;
  push: boolean;
  slack: boolean;
  telegram: boolean;
}

export interface IntegrationConfig {
  id: string;
  name: string;
  type: 'exchange' | 'data' | 'notification';
  status: 'connected' | 'disconnected' | 'error';
  lastSync: string;
}

export const userProfile: UserProfile = {
  id: 'user-1',
  name: 'Admin User',
  email: 'admin@example.com',
  role: 'admin',
  lastLogin: new Date(Date.now() - 3600000).toISOString(),
  twoFactorEnabled: true
};

export const apiKeys: APIKey[] = [
  {
    id: 'key-1',
    name: 'Production API Key',
    key: '********-****-****-****-************',
    created: new Date(Date.now() - 7776000000).toISOString(),
    lastUsed: new Date(Date.now() - 3600000).toISOString(),
    permissions: ['read', 'write', 'execute']
  },
  {
    id: 'key-2',
    name: 'Monitoring Key',
    key: '********-****-****-****-************',
    created: new Date(Date.now() - 2592000000).toISOString(),
    lastUsed: new Date(Date.now() - 86400000).toISOString(),
    permissions: ['read']
  }
];

export const notificationSettings: NotificationSetting[] = [
  {
    type: 'System Alerts',
    email: true,
    push: true,
    slack: true,
    telegram: false
  },
  {
    type: 'Strategy Updates',
    email: true,
    push: false,
    slack: true,
    telegram: false
  },
  {
    type: 'Performance Reports',
    email: true,
    push: false,
    slack: false,
    telegram: false
  },
  {
    type: 'Security Alerts',
    email: true,
    push: true,
    slack: true,
    telegram: true
  }
];

export const integrationConfigs: IntegrationConfig[] = [
  {
    id: 'int-1',
    name: 'Binance',
    type: 'exchange',
    status: 'connected',
    lastSync: new Date(Date.now() - 300000).toISOString()
  },
  {
    id: 'int-2',
    name: 'CoinGecko',
    type: 'data',
    status: 'connected',
    lastSync: new Date(Date.now() - 600000).toISOString()
  },
  {
    id: 'int-3',
    name: 'Slack',
    type: 'notification',
    status: 'connected',
    lastSync: new Date(Date.now() - 900000).toISOString()
  },
  {
    id: 'int-4',
    name: 'Telegram',
    type: 'notification',
    status: 'disconnected',
    lastSync: new Date(Date.now() - 86400000).toISOString()
  }
];