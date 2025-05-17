import React from 'react';
import { Brain } from 'lucide-react';
import Layout from '../components/layout/Layout';
import Card from '../components/ui/Card';

function MLPage() {
  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Machine Learning
        </h1>
        <div className="flex items-center space-x-4">
          <select className="bg-white dark:bg-dark-card border border-gray-200 dark:border-dark-border rounded-lg px-3 py-2 text-sm">
            <option value="all">All Models</option>
            <option value="active">Active Models</option>
            <option value="training">Training Models</option>
          </select>
        </div>
      </div>

      <div className="grid gap-6">
        <Card title="ML Models Overview">
          <p className="text-gray-700 dark:text-gray-300">Machine Learning dashboard content will be displayed here.</p>
        </Card>
      </div>
    </Layout>
  );
}

export default MLPage;