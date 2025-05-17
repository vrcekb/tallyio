export interface MLModel {
  id: string;
  name: string;
  type: 'classification' | 'regression' | 'clustering';
  status: 'training' | 'deployed' | 'stopped';
  accuracy: number;
  lastTrained: string;
  version: string;
}

export interface TrainingMetric {
  epoch: number;
  accuracy: number;
  loss: number;
  validationAccuracy: number;
  validationLoss: number;
}

export interface PredictionMetrics {
  title: string;
  value: number;
  change: number;
  unit?: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  correlation: number;
}

export const mlModels: MLModel[] = [
  {
    id: 'model-1',
    name: 'Price Direction Predictor',
    type: 'classification',
    status: 'deployed',
    accuracy: 82.4,
    lastTrained: new Date(Date.now() - 86400000).toISOString(),
    version: '1.2.0'
  },
  {
    id: 'model-2',
    name: 'Volatility Forecaster',
    type: 'regression',
    status: 'deployed',
    accuracy: 78.6,
    lastTrained: new Date(Date.now() - 172800000).toISOString(),
    version: '1.1.0'
  },
  {
    id: 'model-3',
    name: 'Market Regime Classifier',
    type: 'clustering',
    status: 'training',
    accuracy: 75.2,
    lastTrained: new Date(Date.now() - 259200000).toISOString(),
    version: '0.9.0'
  }
];

export const generateTrainingMetrics = (epochs: number): TrainingMetric[] => {
  const metrics: TrainingMetric[] = [];
  let accuracy = 50;
  let loss = 1;
  let valAccuracy = 50;
  let valLoss = 1;

  for (let i = 1; i <= epochs; i++) {
    accuracy += Math.random() * 2;
    loss -= Math.random() * 0.05;
    valAccuracy += Math.random() * 1.8;
    valLoss -= Math.random() * 0.04;

    metrics.push({
      epoch: i,
      accuracy: Math.min(accuracy, 95),
      loss: Math.max(loss, 0.1),
      validationAccuracy: Math.min(valAccuracy, 90),
      validationLoss: Math.max(valLoss, 0.15)
    });
  }

  return metrics;
};

export const predictionMetrics: PredictionMetrics[] = [
  {
    title: 'Average Accuracy',
    value: 82.4,
    change: 1.2,
    unit: '%'
  },
  {
    title: 'Prediction Latency',
    value: 124,
    change: -15.3,
    unit: 'ms'
  },
  {
    title: 'Active Models',
    value: 2,
    change: 0
  },
  {
    title: 'Daily Predictions',
    value: 12458,
    change: 8.5
  }
];

export const featureImportance: FeatureImportance[] = [
  {
    feature: 'Price Momentum',
    importance: 0.85,
    correlation: 0.78
  },
  {
    feature: 'Volume',
    importance: 0.72,
    correlation: 0.65
  },
  {
    feature: 'Market Volatility',
    importance: 0.68,
    correlation: 0.58
  },
  {
    feature: 'Order Book Imbalance',
    importance: 0.64,
    correlation: 0.52
  },
  {
    feature: 'Network Activity',
    importance: 0.58,
    correlation: 0.45
  }
];