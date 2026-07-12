/**
 * Design tokens for the lettuce growth visualization.
 * Central place for colors, spacing, and model-specific styling.
 */

export const colors = {
  background: {
    primary: '#030712',
    secondary: '#0a0f1a',
    card: 'rgba(17, 24, 39, 0.6)',
  },
  text: {
    primary: '#f9fafb',
    secondary: '#9ca3af',
    muted: '#6b7280',
  },
  models: {
    observed: '#ffffff',
    linear: '#60a5fa',
    polynomial: '#fb923c',
    logistic: '#34d399',
    gompertz: '#f87171',
  },
  accent: {
    green: '#34d399',
    blue: '#60a5fa',
    orange: '#fb923c',
    red: '#f87171',
    purple: '#a78bfa',
    cyan: '#22d3ee',
  },
  glow: {
    green: '0 0 20px rgba(52, 211, 153, 0.4)',
    blue: '0 0 20px rgba(96, 165, 250, 0.4)',
    orange: '0 0 20px rgba(251, 146, 60, 0.4)',
    red: '0 0 20px rgba(248, 113, 113, 0.4)',
  },
};

export const modelConfig = {
  linear: {
    id: 'linear',
    name: 'Linear Regression',
    shortName: 'Linear',
    color: colors.models.linear,
    equation: 'y = β₀ + β₁x',
  },
  polynomial: {
    id: 'polynomial',
    name: 'Polynomial Regression',
    shortName: 'Polynomial',
    color: colors.models.polynomial,
    equation: 'y = β₀ + β₁x + β₂x²',
  },
  logistic: {
    id: 'logistic',
    name: 'Logistic Growth',
    shortName: 'Logistic',
    color: colors.models.logistic,
    equation: 'y = A / (1 + B·e^(-kx))',
  },
  gompertz: {
    id: 'gompertz',
    name: 'Gompertz Growth',
    shortName: 'Gompertz',
    color: colors.models.gompertz,
    equation: 'y = A·e^(-e^(b-kx))',
  },
};

export const glassCard = {
  background: 'rgba(17, 24, 39, 0.5)',
  backdropFilter: 'blur(12px)',
  border: '1px solid rgba(255, 255, 255, 0.08)',
  borderRadius: '16px',
};
