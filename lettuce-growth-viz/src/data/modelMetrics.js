/**
 * Statistical performance metrics for each growth model.
 * Values computed from UC Davis CEE Lab dataset.
 */

export const modelMetrics = {
  linear: {
    r2: 0.9778,
    rmse: 7.5547,
    mape: 36.93,
  },
  polynomial: {
    r2: 0.9859,
    rmse: 6.0226,
    mape: 19.49,
  },
  logistic: {
    r2: 0.9991,
    rmse: 1.4874,
    mape: 4.62,
  },
  gompertz: {
    r2: 0.9969,
    rmse: 2.8274,
    mape: 8.53,
  },
};

/** The best-performing model */
export const bestModel = 'logistic';

/** Metric display labels */
export const metricLabels = {
  r2: { label: 'R²', description: 'Coefficient of Determination', higherIsBetter: true },
  rmse: { label: 'RMSE', description: 'Root Mean Squared Error (g)', higherIsBetter: false },
  mape: { label: 'MAPE', description: 'Mean Absolute Percentage Error', higherIsBetter: false },
};
