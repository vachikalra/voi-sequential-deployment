/**
 * UC Davis CEE Lab — Lettuce Biomass Data
 * ========================================
 * This file contains your REAL research data.
 *
 * TO UPDATE YOUR DATA LATER:
 * Replace the values in the arrays below with your new measurements.
 * Each array must have exactly 30 entries (Days 1–30).
 *
 * Data source: Lettuce_Biomass_Vachi (1).xlsx — "Average" column
 * Columns B22–B47 averaged across 18 interior plant columns
 */

export const biomassData = {
  /** Days After Transplanting (DAT) — 1 through 30 */
  days: [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
  ],

  /** Observed average biomass (grams) — measured daily */
  observed: [
    3.3722, 7.9833, 9.6222, 10.2889, 12.8444, 15.6167, 18.6111, 21.9667, 26.2167, 31.0444,
    35.7278, 40.3889, 45.8333, 52.4222, 60.2556, 69.1444, 77.8444, 84.3389, 93.1889, 100.55,
    108.5333, 114.8389, 120.7389, 127.1389, 134.4556, 141.4, 146.1889, 141.6944, 148.2056, 152.8444,
  ],

  /** Linear Regression predictions (grams) */
  linear: [
    -12.3121, -6.5129, -0.7136, 5.0856, 10.8848, 16.684, 22.4833, 28.2825, 34.0817, 39.8809,
    45.6802, 51.4794, 57.2786, 63.0778, 68.8771, 74.6763, 80.4755, 86.2747, 92.074, 97.8732,
    103.6724, 109.4716, 115.2708, 121.0701, 126.8693, 132.6685, 138.4677, 144.267, 150.0662, 155.8654,
  ],

  /** Polynomial Regression (degree 2) predictions (grams) */
  polynomial: [
    -3.085, 0.8051, 4.8317, 8.9946, 13.2938, 17.7295, 22.3015, 27.0098, 31.8545, 36.8356,
    41.953, 47.2068, 52.5969, 58.1234, 63.7863, 69.5855, 75.5211, 81.593, 87.8013, 94.146,
    100.627, 107.2444, 113.9981, 120.8883, 127.9147, 135.0775, 142.3767, 149.8123, 157.3842, 165.0925,
  ],

  /** Logistic Growth Model predictions (grams) */
  logistic: [
    6.1237, 7.3786, 8.8763, 10.6577, 12.7676, 15.2544, 18.1683, 21.5594, 25.4747, 29.9542,
    35.0256, 40.6997, 46.9646, 53.7815, 61.0822, 68.7692, 76.7194, 84.7911, 92.8338, 100.6989,
    108.251, 115.376, 121.9876, 128.0288, 133.4718, 138.3137, 142.5727, 146.2817, 149.4839, 152.2279,
  ],

  /** Gompertz Growth Model predictions (grams) */
  gompertz: [
    2.6046, 3.8432, 5.4787, 7.569, 10.1619, 13.2919, 16.9776, 21.2205, 26.0053, 31.3005,
    37.0609, 43.2302, 49.7438, 56.5322, 63.5235, 70.6466, 77.8328, 85.0178, 92.1431, 99.1566,
    106.0134, 112.6755, 119.1119, 125.2984, 131.2168, 136.8545, 142.2039, 147.2618, 152.0285, 156.5076,
  ],
};

/** Model parameter values from curve fitting */
export const modelParameters = {
  linear: { beta0: -18.1113, beta1: 5.7992 },
  polynomial: { beta0: -6.8388, beta1: 3.6856, beta2: 0.0682 },
  logistic: { A: 166.4804, B: 31.8009, k: 0.1943 },
  gompertz: { A: 211.0526, b: 1.5731, k: 0.0927 },
};

/** Helper: get data for a specific day (1-indexed) */
export function getDayData(day) {
  const index = day - 1;
  if (index < 0 || index >= 30) return null;

  return {
    day,
    observed: biomassData.observed[index],
    linear: biomassData.linear[index],
    polynomial: biomassData.polynomial[index],
    logistic: biomassData.logistic[index],
    gompertz: biomassData.gompertz[index],
  };
}

/** Helper: get prediction errors (absolute) for a specific day */
export function getDayErrors(day) {
  const data = getDayData(day);
  if (!data) return null;

  return {
    linear: Math.abs(data.observed - data.linear),
    polynomial: Math.abs(data.observed - data.polynomial),
    logistic: Math.abs(data.observed - data.logistic),
    gompertz: Math.abs(data.observed - data.gompertz),
  };
}

/** Total number of simulation days */
export const TOTAL_DAYS = 30;
