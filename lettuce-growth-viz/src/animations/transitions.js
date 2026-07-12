/**
 * Shared transition presets for Framer Motion.
 */

export const smooth = { duration: 0.6, ease: [0.22, 1, 0.36, 1] };

export const spring = { type: 'spring', stiffness: 300, damping: 30 };

export const slowSpring = { type: 'spring', stiffness: 120, damping: 20 };

export const simulationStep = {
  duration: 0.4,
  ease: 'easeInOut',
};

/** Duration between each simulation day (milliseconds) */
export const DAY_INTERVAL_MS = 400;

/** Total simulation duration */
export const SIMULATION_DURATION_MS = 30 * DAY_INTERVAL_MS;
