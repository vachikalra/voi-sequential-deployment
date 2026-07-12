/**
 * Biological growth phases for lettuce biomass accumulation.
 * Based on S-shaped (sigmoidal) growth pattern observed in CEA systems.
 */

export const growthPhases = [
  {
    id: 'lag',
    name: 'Lag Phase',
    dayStart: 1,
    dayEnd: 8,
    color: '#60a5fa',
    description:
      'After transplanting, lettuce adapts to its new hydroponic environment. Root systems establish while biomass accumulates slowly.',
    icon: '🌱',
  },
  {
    id: 'rapid',
    name: 'Rapid Growth',
    dayStart: 9,
    dayEnd: 22,
    color: '#34d399',
    description:
      'Leaf expansion accelerates dramatically. Photosynthesis drives rapid biomass accumulation as the plant enters its main vegetative stage.',
    icon: '📈',
  },
  {
    id: 'maturity',
    name: 'Maturity',
    dayStart: 23,
    dayEnd: 30,
    color: '#a78bfa',
    description:
      'Growth slows as the plant approaches its maximum size. Available space, light, and nutrients become limiting factors.',
    icon: '🏁',
  },
];

/** Get the active growth phase for a given day */
export function getPhaseForDay(day) {
  return growthPhases.find((phase) => day >= phase.dayStart && day <= phase.dayEnd) || growthPhases[0];
}
