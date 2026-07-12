# Lettuce Biomass Growth Visualization

Interactive scientific visualization for UC Davis CEE Lab research on mathematical modeling of lettuce biomass accumulation.

## Quick Start

```bash
cd lettuce-growth-viz
npm install
npm run dev
```

Open the URL shown in your terminal (usually `http://localhost:5173`).

## Project Structure

```
src/
├── data/           # Your research data (edit biomassData.js to update values)
├── components/     # UI sections (landing, farm, graph, dashboard, etc.)
├── animations/     # Framer Motion variants and transitions
└── styles/         # Design tokens and theme
```

## Updating Your Data

Edit `src/data/biomassData.js` — replace the arrays with your new measurements.

## Tech Stack

- React + Vite
- Tailwind CSS
- Framer Motion
- Plotly.js

## Research

American High School (Fremont, CA) & UC Davis Department of Biological and Agricultural Engineering
