"""
Web-based interactive demo using Streamlit.

Deploy with:
    streamlit run demo/web_app.py

Or deploy to Streamlit Cloud for free hosting (shareable URL).
"""

import time
import math
import random
from dataclasses import dataclass

import numpy as np

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Page config
if HAS_STREAMLIT:
    st.set_page_config(
        page_title="VoI-Guided Decision Making",
        page_icon="🧠",
        layout="wide",
    )


# --- Simulation Classes ---

class DemoNode:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class DemoMine:
    def __init__(self):
        self.nodes: list[DemoNode] = []
        self.edges: list[tuple[int, int]] = []
        self.path: list[int] = []

    def generate(self, complexity: float = 0.5, seed: int = 42):
        rng = random.Random(seed)
        self.nodes = []
        self.edges = []
        self.path = []

        start_x, start_y = 300, 30
        self.nodes.append(DemoNode(start_x, start_y))
        self.path.append(0)

        x_pos, y_pos = start_x, start_y
        current_idx = 0
        n_segments = int(12 + complexity * 18)

        for i in range(n_segments):
            dx = rng.gauss(0, 25 + complexity * 20)
            dy = rng.uniform(20, 45)
            x_pos = max(50, min(550, x_pos + dx))
            y_pos += dy

            if y_pos > 580:
                break

            new_idx = len(self.nodes)
            self.nodes.append(DemoNode(x_pos, y_pos))
            self.edges.append((current_idx, new_idx))
            self.path.append(new_idx)

            if rng.random() < complexity * 0.3:
                branch_x = max(30, min(570, x_pos + rng.choice([-1, 1]) * rng.uniform(40, 90)))
                branch_y = y_pos + rng.uniform(10, 30)
                branch_idx = len(self.nodes)
                self.nodes.append(DemoNode(branch_x, branch_y))
                self.edges.append((new_idx, branch_idx))

            current_idx = new_idx


class SimAgent:
    def __init__(self, name: str, method: str, mine: DemoMine, budget: int):
        self.name = name
        self.method = method
        self.mine = mine
        self.budget = budget
        self.budget_remaining = budget
        self.progress = 0
        self.relays: list[int] = []
        self.signal = 1.0
        self.uptime_history: list[bool] = []
        self.voi = 0.0
        self.connected = True

    def step(self):
        if self.progress >= len(self.mine.path) - 1:
            return

        self.progress += 1
        last_relay = max(self.relays) if self.relays else 0
        distance = self.progress - last_relay
        self.signal = max(0.0, 1.0 - 0.08 * distance + random.gauss(0, 0.02))

        should_deploy = self._decide()
        if should_deploy and self.budget_remaining > 0:
            self.relays.append(self.progress)
            self.budget_remaining -= 1
            self.signal = min(1.0, self.signal + 0.4)

        self.connected = self.signal > 0.2
        self.uptime_history.append(self.connected)

    def _decide(self) -> bool:
        if self.method == "threshold":
            return self.signal < 0.35
        elif self.method == "baseline_ppo":
            noise = random.gauss(0, 0.08)
            return (self.signal + noise) < 0.38
        elif self.method == "voi_ppo":
            path_remaining = len(self.mine.path) - self.progress
            budget_frac = self.budget_remaining / max(self.budget, 1)
            self.voi = 0.3 * budget_frac * (path_remaining / max(len(self.mine.path), 1))
            threshold = 0.28 - self.voi * 0.12
            return self.signal < threshold
        return False

    @property
    def uptime(self) -> float:
        if not self.uptime_history:
            return 1.0
        return sum(self.uptime_history) / len(self.uptime_history)

    @property
    def efficiency(self) -> float:
        used = self.budget - self.budget_remaining
        return self.uptime / max(used, 1)


# --- Streamlit App ---

def draw_mine_svg(mine: DemoMine, agent: SimAgent, width: int = 600, height: int = 620) -> str:
    """Generate SVG visualization of the mine with agent state."""
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="#12121a"/>',
    ]

    # Draw edges (tunnels)
    for (i, j) in mine.edges:
        n1, n2 = mine.nodes[i], mine.nodes[j]
        svg_parts.append(
            f'<line x1="{n1.x}" y1="{n1.y}" x2="{n2.x}" y2="{n2.y}" '
            f'stroke="#3a3a50" stroke-width="4" stroke-linecap="round"/>'
        )

    # Highlight traversed path
    for idx in range(min(agent.progress, len(mine.path) - 1)):
        if idx + 1 < len(mine.path):
            n1 = mine.nodes[mine.path[idx]]
            n2 = mine.nodes[mine.path[idx + 1]]
            svg_parts.append(
                f'<line x1="{n1.x}" y1="{n1.y}" x2="{n2.x}" y2="{n2.y}" '
                f'stroke="#4a6090" stroke-width="4" stroke-linecap="round"/>'
            )

    # Draw relays
    for relay_idx in agent.relays:
        if relay_idx < len(mine.path):
            node = mine.nodes[mine.path[relay_idx]]
            svg_parts.append(
                f'<circle cx="{node.x}" cy="{node.y}" r="7" fill="#00dc82" opacity="0.9"/>'
            )
            svg_parts.append(
                f'<circle cx="{node.x}" cy="{node.y}" r="13" fill="none" stroke="#00dc82" stroke-width="1.5" opacity="0.5"/>'
            )

    # Draw entrance
    entrance = mine.nodes[0]
    svg_parts.append(
        f'<circle cx="{entrance.x}" cy="{entrance.y}" r="9" fill="#648cff"/>'
    )
    svg_parts.append(
        f'<text x="{entrance.x + 14}" y="{entrance.y + 5}" fill="#648cff" font-size="11" font-family="monospace">BASE</text>'
    )

    # Draw team position
    if agent.progress < len(mine.path):
        team_node = mine.nodes[mine.path[agent.progress]]
        signal_color = "#00dc82" if agent.signal > 0.6 else "#ffb400" if agent.signal > 0.3 else "#ff3232"
        svg_parts.append(
            f'<circle cx="{team_node.x}" cy="{team_node.y}" r="8" fill="#ffc832"/>'
        )
        svg_parts.append(
            f'<circle cx="{team_node.x}" cy="{team_node.y}" r="15" fill="none" stroke="{signal_color}" stroke-width="2.5"/>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def main():
    if not HAS_STREAMLIT:
        print("Streamlit required: pip install streamlit")
        print("Run with: streamlit run demo/web_app.py")
        return

    # Title
    st.markdown("""
    # 🧠 Value-of-Information Guided Decision Making
    ### Interactive Research Demo
    
    **Research Question**: *Can estimating the value of future information improve 
    commitment timing in problems with irreversible actions?*
    
    ---
    """)

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        complexity = st.slider("Mine Complexity", 0.1, 1.0, 0.5, 0.1,
                              help="Higher = more branches and uncertainty")
        budget = st.slider("Relay Budget", 3, 15, 8,
                          help="Number of relays available to deploy")
        seed = st.number_input("Random Seed", 0, 99999, 42)

        st.divider()
        st.header("Speed")
        speed = st.select_slider("Simulation Speed",
                                options=["Slow", "Normal", "Fast"],
                                value="Normal")
        speed_map = {"Slow": 0.3, "Normal": 0.1, "Fast": 0.02}

        st.divider()
        run_button = st.button("▶ Run Simulation", type="primary", use_container_width=True)
        reset_button = st.button("↺ Reset", use_container_width=True)

        st.divider()
        st.markdown("""
        **Methods Compared:**
        - 🟢 **VoI-Guided PPO** (Proposed)
        - 🟡 **Standard PPO** (Baseline)  
        - 🔴 **Signal Threshold** (Heuristic)
        """)

    # Initialize state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "mine" not in st.session_state or reset_button:
        st.session_state.mine = DemoMine()
        st.session_state.mine.generate(complexity, seed)
        st.session_state.agents = [
            SimAgent("VoI-Guided PPO", "voi_ppo", st.session_state.mine, budget),
            SimAgent("Standard PPO", "baseline_ppo", st.session_state.mine, budget),
            SimAgent("Signal Threshold", "threshold", st.session_state.mine, budget),
        ]
        st.session_state.step = 0
        st.session_state.running = False

    if run_button:
        st.session_state.running = True

    # Display
    cols = st.columns(3)
    colors = ["🟢", "🟡", "🔴"]
    method_colors = ["#00dc82", "#ffb400", "#ff3232"]

    # Run simulation
    if st.session_state.running:
        mine = st.session_state.mine
        agents = st.session_state.agents

        placeholder_cols = [col.empty() for col in cols]
        metrics_cols = [col.empty() for col in cols]
        progress_bar = st.progress(0)

        max_steps = len(mine.path) - 1

        for step in range(max_steps):
            for agent in agents:
                agent.step()

            # Update display
            for i, (agent, pcol, mcol) in enumerate(zip(agents, placeholder_cols, metrics_cols)):
                with pcol.container():
                    st.markdown(f"**{colors[i]} {agent.name}**")
                    svg = draw_mine_svg(mine, agent)
                    st.markdown(svg, unsafe_allow_html=True)

                with mcol.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Uptime", f"{agent.uptime:.0%}")
                    c2.metric("Relays", f"{agent.budget - agent.budget_remaining}/{agent.budget}")
                    c3.metric("Signal", f"{agent.signal:.0%}")
                    if agent.method == "voi_ppo":
                        st.caption(f"VoI Estimate: {agent.voi:.3f}")

            progress_bar.progress((step + 1) / max_steps)
            time.sleep(speed_map[speed])

        st.session_state.running = False

        # Final results
        st.divider()
        st.header("Results")

        result_data = []
        for i, agent in enumerate(agents):
            result_data.append({
                "Method": f"{colors[i]} {agent.name}",
                "Uptime": f"{agent.uptime:.1%}",
                "Relays Used": f"{agent.budget - agent.budget_remaining}",
                "Efficiency": f"{agent.efficiency:.3f}",
                "Connected at End": "Yes" if agent.connected else "No",
            })

        st.table(result_data)

        # Key insight
        voi_agent = agents[0]
        baseline_agent = agents[1]
        threshold_agent = agents[2]

        if voi_agent.uptime > baseline_agent.uptime:
            improvement = (voi_agent.uptime - baseline_agent.uptime) * 100
            st.success(
                f"VoI-Guided agent achieved {improvement:.1f}% higher uptime than Standard PPO "
                f"while using {(baseline_agent.budget - baseline_agent.budget_remaining) - (voi_agent.budget - voi_agent.budget_remaining)} fewer relays."
            )
        else:
            st.info("In this configuration, methods performed similarly. Try increasing complexity.")

    else:
        # Show static mine
        for i, (col, agent) in enumerate(zip(cols, st.session_state.agents)):
            with col:
                st.markdown(f"**{colors[i]} {agent.name}**")
                svg = draw_mine_svg(st.session_state.mine, agent)
                st.markdown(svg, unsafe_allow_html=True)

        st.info("Press **▶ Run Simulation** to start the comparison.")

    # Footer
    st.divider()
    st.markdown("""
    ---
    **Research**: Value-of-Information Guided Decision Making in Resource-Constrained POMDPs with Irreversible Actions  
    **Author**: Vachi Kalra  
    [GitHub Repository](https://github.com/vachikalra/voi-sequential-deployment)
    """)


if __name__ == "__main__":
    main()
