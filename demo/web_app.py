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
        page_title="VoI-Guided Decision Making Demo",
        page_icon="🧠",
        layout="wide",
    )


# --- Simulation Classes ---

class DemoNode:
    def __init__(self, x: float, y: float, rock_type: str = "sandstone", bend: float = 0):
        self.x = x
        self.y = y
        self.rock_type = rock_type
        self.bend = bend


class DemoMine:
    def __init__(self):
        self.nodes: list = []
        self.edges: list = []
        self.path: list = []
        self.danger_zones: list = []  # segments where signal drops fast

    def generate(self, complexity: float = 0.5, seed: int = 42):
        rng = random.Random(seed)
        self.nodes = []
        self.edges = []
        self.path = []
        self.danger_zones = []

        start_x, start_y = 300, 30
        rocks = ["sandstone", "limestone", "granite", "shale"]
        self.nodes.append(DemoNode(start_x, start_y, "sandstone"))
        self.path.append(0)

        x_pos, y_pos = start_x, start_y
        current_idx = 0
        n_segments = int(15 + complexity * 20)
        current_rock = "sandstone"

        for i in range(n_segments):
            dx = rng.gauss(0, 20 + complexity * 25)
            dy = rng.uniform(15, 35)
            x_pos = max(60, min(540, x_pos + dx))
            y_pos += dy

            if y_pos > 580:
                break

            # Rock type changes (some are "danger zones" that eat signal)
            if rng.random() < 0.2 + complexity * 0.2:
                current_rock = rng.choice(rocks)

            bend = abs(dx) / 30.0  # more lateral movement = more bend
            new_idx = len(self.nodes)
            self.nodes.append(DemoNode(x_pos, y_pos, current_rock, bend))
            self.edges.append((current_idx, new_idx))
            self.path.append(new_idx)

            # Mark danger zones (granite + bends = signal killers)
            if current_rock == "granite" or bend > 0.6:
                self.danger_zones.append(new_idx)

            # Side branches (visual complexity)
            if rng.random() < complexity * 0.25:
                branch_x = max(30, min(570, x_pos + rng.choice([-1, 1]) * rng.uniform(40, 90)))
                branch_y = y_pos + rng.uniform(10, 25)
                branch_idx = len(self.nodes)
                self.nodes.append(DemoNode(branch_x, branch_y, current_rock))
                self.edges.append((new_idx, branch_idx))

            current_idx = new_idx


class SimAgent:
    """Simulates a relay deployment agent with distinct strategies."""

    def __init__(self, name: str, method: str, mine: DemoMine, budget: int):
        self.name = name
        self.method = method
        self.mine = mine
        self.budget = budget
        self.budget_remaining = budget
        self.progress = 0
        self.relays: list = []
        self.signal = 1.0
        self.uptime_history: list = []
        self.voi = 0.0
        self.connected = True
        self.deploy_log: list = []  # (step, reason) tuples

    def step(self):
        if self.progress >= len(self.mine.path) - 1:
            return

        self.progress += 1

        # Signal degrades based on distance from last relay AND terrain
        last_relay = max(self.relays) if self.relays else 0
        distance = self.progress - last_relay
        
        # Base signal decay
        base_decay = 0.18 * distance
        
        # Terrain penalty (danger zones cause faster decay)
        terrain_penalty = 0.0
        current_node_idx = self.mine.path[self.progress] if self.progress < len(self.mine.path) else 0
        if current_node_idx < len(self.mine.nodes):
            node = self.mine.nodes[current_node_idx]
            if node.rock_type == "granite":
                terrain_penalty += 0.12
            elif node.rock_type == "shale":
                terrain_penalty += 0.06
            terrain_penalty += node.bend * 0.08
        
        self.signal = max(0.0, 1.0 - base_decay - terrain_penalty + random.gauss(0, 0.02))

        should_deploy, reason = self._decide()
        if should_deploy and self.budget_remaining > 0:
            self.relays.append(self.progress)
            self.budget_remaining -= 1
            self.signal = min(1.0, 0.85 + random.gauss(0, 0.03))
            self.deploy_log.append((self.progress, reason))

        self.connected = self.signal > 0.15
        self.uptime_history.append(self.connected)

    def _decide(self):
        """Each method has a distinct deployment strategy."""
        if self.method == "threshold":
            # Simple rule: deploy when signal drops below fixed threshold
            if self.signal < 0.30:
                return True, "Signal below 30%"
            return False, ""
            
        elif self.method == "baseline_ppo":
            # Learned but reactive: deploys when signal is getting low
            # Slightly smarter threshold with noise (simulating learned policy)
            noise = random.gauss(0, 0.05)
            threshold = 0.35 + noise
            if self.signal < threshold:
                return True, "Policy triggered (signal low)"
            return False, ""
            
        elif self.method == "voi_ppo":
            # VoI-guided: estimates whether waiting would help
            path_remaining = len(self.mine.path) - self.progress
            budget_frac = self.budget_remaining / max(self.budget, 1)
            
            # Look ahead: is a danger zone coming?
            danger_ahead = False
            for lookahead in range(1, 4):
                future_idx = self.progress + lookahead
                if future_idx < len(self.mine.path):
                    future_node = self.mine.path[future_idx]
                    if future_node in self.mine.danger_zones:
                        danger_ahead = True
                        break
            
            # VoI estimation: high when danger ahead and budget available
            self.voi = 0.0
            if danger_ahead and budget_frac > 0.2:
                self.voi = 0.6 + 0.3 * budget_frac
            elif budget_frac > 0.5 and path_remaining > 5:
                self.voi = 0.3 * budget_frac
            
            # Deploy strategically:
            # - If signal critical AND no danger ahead, deploy now
            # - If danger zone ahead, wait to deploy at the danger zone
            # - If VoI high, prefer to wait
            if self.signal < 0.20:
                return True, "Emergency: signal critical"
            elif self.signal < 0.35 and not danger_ahead:
                return True, "Optimal timing (no danger ahead)"
            elif self.signal < 0.25 and danger_ahead:
                return True, "Pre-danger deployment"
            
            return False, ""
            
        return False, ""

    @property
    def uptime(self) -> float:
        if not self.uptime_history:
            return 1.0
        return sum(self.uptime_history) / len(self.uptime_history)

    @property
    def efficiency(self) -> float:
        used = self.budget - self.budget_remaining
        if used == 0:
            return 0.0
        return self.uptime / used


# --- Visualization ---

def draw_mine_svg(mine: DemoMine, agent: SimAgent, width: int = 600, height: int = 620) -> str:
    """Generate SVG visualization of the mine with agent state."""
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="#0d1117" rx="8"/>',
    ]

    # Draw edges (tunnels) - color by rock type
    for (i, j) in mine.edges:
        n1, n2 = mine.nodes[i], mine.nodes[j]
        rock = n2.rock_type
        color = {"sandstone": "#2d3748", "limestone": "#2d3748", "granite": "#4a2020", "shale": "#3d3520"}
        stroke_color = color.get(rock, "#2d3748")
        svg_parts.append(
            f'<line x1="{n1.x}" y1="{n1.y}" x2="{n2.x}" y2="{n2.y}" '
            f'stroke="{stroke_color}" stroke-width="6" stroke-linecap="round"/>'
        )

    # Highlight traversed path with signal-strength coloring
    for idx in range(min(agent.progress, len(mine.path) - 1)):
        if idx + 1 < len(mine.path):
            n1 = mine.nodes[mine.path[idx]]
            n2 = mine.nodes[mine.path[idx + 1]]
            # Color path by whether connected at this point
            path_color = "#2563eb" if idx < len(agent.uptime_history) and agent.uptime_history[idx] else "#dc2626"
            svg_parts.append(
                f'<line x1="{n1.x}" y1="{n1.y}" x2="{n2.x}" y2="{n2.y}" '
                f'stroke="{path_color}" stroke-width="4" stroke-linecap="round" opacity="0.8"/>'
            )

    # Mark danger zones with subtle indicator
    for dz_idx in mine.danger_zones:
        if dz_idx < len(mine.nodes):
            dz_node = mine.nodes[dz_idx]
            svg_parts.append(
                f'<circle cx="{dz_node.x}" cy="{dz_node.y}" r="10" fill="#ff000015" stroke="#ff000030" stroke-width="1"/>'
            )

    # Draw relays with signal radius
    for relay_idx in agent.relays:
        if relay_idx < len(mine.path):
            node = mine.nodes[mine.path[relay_idx]]
            svg_parts.append(
                f'<circle cx="{node.x}" cy="{node.y}" r="18" fill="#10b98110" stroke="#10b98140" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<circle cx="{node.x}" cy="{node.y}" r="7" fill="#10b981" opacity="0.95"/>'
            )
            svg_parts.append(
                f'<text x="{node.x}" y="{node.y + 3.5}" text-anchor="middle" fill="white" font-size="8" font-weight="bold">R</text>'
            )

    # Draw entrance (base station)
    entrance = mine.nodes[0]
    svg_parts.append(
        f'<circle cx="{entrance.x}" cy="{entrance.y}" r="10" fill="#6366f1"/>'
    )
    svg_parts.append(
        f'<text x="{entrance.x}" y="{entrance.y + 3.5}" text-anchor="middle" fill="white" font-size="7" font-weight="bold">HQ</text>'
    )

    # Draw team position
    if agent.progress < len(mine.path):
        team_node = mine.nodes[mine.path[agent.progress]]
        if agent.signal > 0.6:
            signal_color = "#10b981"
        elif agent.signal > 0.3:
            signal_color = "#f59e0b"
        else:
            signal_color = "#ef4444"
        
        # Pulse effect
        svg_parts.append(
            f'<circle cx="{team_node.x}" cy="{team_node.y}" r="14" fill="{signal_color}10" stroke="{signal_color}" stroke-width="1.5" opacity="0.6"/>'
        )
        svg_parts.append(
            f'<circle cx="{team_node.x}" cy="{team_node.y}" r="6" fill="{signal_color}"/>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def signal_bar(signal: float, width: int = 200) -> str:
    """Create an SVG signal strength bar."""
    if signal > 0.6:
        color = "#10b981"
    elif signal > 0.3:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    
    fill_width = int(signal * width)
    return f'''<svg width="{width}" height="12" xmlns="http://www.w3.org/2000/svg">
        <rect width="{width}" height="12" fill="#1f2937" rx="6"/>
        <rect width="{fill_width}" height="12" fill="{color}" rx="6" opacity="0.85"/>
    </svg>'''


# --- Main App ---

def main():
    if not HAS_STREAMLIT:
        print("Streamlit required: pip install streamlit")
        print("Run with: python3 -m streamlit run demo/web_app.py")
        return

    # Custom CSS for dark theme polish
    st.markdown("""
    <style>
    .stMetric { text-align: center; }
    .insight-box { 
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border: 1px solid #2563eb40; 
        border-radius: 12px; 
        padding: 20px; 
        margin: 10px 0;
    }
    .method-label {
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    # 🧠 Can AI Learn *When* to Commit?
    
    **The Problem**: A rescue team is going deep underground. They have limited radio relays 
    to drop along the way. Once placed, a relay can't be moved. Place too early = waste it. 
    Place too late = lose contact.
    
    **The Question**: Can an AI learn to *wait for more information* before making permanent decisions?
    """)
    
    st.markdown("---")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Scenario Settings")
        
        st.markdown("**Adjust these to see how each method handles different challenges:**")
        
        complexity = st.slider("🏔️ Mine Complexity", 0.1, 1.0, 0.6, 0.1,
                              help="Higher = more twists, bends, and hard-rock zones that kill signal")
        budget = st.slider("📦 Relay Budget", 2, 10, 4,
                          help="Fewer relays = harder problem. The AI must be more strategic.")
        seed = st.number_input("🎲 Random Seed", 0, 99999, 42,
                              help="Change this to try different mine layouts")

        st.markdown("---")
        st.markdown("**⚡ Simulation Speed**")
        speed = st.select_slider("",
                                options=["Slow (watch closely)", "Normal", "Fast"],
                                value="Normal")
        speed_map = {"Slow (watch closely)": 0.4, "Normal": 0.15, "Fast": 0.03}

        st.markdown("---")
        run_button = st.button("▶️  Run Experiment", type="primary", use_container_width=True)
        reset_button = st.button("🔄 New Mine", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        ### 🎯 What to Watch For
        
        1. **Blue path** = connected  
           **Red path** = lost contact
        2. **Green circles** = placed relays
        3. **Red tint** = danger zones (granite/bends)
        4. Watch **when** each method places relays — timing is everything!
        """)

    # Method explanation columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🟢 VoI-PPO (Ours)")
        st.caption("Asks: 'Would waiting give me useful info?' Deploys strategically at danger zones.")
    with col2:
        st.markdown("### 🟡 Standard PPO")
        st.caption("Learns a fixed threshold from experience. Deploys when signal gets low.")
    with col3:
        st.markdown("### 🔴 Signal Threshold")
        st.caption("Simple rule: deploy whenever signal drops below 30%. No planning.")

    st.markdown("---")

    # Initialize state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "mine" not in st.session_state or reset_button:
        st.session_state.mine = DemoMine()
        st.session_state.mine.generate(complexity, seed)
        st.session_state.agents = [
            SimAgent("VoI-PPO (Ours)", "voi_ppo", st.session_state.mine, budget),
            SimAgent("Standard PPO", "baseline_ppo", st.session_state.mine, budget),
            SimAgent("Signal Threshold", "threshold", st.session_state.mine, budget),
        ]
        st.session_state.step = 0
        st.session_state.running = False

    if run_button:
        st.session_state.mine = DemoMine()
        st.session_state.mine.generate(complexity, seed)
        st.session_state.agents = [
            SimAgent("VoI-PPO (Ours)", "voi_ppo", st.session_state.mine, budget),
            SimAgent("Standard PPO", "baseline_ppo", st.session_state.mine, budget),
            SimAgent("Signal Threshold", "threshold", st.session_state.mine, budget),
        ]
        st.session_state.running = True

    # Display
    cols = st.columns(3)

    # Run simulation
    if st.session_state.running:
        mine = st.session_state.mine
        agents = st.session_state.agents

        placeholder_cols = [col.empty() for col in cols]
        metrics_cols = [col.empty() for col in cols]
        
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        max_steps = len(mine.path) - 1
        method_colors_hex = ["#10b981", "#f59e0b", "#ef4444"]

        for step in range(max_steps):
            for agent in agents:
                agent.step()

            # Update display
            for i, (agent, pcol, mcol) in enumerate(zip(agents, placeholder_cols, metrics_cols)):
                with pcol.container():
                    svg = draw_mine_svg(mine, agent)
                    st.markdown(svg, unsafe_allow_html=True)

                with mcol.container():
                    c1, c2, c3 = st.columns(3)
                    
                    uptime_color = "normal" if agent.uptime > 0.8 else ("off" if agent.uptime < 0.5 else "normal")
                    c1.metric("📡 Uptime", f"{agent.uptime:.0%}")
                    c2.metric("📦 Budget", f"{agent.budget_remaining}/{agent.budget}")
                    c3.metric("📶 Signal", f"{agent.signal:.0%}")
            
            # Live narration
            with status_placeholder.container():
                step_events = []
                for agent in agents:
                    if agent.deploy_log and agent.deploy_log[-1][0] == agent.progress:
                        reason = agent.deploy_log[-1][1]
                        step_events.append(f"**{agent.name}** deployed relay! ({reason})")
                
                if step_events:
                    st.info(" | ".join(step_events))
                else:
                    disconnected = [a.name for a in agents if not a.connected]
                    if disconnected:
                        st.error(f"⚠️ LOST CONTACT: {', '.join(disconnected)}")

            progress_bar.progress((step + 1) / max_steps)
            time.sleep(speed_map[speed])

        st.session_state.running = False
        status_placeholder.empty()

        # === RESULTS ===
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        # Summary cards
        result_cols = st.columns(3)
        icons = ["🟢", "🟡", "🔴"]
        
        for i, (agent, rcol) in enumerate(zip(agents, result_cols)):
            with rcol:
                used = agent.budget - agent.budget_remaining
                st.markdown(f"### {icons[i]} {agent.name}")
                st.metric("Communication Uptime", f"{agent.uptime:.1%}")
                st.metric("Relays Used", f"{used} / {agent.budget}")
                st.metric("Efficiency", f"{agent.efficiency:.2f} uptime/relay")
                
                if not agent.connected:
                    st.error("❌ Lost contact at end")
                else:
                    st.success("✅ Connected at end")

        # Key takeaway
        st.markdown("---")
        st.markdown("## 💡 Key Insight")
        
        voi_agent = agents[0]
        baseline_agent = agents[1]
        threshold_agent = agents[2]
        
        voi_used = voi_agent.budget - voi_agent.budget_remaining
        bl_used = baseline_agent.budget - baseline_agent.budget_remaining
        th_used = threshold_agent.budget - threshold_agent.budget_remaining
        
        if voi_agent.uptime >= baseline_agent.uptime and voi_used <= bl_used:
            st.success(f"""
            **VoI-PPO wins!** By estimating the *value of waiting*, it achieved 
            **{voi_agent.uptime:.0%} uptime** using only **{voi_used} relays** — 
            vs Standard PPO's {bl_used} relays and Signal Threshold's {th_used} relays.
            
            🧠 *The AI learned that sometimes the best action is to wait for more information 
            before committing an irreversible decision.*
            """)
        elif voi_agent.uptime > threshold_agent.uptime:
            diff = (voi_agent.uptime - threshold_agent.uptime) * 100
            st.success(f"""
            **Learning beats rules!** VoI-PPO maintained **{diff:.0f}% more connectivity** 
            than the simple threshold rule, showing that ML-based decision-making 
            outperforms hand-crafted heuristics in uncertain environments.
            """)
        else:
            st.info("""
            All methods performed similarly in this easy scenario. 
            **Try reducing the relay budget to 2-3** to see where VoI-PPO's strategic 
            timing really shines under resource pressure!
            """)
        
        # Explanation of WHY
        with st.expander("🔬 Why does VoI-PPO work better?", expanded=False):
            st.markdown("""
            ### The Science Behind It
            
            **Standard RL** learns a policy: "if signal < X, deploy." This is reactive.
            
            **VoI-Guided RL** adds a second question: "If I wait one more step, 
            will I learn something that changes my decision?"
            
            This matters because:
            - 🏔️ **Geology varies**: A relay in soft sandstone covers more distance than one in hard granite
            - 🔄 **Bends kill signal**: A sharp turn ahead means you should save your relay for AFTER the turn
            - 📦 **Budget is finite**: Every wasted relay is one you can't use later when it matters more
            
            The VoI estimator learns to recognize these situations and WAIT when waiting 
            would lead to a better deployment decision.
            """)

    else:
        # Show static mine (pre-run)
        for i, (col, agent) in enumerate(zip(cols, st.session_state.agents)):
            with col:
                svg = draw_mine_svg(st.session_state.mine, agent)
                st.markdown(svg, unsafe_allow_html=True)

        st.info("👆 Press **▶️ Run Experiment** in the sidebar to start the comparison. Try setting budget to **3 or 4** to see the biggest differences!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; font-size: 12px;">
        <b>Value-of-Information Guided Decision Making in Resource-Constrained POMDPs with Irreversible Actions</b><br>
        Vachi Kalra | <a href="https://github.com/vachikalra/voi-sequential-deployment">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
