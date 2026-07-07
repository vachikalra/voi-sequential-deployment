"""
Interactive Demo Application.

A visual, game-style interface where users can:
1. Generate or draw underground tunnel networks
2. Watch different algorithms deploy relays in real-time
3. Compare VoI-guided agent vs. standard PPO vs. heuristics side-by-side
4. See VoI estimates visualized (when is the agent "waiting for info"?)
5. Track scores: communication uptime, relays used, efficiency

Built with Pygame for local presentation or can be adapted to web.
"""

import sys
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Pygame not installed. Run: pip install pygame")


# --- Configuration ---

@dataclass
class DemoConfig:
    """Visual demo configuration."""
    screen_width: int = 1400
    screen_height: int = 800
    fps: int = 30
    simulation_speed: float = 1.0

    # Colors
    bg_color: tuple = (18, 18, 24)
    tunnel_color: tuple = (60, 60, 80)
    tunnel_highlight: tuple = (80, 100, 140)
    relay_color: tuple = (0, 220, 130)
    team_color: tuple = (255, 200, 50)
    signal_good: tuple = (0, 220, 130)
    signal_warn: tuple = (255, 180, 0)
    signal_bad: tuple = (255, 50, 50)
    voi_color: tuple = (130, 80, 255)
    text_color: tuple = (220, 220, 230)
    panel_color: tuple = (28, 28, 38)
    accent_color: tuple = (100, 140, 255)


class GameState(Enum):
    MENU = "menu"
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


class AlgorithmType(Enum):
    VOI_PPO = "VoI-Guided RL"
    BASELINE_PPO = "Standard PPO"
    THRESHOLD = "Signal Threshold"
    GREEDY = "Greedy Lookahead"


# --- Mine Generation for Demo ---

@dataclass
class DemoNode:
    x: float
    y: float
    is_relay: bool = False
    is_entrance: bool = False
    is_team: bool = False
    sinr: float = 50.0


@dataclass
class DemoEdge:
    start_idx: int
    end_idx: int
    length: float
    signal_quality: float = 1.0  # 0-1


class DemoMine:
    """Simplified mine representation for visualization."""

    def __init__(self):
        self.nodes: list[DemoNode] = []
        self.edges: list[DemoEdge] = []
        self.path: list[int] = []  # ordered node indices for team traversal

    def generate_random(self, complexity: float = 0.5, seed: Optional[int] = None):
        """Generate a random mine topology for demo."""
        rng = random.Random(seed)
        self.nodes = []
        self.edges = []
        self.path = []

        # Start at top-center
        start_x, start_y = 400, 60
        self.nodes.append(DemoNode(x=start_x, y=start_y, is_entrance=True))

        # Generate path downward with branches
        current_idx = 0
        y_pos = start_y
        x_pos = start_x

        n_segments = int(15 + complexity * 20)

        for i in range(n_segments):
            # Direction: mostly down, some lateral
            dx = rng.gauss(0, 30 + complexity * 20)
            dy = rng.uniform(25, 50)
            x_pos = max(100, min(700, x_pos + dx))
            y_pos += dy

            if y_pos > 700:
                break

            new_idx = len(self.nodes)
            self.nodes.append(DemoNode(x=x_pos, y=y_pos))

            edge_length = math.sqrt(dx**2 + dy**2)
            self.edges.append(DemoEdge(
                start_idx=current_idx,
                end_idx=new_idx,
                length=edge_length,
            ))
            self.path.append(new_idx)

            # Branch with probability based on complexity
            if rng.random() < complexity * 0.3:
                branch_dx = rng.choice([-1, 1]) * rng.uniform(40, 100)
                branch_dy = rng.uniform(10, 40)
                branch_x = max(50, min(750, x_pos + branch_dx))
                branch_y = y_pos + branch_dy

                branch_idx = len(self.nodes)
                self.nodes.append(DemoNode(x=branch_x, y=branch_y))
                self.edges.append(DemoEdge(
                    start_idx=new_idx,
                    end_idx=branch_idx,
                    length=math.sqrt(branch_dx**2 + branch_dy**2),
                ))

            current_idx = new_idx

    def get_bounds(self) -> tuple:
        """Get bounding box of mine."""
        if not self.nodes:
            return (0, 0, 800, 800)
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return (min(xs) - 20, min(ys) - 20, max(xs) + 20, max(ys) + 20)


# --- Simulation State ---

class SimulationAgent:
    """Represents one algorithm running the simulation."""

    def __init__(self, algorithm: AlgorithmType, mine: DemoMine, budget: int):
        self.algorithm = algorithm
        self.mine = mine
        self.budget = budget
        self.budget_remaining = budget
        self.team_progress = 0  # index into mine.path
        self.relays_placed: list[int] = []  # node indices where relays are
        self.signal_strength = 1.0  # 0-1 normalized
        self.uptime_history: list[bool] = []
        self.voi_estimate = 0.0  # only for VoI agent
        self.is_connected = True
        self.total_uptime = 0.0
        self.steps = 0

    def step(self) -> None:
        """Advance simulation by one step."""
        if self.team_progress >= len(self.mine.path) - 1:
            return

        self.steps += 1
        self.team_progress += 1

        # Compute signal degradation
        self._update_signal()

        # Decide whether to deploy
        should_deploy = self._decide()
        if should_deploy and self.budget_remaining > 0:
            self.relays_placed.append(self.team_progress)
            self.budget_remaining -= 1
            self.signal_strength = min(1.0, self.signal_strength + 0.4)

        # Update connectivity
        self.is_connected = self.signal_strength > 0.2
        self.uptime_history.append(self.is_connected)
        if self.uptime_history:
            self.total_uptime = sum(self.uptime_history) / len(self.uptime_history)

    def _update_signal(self) -> None:
        """Simulate signal degradation with distance from last relay."""
        last_relay = max(self.relays_placed) if self.relays_placed else 0
        distance = self.team_progress - last_relay
        # Signal decays with distance (simplified model)
        decay_rate = 0.08 + random.gauss(0, 0.01)
        self.signal_strength = max(0.0, 1.0 - decay_rate * distance)

    def _decide(self) -> bool:
        """Algorithm-specific deployment decision."""
        if self.algorithm == AlgorithmType.THRESHOLD:
            return self.signal_strength < 0.35

        elif self.algorithm == AlgorithmType.GREEDY:
            # Deploy when signal is declining and approaching threshold
            if self.signal_strength < 0.45 and self.budget_remaining > 2:
                return True
            if self.signal_strength < 0.3:
                return True
            return False

        elif self.algorithm == AlgorithmType.BASELINE_PPO:
            # Simulate PPO behavior: somewhat random early, learns threshold-ish
            noise = random.gauss(0, 0.1)
            return (self.signal_strength + noise) < 0.4

        elif self.algorithm == AlgorithmType.VOI_PPO:
            # Simulate VoI behavior: waits longer, deploys more strategically
            # VoI estimate: high when uncertain about what's ahead
            path_remaining = len(self.mine.path) - self.team_progress
            budget_fraction = self.budget_remaining / max(self.budget, 1)

            # VoI is high when we have budget and path remaining
            self.voi_estimate = 0.3 * budget_fraction * (path_remaining / max(len(self.mine.path), 1))

            # Only commit when signal is truly low AND VoI is low
            commit_threshold = 0.3 - self.voi_estimate * 0.15
            return self.signal_strength < commit_threshold

        return False

    @property
    def efficiency(self) -> float:
        relays_used = self.budget - self.budget_remaining
        if relays_used == 0:
            return 0.0
        return self.total_uptime / relays_used


# --- Main Demo App ---

class DemoApp:
    """Interactive demo application."""

    def __init__(self, config: DemoConfig = None):
        if not HAS_PYGAME:
            raise RuntimeError("Pygame required. Install with: pip install pygame")

        self.config = config or DemoConfig()
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(
            (self.config.screen_width, self.config.screen_height)
        )
        pygame.display.set_caption(
            "VoI-Guided Decision Making — Interactive Research Demo"
        )

        self.clock = pygame.time.Clock()
        self.state = GameState.MENU

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)

        # Simulation
        self.mine = DemoMine()
        self.agents: list[SimulationAgent] = []
        self.simulation_tick = 0
        self.simulation_speed = 8  # ticks between steps
        self.tick_counter = 0
        self.complexity = 0.5
        self.budget = 8

    def run(self) -> None:
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self._handle_event(event)

            self._update()
            self._draw()

            pygame.display.flip()
            self.clock.tick(self.config.fps)

        pygame.quit()

    def _handle_event(self, event) -> None:
        """Handle user input events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.state == GameState.RUNNING:
                    self.state = GameState.PAUSED
                elif self.state == GameState.PAUSED:
                    self.state = GameState.RUNNING
                elif self.state in (GameState.MENU, GameState.FINISHED):
                    pygame.event.post(pygame.event.Event(pygame.QUIT))

            elif event.key == pygame.K_SPACE:
                if self.state == GameState.MENU:
                    self._start_simulation()
                elif self.state == GameState.FINISHED:
                    self.state = GameState.MENU
                elif self.state == GameState.PAUSED:
                    self.state = GameState.RUNNING

            elif event.key == pygame.K_r:
                self._start_simulation()

            elif event.key == pygame.K_UP:
                self.complexity = min(1.0, self.complexity + 0.1)
            elif event.key == pygame.K_DOWN:
                self.complexity = max(0.1, self.complexity - 0.1)
            elif event.key == pygame.K_RIGHT:
                self.budget = min(15, self.budget + 1)
            elif event.key == pygame.K_LEFT:
                self.budget = max(3, self.budget - 1)

            elif event.key == pygame.K_1:
                self.simulation_speed = 15  # slow
            elif event.key == pygame.K_2:
                self.simulation_speed = 8   # normal
            elif event.key == pygame.K_3:
                self.simulation_speed = 3   # fast

    def _start_simulation(self) -> None:
        """Initialize and start a new simulation run."""
        seed = random.randint(0, 99999)
        self.mine = DemoMine()
        self.mine.generate_random(complexity=self.complexity, seed=seed)

        self.agents = [
            SimulationAgent(AlgorithmType.VOI_PPO, self.mine, self.budget),
            SimulationAgent(AlgorithmType.BASELINE_PPO, self.mine, self.budget),
            SimulationAgent(AlgorithmType.THRESHOLD, self.mine, self.budget),
        ]

        self.simulation_tick = 0
        self.tick_counter = 0
        self.state = GameState.RUNNING

    def _update(self) -> None:
        """Update simulation state."""
        if self.state != GameState.RUNNING:
            return

        self.tick_counter += 1
        if self.tick_counter >= self.simulation_speed:
            self.tick_counter = 0
            self.simulation_tick += 1

            all_done = True
            for agent in self.agents:
                agent.step()
                if agent.team_progress < len(self.mine.path) - 1:
                    all_done = False

            if all_done:
                self.state = GameState.FINISHED

    def _draw(self) -> None:
        """Render current frame."""
        self.screen.fill(self.config.bg_color)

        if self.state == GameState.MENU:
            self._draw_menu()
        elif self.state in (GameState.RUNNING, GameState.PAUSED):
            self._draw_simulation()
        elif self.state == GameState.FINISHED:
            self._draw_simulation()
            self._draw_results_overlay()

    def _draw_menu(self) -> None:
        """Draw start menu."""
        # Title
        title = self.font_large.render(
            "VoI-Guided Decision Making", True, self.config.accent_color
        )
        subtitle = self.font_medium.render(
            "Interactive Research Demo", True, self.config.text_color
        )
        self.screen.blit(title, (self.config.screen_width // 2 - title.get_width() // 2, 100))
        self.screen.blit(subtitle, (self.config.screen_width // 2 - subtitle.get_width() // 2, 160))

        # Research question
        rq = self.font_small.render(
            "Can estimating the Value of Information improve irreversible decision timing?",
            True, (180, 180, 200)
        )
        self.screen.blit(rq, (self.config.screen_width // 2 - rq.get_width() // 2, 220))

        # Controls
        y_start = 320
        controls = [
            f"Complexity: {'█' * int(self.complexity * 10)}{'░' * (10 - int(self.complexity * 10))} ({self.complexity:.1f})  [↑/↓]",
            f"Relay Budget: {self.budget}  [←/→]",
            "",
            "SPACE — Start Simulation",
            "R — Regenerate Mine",
            "1/2/3 — Speed (Slow/Normal/Fast)",
            "ESC — Quit",
        ]
        for i, text in enumerate(controls):
            color = self.config.text_color if text else self.config.bg_color
            if "SPACE" in text:
                color = self.config.accent_color
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (self.config.screen_width // 2 - 200, y_start + i * 35))

        # Algorithm labels
        y_algo = 580
        algo_title = self.font_medium.render("Algorithms Compared:", True, self.config.text_color)
        self.screen.blit(algo_title, (self.config.screen_width // 2 - 150, y_algo))

        algos = [
            ("● VoI-Guided PPO (Proposed)", self.config.relay_color),
            ("● Standard PPO (Baseline)", self.config.signal_warn),
            ("● Signal Threshold (Heuristic)", self.config.signal_bad),
        ]
        for i, (name, color) in enumerate(algos):
            rendered = self.font_small.render(name, True, color)
            self.screen.blit(rendered, (self.config.screen_width // 2 - 150, y_algo + 40 + i * 30))

    def _draw_simulation(self) -> None:
        """Draw the running simulation with side-by-side agents."""
        n_agents = len(self.agents)
        panel_width = self.config.screen_width // n_agents

        for i, agent in enumerate(self.agents):
            x_offset = i * panel_width
            self._draw_agent_panel(agent, x_offset, panel_width)

            # Draw separator lines
            if i > 0:
                pygame.draw.line(
                    self.screen,
                    (50, 50, 70),
                    (x_offset, 0),
                    (x_offset, self.config.screen_height),
                    2,
                )

    def _draw_agent_panel(self, agent: SimulationAgent, x_offset: int, width: int) -> None:
        """Draw a single agent's simulation panel."""
        # Header
        color = {
            AlgorithmType.VOI_PPO: self.config.relay_color,
            AlgorithmType.BASELINE_PPO: self.config.signal_warn,
            AlgorithmType.THRESHOLD: self.config.signal_bad,
        }.get(agent.algorithm, self.config.text_color)

        title = self.font_small.render(agent.algorithm.value, True, color)
        self.screen.blit(title, (x_offset + width // 2 - title.get_width() // 2, 10))

        # Stats
        stats = [
            f"Uptime: {agent.total_uptime * 100:.0f}%",
            f"Relays: {agent.budget - agent.budget_remaining}/{agent.budget}",
            f"Signal: {agent.signal_strength:.0%}",
        ]
        if agent.algorithm == AlgorithmType.VOI_PPO:
            stats.append(f"VoI: {agent.voi_estimate:.2f}")

        for j, stat in enumerate(stats):
            rendered = self.font_tiny.render(stat, True, self.config.text_color)
            self.screen.blit(rendered, (x_offset + 10, 35 + j * 18))

        # Draw mine
        mine_x_offset = x_offset + 20
        mine_y_offset = 110
        scale = min((width - 40) / 700, (self.config.screen_height - 150) / 750)

        # Draw edges (tunnels)
        for edge in agent.mine.edges:
            n1 = agent.mine.nodes[edge.start_idx]
            n2 = agent.mine.nodes[edge.end_idx]
            p1 = (int(mine_x_offset + n1.x * scale), int(mine_y_offset + n1.y * scale))
            p2 = (int(mine_x_offset + n2.x * scale), int(mine_y_offset + n2.y * scale))

            # Color based on whether this segment is "covered" by relay
            tunnel_color = self.config.tunnel_color
            pygame.draw.line(self.screen, tunnel_color, p1, p2, 3)

        # Draw relays
        for relay_idx in agent.relays_placed:
            if relay_idx < len(agent.mine.path):
                node_idx = agent.mine.path[relay_idx]
                if node_idx < len(agent.mine.nodes):
                    node = agent.mine.nodes[node_idx]
                    pos = (int(mine_x_offset + node.x * scale), int(mine_y_offset + node.y * scale))
                    pygame.draw.circle(self.screen, self.config.relay_color, pos, 6)
                    pygame.draw.circle(self.screen, self.config.relay_color, pos, 12, 1)

        # Draw entrance
        entrance = agent.mine.nodes[0]
        entrance_pos = (int(mine_x_offset + entrance.x * scale), int(mine_y_offset + entrance.y * scale))
        pygame.draw.circle(self.screen, self.config.accent_color, entrance_pos, 8)

        # Draw team position
        if agent.team_progress < len(agent.mine.path):
            team_node_idx = agent.mine.path[agent.team_progress]
            if team_node_idx < len(agent.mine.nodes):
                team_node = agent.mine.nodes[team_node_idx]
                team_pos = (
                    int(mine_x_offset + team_node.x * scale),
                    int(mine_y_offset + team_node.y * scale)
                )
                pygame.draw.circle(self.screen, self.config.team_color, team_pos, 7)

                # Signal quality indicator (ring around team)
                signal_color = self._signal_color(agent.signal_strength)
                pygame.draw.circle(self.screen, signal_color, team_pos, 14, 2)

        # VoI indicator bar (only for VoI agent)
        if agent.algorithm == AlgorithmType.VOI_PPO:
            bar_x = x_offset + width - 30
            bar_y = 110
            bar_height = 200
            bar_width = 12

            # Background
            pygame.draw.rect(self.screen, (40, 40, 50), (bar_x, bar_y, bar_width, bar_height))
            # Fill based on VoI
            fill_height = int(agent.voi_estimate * bar_height * 3)
            fill_height = min(fill_height, bar_height)
            pygame.draw.rect(
                self.screen, self.config.voi_color,
                (bar_x, bar_y + bar_height - fill_height, bar_width, fill_height)
            )
            # Label
            label = self.font_tiny.render("VoI", True, self.config.voi_color)
            self.screen.blit(label, (bar_x - 2, bar_y + bar_height + 5))

    def _draw_results_overlay(self) -> None:
        """Draw final results comparison overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.config.screen_width, 200))
        overlay.set_alpha(230)
        overlay.fill(self.config.panel_color)
        self.screen.blit(overlay, (0, self.config.screen_height - 200))

        y_base = self.config.screen_height - 180
        title = self.font_medium.render("RESULTS", True, self.config.accent_color)
        self.screen.blit(title, (self.config.screen_width // 2 - title.get_width() // 2, y_base))

        # Results table
        y_base += 40
        headers = ["Method", "Uptime", "Relays Used", "Efficiency"]
        col_widths = [250, 150, 150, 150]
        x_start = (self.config.screen_width - sum(col_widths)) // 2

        for i, header in enumerate(headers):
            h = self.font_small.render(header, True, (150, 150, 170))
            self.screen.blit(h, (x_start + sum(col_widths[:i]), y_base))

        for j, agent in enumerate(self.agents):
            y = y_base + 30 + j * 28
            color = {
                AlgorithmType.VOI_PPO: self.config.relay_color,
                AlgorithmType.BASELINE_PPO: self.config.signal_warn,
                AlgorithmType.THRESHOLD: self.config.signal_bad,
            }.get(agent.algorithm, self.config.text_color)

            values = [
                agent.algorithm.value,
                f"{agent.total_uptime * 100:.1f}%",
                f"{agent.budget - agent.budget_remaining}",
                f"{agent.efficiency:.2f}",
            ]
            for i, val in enumerate(values):
                v = self.font_small.render(val, True, color)
                self.screen.blit(v, (x_start + sum(col_widths[:i]), y))

        # Restart hint
        hint = self.font_tiny.render("SPACE = Menu  |  R = New Run", True, (120, 120, 140))
        self.screen.blit(hint, (self.config.screen_width // 2 - hint.get_width() // 2, y_base + 130))

    def _signal_color(self, strength: float) -> tuple:
        """Map signal strength to color."""
        if strength > 0.6:
            return self.config.signal_good
        elif strength > 0.3:
            return self.config.signal_warn
        else:
            return self.config.signal_bad


def main():
    """Entry point for the interactive demo."""
    if not HAS_PYGAME:
        print("Error: Pygame is required for the interactive demo.")
        print("Install it with: pip install pygame")
        sys.exit(1)

    app = DemoApp()
    app.run()


if __name__ == "__main__":
    main()
