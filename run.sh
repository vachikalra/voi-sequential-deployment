#!/bin/bash
# Run script for VoI Sequential Deployment Research Project
# Usage: ./run.sh [command]
#
# Commands:
#   train       - Train the VoI-PPO agent
#   train-all   - Train all methods for comparison
#   evaluate    - Evaluate trained agents
#   demo        - Launch interactive web demo
#   demo-local  - Launch Pygame local demo
#   experiment  - Run full experiment suite

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 required"
    exit 1
fi

case "${1:-help}" in
    install)
        echo "Installing dependencies..."
        pip install -r requirements.txt
        echo "Done."
        ;;

    train)
        echo "Training VoI-PPO agent..."
        python3 train_agent.py --method voi_ppo --steps ${2:-500000} --device ${3:-cpu}
        ;;

    train-baseline)
        echo "Training baseline PPO (no VoI)..."
        python3 train_agent.py --method baseline_ppo --steps ${2:-500000} --device ${3:-cpu}
        ;;

    train-all)
        echo "Training all learned methods..."
        echo ""
        echo "=== VoI-PPO ==="
        python3 train_agent.py --method voi_ppo --steps ${2:-500000} --device ${3:-cpu}
        echo ""
        echo "=== Baseline PPO ==="
        python3 train_agent.py --method baseline_ppo --steps ${2:-500000} --device ${3:-cpu}
        echo ""
        echo "All training complete."
        ;;

    evaluate)
        echo "Evaluating all methods..."
        python3 train_agent.py --method voi_ppo --eval-only --model-path checkpoints/voi_ppo/best_model.pt
        python3 train_agent.py --method baseline_ppo --eval-only --model-path checkpoints/baseline_ppo/best_model.pt
        python3 train_agent.py --method threshold --eval-only
        python3 train_agent.py --method greedy --eval-only
        python3 train_agent.py --method fixed --eval-only
        ;;

    demo)
        echo "Launching web demo..."
        echo "Open http://localhost:8501 in your browser"
        streamlit run demo/web_app.py
        ;;

    demo-local)
        echo "Launching local Pygame demo..."
        python3 demo/app.py
        ;;

    experiment)
        echo "Running full experiment suite..."
        python3 experiments/run_all.py --experiment ${2:-exp1_complexity}
        ;;

    help|*)
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  VoI Sequential Deployment - Research Project           ║"
        echo "╠══════════════════════════════════════════════════════════╣"
        echo "║                                                          ║"
        echo "║  Commands:                                               ║"
        echo "║    ./run.sh install      - Install dependencies          ║"
        echo "║    ./run.sh train        - Train VoI-PPO agent           ║"
        echo "║    ./run.sh train-baseline - Train standard PPO          ║"
        echo "║    ./run.sh train-all    - Train all methods             ║"
        echo "║    ./run.sh evaluate     - Evaluate all methods          ║"
        echo "║    ./run.sh demo         - Launch web demo               ║"
        echo "║    ./run.sh demo-local   - Launch Pygame demo            ║"
        echo "║    ./run.sh experiment   - Run experiments               ║"
        echo "║                                                          ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        ;;
esac
