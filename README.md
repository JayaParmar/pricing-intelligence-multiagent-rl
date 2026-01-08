# Pricing Intelligence with Multi-Agent Reinforcement Learning (SAC)

This repository presents a **Pricing Intelligence system** based on **Multi-Agent Reinforcement Learning**, designed to model and optimize competitive pricing strategies in dynamic e-commerce marketplaces.

The project explores how **learning agents** can interact in a simulated market environment, adapt to competitors, and converge toward effective pricing policies under realistic constraints.

---

## 🔍 Project Motivation

Modern pricing systems operate in highly competitive, data-driven environments where:
- Vendors react strategically to competitors
- Demand is dynamic and partially observable
- Classical rule-based pricing fails to generalize

This project investigates **reinforcement learning–based pricing agents** that learn directly from market interactions, rather than relying on fixed heuristics.

---

## 🧠 Core Ideas

- **Multi-Agent Reinforcement Learning (MARL)** for competitive pricing
- **Soft Actor-Critic (SAC)** for stable learning in continuous action spaces
- Explicit modeling of **market dynamics**, competitor behavior, and demand response
- Separation between:
  - Environment (market simulation)
  - Agents (learning & decision-making)
  - Data pipelines (training & evaluation)

---

## 🏗️ System Architecture

**High-level flow:**

1. Market environment simulates demand, competition, and rewards
2. Multiple pricing agents interact simultaneously
3. Each agent learns a pricing policy via **Soft Actor-Critic**
4. Policies are evaluated under different competitive scenarios
5. Results support strategy analysis and pricing optimization

---

## 📦 Repository Structure

agent/ # SAC agents and policy networks
environment/ # Market & pricing environment
preprocessing/ # Data preparation & feature pipelines
models/ # Neural network definitions
streamlit_app/ # Interactive visualization & demo
config.py # Experiment configuration
main_multiagent_train.py # Training entry point

---

---

## ⚙️ Technologies Used

- Python
- PyTorch
- Multi-Agent Reinforcement Learning
- Soft Actor-Critic (Actor–Critic methods)
- Simulation-based optimization
- Streamlit (for interactive demos)

---

## 🚀 How to Run (Minimal)

```bash
pip install -r requirements.txt
python main_multiagent_train.py
streamlit run streamlit_app/app.py
```
---
## 📈 Outcomes & Learnings

Demonstrated feasibility of learning-based pricing strategies

Observed emergent competitive behaviors between agents

Gained insights into stability, convergence, and reward shaping in MARL systems

Highlighted challenges in scaling RL for real-world economic systems

---
## 🔬 Research & Engineering Context

This project was developed as an exploratory research and engineering effort to study:

Decision-making under competition

Learning-based optimization of system-level objectives

Practical challenges in deploying RL for business-critical workflows

It reflects a systems-oriented approach to AI, emphasizing reproducibility, modularity, and extensibility.

---

## 📜 Disclaimer

This repository is intended for research and demonstration purposes.
It does not include proprietary data or production models.
