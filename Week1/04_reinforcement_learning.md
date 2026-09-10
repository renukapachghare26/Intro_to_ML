# Reinforcement Learning

## Definition

An agent learns to make decisions by interacting with an environment. It takes actions, receives rewards or penalties, and adjusts its behavior (policy) to maximize cumulative reward over time.

## Key components

- **Agent** — the learner/decision maker
- **Environment** — everything the agent interacts with
- **State** — current situation of the agent
- **Action** — what the agent can do
- **Reward** — feedback signal for an action
- **Policy** — the strategy the agent uses to pick actions

## RL Framework (Agent ↔ Environment loop)

- Two core blocks: **Environment** and **Agent**, sitting in a closed loop with each other
- Defining characteristics of the setup:
  - Agent **learns from close interaction** with the environment (not from a static dataset)
  - The **environment is stochastic** — the same action can lead to different outcomes
  - Feedback is a **noisy, delayed scalar evaluation** — the reward signal isn't clean or immediate, and a good/bad outcome may only become clear several steps later
  - The end goal is to **learn a policy** that **maximizes a measure of long-term performance** (not just the next immediate reward)
- **Why it matters**: this formalizes the "Key components" above into an actual loop — Agent and Environment are the two blocks, and State/Action/Reward are what flows between them. The "noisy, delayed" nature of the reward is exactly why the exploration-exploitation tradeoff (below) is hard: you often can't tell right away whether an action was good.

## Applications of RL

- **Game playing**
  - Backgammon — produced the world's best player through RL
  - Atari games learned from scratch (no hand-coded strategy)
- **Autonomous agents** — robot navigation
- **Adaptive control** — e.g., a helicopter pilot controller
- **Combinatorial optimization** — e.g., VLSI (chip) placement
- **Intelligent Tutoring Systems**
- **Why it matters**: these span very different domains (games, robotics, control, chip design, education) but all fit the same RL Framework above — each has an Agent, a stochastic Environment, and long-term performance to maximize (winning a game, safely reaching a destination, stable flight, an efficient chip layout, student learning outcomes) rather than one-shot correct answers like in supervised learning.

## How it's different from supervised learning

There's no fixed dataset of "correct" actions — the agent has to explore and discover what works through trial and error, balancing:
- **Exploration** — trying new actions to discover their effects
- **Exploitation** — using known good actions to maximize reward

## My own example

Training an agent to play a simple game like Snake — it doesn't know beforehand which moves are good, but learns over many episodes that moving toward food and avoiding walls leads to higher rewards.

## Questions to revisit

- How is the exploration-exploitation tradeoff actually tuned in practice?
- What makes reward design so tricky (reward hacking)?
- How do you deal with the "noisy, delayed" reward problem — is this what credit assignment refers to?