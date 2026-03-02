# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BehAV is an autonomous robot navigation system that uses Vision Language Models (VLMs), LLMs, and Model Predictive Control (MPC) to enable quadruped robots to follow natural language navigation instructions in outdoor environments.

Paper: https://arxiv.org/pdf/2409.16484

## Architecture

The system has four modules that form a connected pipeline:

1. **Instruction Decomposition** (`instruction-decomposition/GPT4o/instruction-to-behavioral-costs.py`)
   - Parses natural language into landmarks, navigation actions, behavioral actions, and targets using GPT-4
   - Key function: `get_instruction_breakdown(client, instruction)` → dict with keys `behavioral_actions`, `behavioral_targets`, etc.
   - Cannot be imported normally (hyphenated filename) — load with `importlib.util.spec_from_file_location`

2. **Director / Orchestration** (`planning/behav-director.py`) ← *added*
   - Bridges instruction decomposition and the planner
   - Accepts `--instruction`, `--demo`, `--interactive` modes; falls back to demo config if no API key
   - Maps `(behavioral_action, behavioral_target)` pairs to CLIPSeg cost values via `_ACTION_COST_MAP`
   - Publishes JSON to `/behav_config` (std_msgs/String) after 1 s delay, then every `--republish-interval` s

3. **Landmark Detection** (`landmark-tracking/`) - Locates landmarks using GPT-4 Vision + SAM

4. **Motion Planner** (`planning/behav-planner.py`) - ROS2 node with MPC optimization + CLIPSeg cost maps
   - Subscribes to `/behav_config` → `_behav_config_callback()` updates `self.prompts` / `self.cost_values` live
   - Resetting `received_final_goal_odom = False` in the callback triggers `goal_to_odom_pose()` recalculation

Data flow:
```
User instruction
  → behav-director.py (GPT-4 decomposition → CLIPSeg prompts + costs + goal)
  → /behav_config (JSON, std_msgs/String)
  → behav-planner.py (CLIPSeg costmap + MPC trajectory optimization)
  → /mcu/command/manual_twist (motor commands)
```

## Running with Docker (recommended)

### Full stack (sim + planner + director)
```bash
# Copy and fill in your API key
cp .env.example .env   # or create .env manually
echo "OPENAI_API_KEY=sk-..." >> .env

# Start everything
docker-compose up go2_sim behav director
```

### Override instruction / goal via environment variables
```bash
BEHAV_INSTRUCTION="Walk on the sidewalk, avoid puddles" \
BEHAV_GOAL_RADIUS=8.0 \
docker-compose up go2_sim behav director
```

### Interactive director (change instruction at runtime)
```bash
docker-compose up -d go2_sim behav
docker exec -it behav_director python3 /app/planning/behav-director.py --interactive
```

### Demo mode (no OpenAI key required)
```bash
docker-compose up go2_sim behav
docker exec -it behav_director python3 /app/planning/behav-director.py --demo
```

### Other useful services
```bash
# Landmark detector (GPU)
docker-compose --profile landmark up landmark_detector

# Instruction decomposition standalone test
docker-compose --profile test run --rm instruction_decomposition --demo
```

## ROS2 Interface

**behav-planner subscriptions:** `/odom_lidar`, `/camera/color/image_raw`, `/scan`, `/behav_config`
**behav-planner publications:** `/mcu/command/manual_twist`, `/behav_costmap`, `/traj_marked_image`
**behav-director publications:** `/behav_config`

Topic remaps for Go2 sim (set in docker-compose):
- `/odom_lidar` → `/robot1/odom`
- `/camera/color/image_raw` → `/robot1/camera/image_raw`
- `/mcu/command/manual_twist` → `/robot1/cmd_vel`

## Key Patterns

- **Ego-polar coordinates** (r, δ, θ) used throughout motion planning
- **Two-stage optimization**: Global (GN_ESCH) then local (LN_BOBYQA) using NLopt
- **CLIPSeg cost values**: higher = preferred region (e.g., pavement=0.9), lower = avoid (e.g., grass=0.1)
- **`/behav_config` JSON schema**: `{"prompts": [...], "costs": [...], "goal_radius": f, "goal_theta": f, "goal_delta": f}`
- **Thread synchronization** via mutexes and condition variables for real-time ROS2 operation

## Configuration

Key parameters in `behav-planner.py`:
- Control gains: `K1=1.2, K2=1, BETA=0.4, LAMBDA=2`
- Velocity limits: `V_MAX=0.8, V_MIN=0.0`
- Time horizon: 4 seconds with 0.5s simulation steps
- CLIPSeg prompts/costs updated live via `/behav_config`

Environment variables for `director` service (set in `.env` or shell):
| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for GPT-4 decomposition |
| `BEHAV_INSTRUCTION` | `"Stay on the pavement and avoid grass"` | Navigation instruction |
| `BEHAV_GOAL_RADIUS` | `5.0` | Goal distance (m) |
| `BEHAV_GOAL_THETA` | `0.0` | Goal heading (°) |
| `BEHAV_GOAL_DELTA` | `0.0` | Goal pose angle (°) |
| `ROS_DOMAIN_ID` | `0` | DDS domain |
