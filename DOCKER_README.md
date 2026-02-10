# BehAV Docker Setup

This guide explains how to run the BehAV navigation system modules using Docker Compose.

## Prerequisites

1. **Docker** with Docker Compose v2
2. **NVIDIA Docker Runtime** (nvidia-container-toolkit) - only for `behav` and `go2_sim`
3. **OpenAI API Key** for GPT-4 / GPT-4o Vision

## Quick Start

### 1. Set up environment variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-key-here
DISPLAY=:0
ROS_DOMAIN_ID=0
```

### 2. Download model weights (for Landmark Detection)

Download FastSAM-x.pt (~140MB) and place it in the `models/` directory:

```bash
mkdir -p models
wget -O models/FastSAM-x.pt https://huggingface.co/conrevo/Segment-Anything-A1111/resolve/main/FastSAM-x.pt
```

---

## Services Overview

| Service | Profile | GPU | ROS2 | Description |
|---------|---------|-----|------|-------------|
| `instruction_decomposition` | `test` | No | No | Parses natural language instructions into components and behavioral costs |
| `landmark_test` | `test` | No | No | Detects landmarks in images using FastSAM + GPT-4o Vision |
| `behav` | default | Yes | Yes | MPC navigation planner with CLIPSeg behavioral cost maps |
| `go2_sim` | default | Yes | Yes | Gazebo Harmonic simulator with Go2 robot (headless + noVNC) |
| `landmark_detector` | `landmark` | Yes | Yes | ROS2 landmark detection node (requires `go2_sim`) |

---

## 1. Instruction Decomposition

Lightweight service (no GPU, no ROS2). Decomposes natural language navigation instructions into:
- **Landmarks** (e.g., building, stop sign)
- **Navigation actions** (e.g., turn left, go forward)
- **Behavioral actions** (e.g., stay on, avoid) with associated costs

### Build

No build needed — uses `python:3.10-slim` image directly.

### Run

```bash
# Show help
docker-compose --profile test run --rm instruction_decomposition --help

# Run demo with example instruction
docker-compose --profile test run --rm instruction_decomposition --demo

# Process a custom instruction
docker-compose --profile test run --rm instruction_decomposition -I "Turn left at the tree, stay on the path"

# Interactive mode - enter multiple instructions
docker-compose --profile test run --rm instruction_decomposition -i

# Verbose mode (shows raw GPT-4 responses)
docker-compose --profile test run --rm instruction_decomposition -v --demo
```

### Example Output

```
======================================================================
INSTRUCTION DECOMPOSITION RESULTS
======================================================================

Input Instruction:
  "Go forward until you see a stop sign, then turn left..."

----------------------------------------------------------------------
EXTRACTED COMPONENTS
----------------------------------------------------------------------

  Landmarks:          "stop sign", "white building"
  Navigation Actions: "go forward", "turn left", "go straight"
  Behavioral Actions: "stay on", "stop for", "stay away from"
  Behavioral Targets: "pavements", "red traffic lights", "grass"

----------------------------------------------------------------------
BEHAVIORAL COST MAPPING
----------------------------------------------------------------------

  Reference Actions & Costs:
    Stay on    -> 0
    Avoid      -> 0.5
    Yield      -> 0.7
    Stop       -> 1

  Computed Costs for Input Actions:
  Action                      Assigned Cost
  ----------------------------------------
  stay on                                 0
  stop for                                1
  stay away from                        0.5

======================================================================
```

---

## 2. Landmark Detection (Standalone)

CPU-based standalone service for testing landmark detection. Uses FastSAM for image segmentation and GPT-4o Vision to identify landmarks by comparing a ground truth reference image with a test image.

**Requires:** `models/FastSAM-x.pt` (see Quick Start step 2).

### Build

```bash
docker-compose --profile test build landmark_test
```

### Run

```bash
# Show help
docker-compose --profile test run --rm landmark_test --help

# Demo with Testudo statue (default)
docker-compose --profile test run --rm landmark_test --demo

# Demo with other landmarks
docker-compose --profile test run --rm landmark_test --demo --landmark Iribe
docker-compose --profile test run --rm landmark_test --demo --landmark Chapel

# Custom images (paths relative to /app inside container)
docker-compose --profile test run --rm landmark_test \
  -g landmark-tracking/Images/Testudo/Ground_truth.webp \
  -t landmark-tracking/Images/Testudo/2.jpg \
  --ref-distance 3

# Verbose mode (shows raw GPT-4o Vision response)
docker-compose --profile test run --rm landmark_test --demo -v
```

Available demo landmarks: `Testudo` (3m), `Iribe` (80m), `Douglass` (15m), `M_circle` (21.38m), `Idea_factory` (29.1m), `Chapel` (32m)

Output images are saved to `./output/`.

### Example Output

```
Running demo with Testudo landmark
  Ground truth: /app/landmark-tracking/Images/Testudo/Ground_truth.webp
  Test image:   /app/landmark-tracking/Images/Testudo/1.jpg
  Ref distance: 3.0m

0: 1024x800 85 objects, 5207.4ms

======================================================================
LANDMARK DETECTION RESULTS
======================================================================

  Landmark Present: YES

----------------------------------------------------------------------
LOCALIZATION
----------------------------------------------------------------------

  Mask Number:      1
  Estimated Distance: Approximately 10 meters
  Pixel Location:   X=600, Y=204

  Processing Time:  19.07 seconds

======================================================================
```

---

## 3. Go2 Simulator (Gazebo Harmonic)

ROS2 Jazzy + Gazebo Harmonic simulator for the Go2 quadruped robot. Runs **headless by default** (no display needed) and optionally provides a **browser-based GUI via noVNC** — works on both Windows and Linux without X11 forwarding.

**Note:** Uses ROS2 Jazzy (required by the `go2_ros2_sim_py` repo). Communication with `behav` (Humble) works via DDS, which is compatible across ROS2 distros.

### Build

```bash
docker-compose build go2_sim
```

### Run (headless, default)

```bash
# Start simulator headless (no GUI)
docker-compose up go2_sim

# Verify topics are published
docker-compose exec go2_sim bash -c "source /opt/ros/jazzy/setup.bash && ros2 topic list"
```

### Run (with noVNC GUI)

```bash
# Start with browser-based GUI
ENABLE_VNC=true docker-compose up go2_sim

# On Windows (PowerShell):
$env:ENABLE_VNC="true"; docker-compose up go2_sim
```

Then open **http://localhost:6080** in your browser to see the Gazebo GUI.

### Run (full stack)

```bash
# Start simulator + planner
docker-compose up

# With GUI
ENABLE_VNC=true docker-compose up
```

---

## 4. BehAV Navigation Planner

The core navigation planner. Uses ROS2 Humble, CUDA 12.1, PyTorch, CLIPSeg and MPC optimization to generate trajectories from behavioral cost maps.

**Requires:** `go2_sim` running (or another ROS2 source providing odometry, camera, and lidar topics).

### Build

```bash
docker-compose build behav
```

### Run (with simulator)

```bash
# Start simulator + planner together
docker-compose up

# Or start planner only (requires go2_sim already running)
docker-compose up behav
```

### Run (standalone shell for testing)

```bash
# Enter container shell to verify dependencies
docker-compose run --rm --no-deps behav bash

# Inside the container:
source /opt/ros/humble/setup.bash
python3 -c "import rclpy, torch, cv_bridge, nlopt, transformers; print('All OK')"
```

### Configuration

Goal parameters via docker-compose command:

```yaml
command: >
  python3 /app/planning/behav-planner.py
  --ros-args
  -p goal_radius:=5.0    # Distance in meters
  -p goal_theta:=30.0    # Angle in degrees (left positive)
  -p goal_delta:=0.0     # Final pose angle
  -p publish_to_motors:=true
```

Camera calibration:

```yaml
-p camera_fx:=554.25
-p camera_fy:=554.25
-p camera_cx:=320.0
-p camera_cy:=240.0
-p camera_height:=0.35
```

Using YAML config file:

```bash
docker-compose run --rm behav \
  python3 /app/planning/behav-planner.py \
  --ros-args --params-file /app/config/behav_params.yaml
```

### ROS2 Topic Mapping (Go2 Simulator)

| BehAV Topic | Go2 Simulator Topic | Message Type |
|-------------|---------------------|--------------|
| `/odom_lidar` | `/robot1/odom` | nav_msgs/Odometry |
| `/camera/color/image_raw` | `/robot1/camera/image_raw` | sensor_msgs/Image |
| `/mcu/command/manual_twist` | `/robot1/cmd_vel` | geometry_msgs/Twist |
| `/scan` | `/robot1/scan` | sensor_msgs/LaserScan |

---

## Development

### Enter Container Shell

```bash
docker-compose exec behav bash
docker-compose exec go2_sim bash
```

### View ROS2 Topics

```bash
docker-compose exec behav ros2 topic list
docker-compose exec behav ros2 topic echo /robot1/odom --once
```

### Rebuild After Code Changes

```bash
docker-compose build --no-cache behav
docker-compose --profile test build landmark_test
```

## Troubleshooting

### GPU Not Detected

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### ROS2 Communication Issues

1. Verify same `ROS_DOMAIN_ID` in both containers
2. Check CycloneDDS: `echo $RMW_IMPLEMENTATION`
3. Test topic visibility: `ros2 topic list`

## Architecture

```
+------------------------------------------+
|        Docker Network (ros2_net)          |
|   ROS_DOMAIN_ID=0, CycloneDDS            |
+------------------------------------------+
        |                    |
+---------------+    +------------------+
|   go2_sim     |    |     behav        |
|---------------|    |------------------|
| ROS2 Jazzy    |    | ROS2 Humble      |
| Gazebo Harm.  |    | CUDA 12.1        |
| Headless/VNC  |    | PyTorch+CLIPSeg  |
| noVNC :6080   |    | FastSAM + nlopt  |
+---------------+    +------------------+

Standalone (no ROS2):
+---------------------------+    +---------------------------+
| instruction_decomposition |    |     landmark_test         |
|---------------------------|    |---------------------------|
| python:3.10-slim          |    | python:3.10-slim          |
| GPT-4                     |    | FastSAM (CPU) + GPT-4o   |
| No GPU needed             |    | No GPU needed             |
+---------------------------+    +---------------------------+
```
