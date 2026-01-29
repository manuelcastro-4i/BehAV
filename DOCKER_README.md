# BehAV Docker Setup

This guide explains how to run BehAV with the Go2 robot simulator using Docker.

## Prerequisites

1. **Docker** with Docker Compose v2
2. **NVIDIA Docker Runtime** (nvidia-container-toolkit)
3. **X11 Server** for GUI display (Gazebo visualization)
4. **OpenAI API Key** for GPT-4 Vision

## Quick Start

### 1. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Download model weights

Download FastSAM-x.pt (~200MB) from [FastSAM releases](https://github.com/CASIA-IVA-Lab/FastSAM) and place it in the `models/` directory:

```bash
wget -O models/FastSAM-x.pt https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-x.pt
```

### 3. Enable X11 forwarding (Linux)

```bash
xhost +local:docker
```

### 4. Build and run

```bash
# Build containers
docker-compose build

# Start the simulator and planner
docker-compose up
```

## Services

| Service | Description | Port |
|---------|-------------|------|
| `go2_sim` | Gazebo simulator with Go2 robot | - |
| `behav` | BehAV navigation planner | - |
| `landmark_detector` | Landmark detection (optional) | - |

## ROS2 Topic Mapping

| BehAV Topic | Go2 Simulator Topic | Message Type |
|-------------|---------------------|--------------|
| `/odom_lidar` | `/robot1/odom` | nav_msgs/Odometry |
| `/camera/color/image_raw` | `/robot1/camera/image_raw` | sensor_msgs/Image |
| `/mcu/command/manual_twist` | `/robot1/cmd_vel` | geometry_msgs/Twist |
| `/scan` | `/robot1/scan` | sensor_msgs/LaserScan |

## Configuration

### Goal Parameters

Set navigation goals via ROS2 parameters or docker-compose command:

```yaml
# In docker-compose.yml
command: >
  python3 /app/planning/behav-planner.py
  --ros-args
  -p goal_radius:=5.0    # Distance in meters
  -p goal_theta:=30.0    # Angle in degrees (left positive)
  -p goal_delta:=0.0     # Final pose angle
```

### Camera Calibration

For custom cameras, update these parameters:

```yaml
-p camera_fx:=554.25
-p camera_fy:=554.25
-p camera_cx:=320.0
-p camera_cy:=240.0
-p camera_height:=0.35
```

### Using YAML Configuration

```bash
docker-compose run --rm behav \
  python3 /app/planning/behav-planner.py \
  --ros-args --params-file /app/config/behav_params.yaml
```

## Running Individual Components

### Simulator Only

```bash
docker-compose up go2_sim
```

### BehAV Planner Only

```bash
docker-compose up behav
```

### Landmark Detector

```bash
docker-compose --profile landmark up landmark_detector
```

## Development

### Enter Container Shell

```bash
# Simulator container
docker-compose exec go2_sim bash

# BehAV container
docker-compose exec behav bash
```

### View ROS2 Topics

```bash
docker-compose exec behav ros2 topic list
docker-compose exec behav ros2 topic echo /robot1/odom --once
```

### Rebuild After Code Changes

```bash
docker-compose build --no-cache behav
```

## Troubleshooting

### GPU Not Detected

Verify NVIDIA runtime:
```bash
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Display Issues

1. Check DISPLAY variable: `echo $DISPLAY`
2. Run `xhost +local:docker`
3. Verify X11 socket: `ls /tmp/.X11-unix/`

### ROS2 Communication Issues

1. Verify same ROS_DOMAIN_ID in both containers
2. Check CycloneDDS is configured: `echo $RMW_IMPLEMENTATION`
3. Test topic visibility: `ros2 topic list`

## Architecture

```
+------------------------------------------+
|        Docker Network (ros2_net)         |
|   ROS_DOMAIN_ID=0, CycloneDDS            |
+------------------------------------------+
        |                    |
+---------------+    +------------------+
|   go2_sim     |    |     behav        |
|---------------|    |------------------|
| ROS2 Jazzy    |    | ROS2 Jazzy       |
| Gazebo        |    | CUDA 12.1        |
| Go2 URDF      |    | PyTorch+CLIPSeg  |
| Camera Plugin |    | FastSAM + nlopt  |
+---------------+    +------------------+
```
