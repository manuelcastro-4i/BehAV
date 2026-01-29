#!/bin/bash
set -e

# Source ROS2 setup
source /opt/ros/jazzy/setup.bash

# Source Gazebo setup
if [ -f /usr/share/gazebo/setup.bash ]; then
    source /usr/share/gazebo/setup.bash
fi

# Source workspace if it exists
if [ -f /app/install/setup.bash ]; then
    source /app/install/setup.bash
fi

# Execute the command
exec "$@"
