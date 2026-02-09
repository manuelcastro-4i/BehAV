#!/bin/bash
set -e

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f /app/install/setup.bash ]; then
    source /app/install/setup.bash
fi

# Execute the command
exec "$@"
