#!/bin/bash

# Check if rosbag file argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <rosbag_file_name> [-r playback_rate]"
    exit 1
fi

# Source ROS setup
source /opt/ros/humble/setup.bash

# Parse arguments
rosbag_file=$1
playback_rate=${2:-1}

# Play rosbag file
ros2 bag play -p -r $playback_rate $rosbag_file
