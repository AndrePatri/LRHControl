#!/bin/bash

set -euo pipefail

# Check if rosbag file argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <rosbag_file_name> [-r playback_rate] [--ros1] [--no-pause]"
    exit 1
fi

# Parse arguments
rosbag_file=$1
playback_rate=1
use_ros1=false
pause_playback=true

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        -r)
            playback_rate=$2
            shift 2
            ;;
        --ros1)
            use_ros1=true
            shift
            ;;
        --no-pause)
            pause_playback=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Play rosbag file using the appropriate ROS version
safe_source() {
    set +u
    source "$1"
    set -u
}

if [ "$use_ros1" = true ]; then
    echo "Replaying rosbag using ROS1 Noetic"
    safe_source /opt/ros/noetic/setup.bash
    rosparam set /use_sim_time true
    rosbag_cmd=(rosbag play -r "$playback_rate")
    if [ "$pause_playback" = true ]; then
        rosbag_cmd+=(--pause)
    fi
    rosbag_cmd+=("$rosbag_file")
    "${rosbag_cmd[@]}"
else
    echo "Replaying rosbag using ROS2 Humble"
    safe_source /opt/ros/humble/setup.bash
    ros2_cmd=(ros2 bag play -r "$playback_rate")
    if [ "$pause_playback" = true ]; then
        ros2_cmd+=(-p)
    fi
    ros2_cmd+=("$rosbag_file")
    "${ros2_cmd[@]}"
fi
