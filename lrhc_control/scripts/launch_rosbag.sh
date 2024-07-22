#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 --ns <namespace> [--output_path <path>]"
    exit 1
}

# Check if the required argument is provided
if [ "$#" -lt 2 ] || [ "$1" != "--ns" ]; then
    usage
fi

# Extract the namespace from the arguments
NAMESPACE=$2

# Default output path to /tmp with namespace and date-time
OUTPUT_PATH="/tmp/rosbag_${NAMESPACE}_$(date +%Y-%m-%d_%H-%M-%S)"

# Check for optional --output_path argument
if [ "$#" -eq 4 ] && [ "$3" == "--output_path" ]; then
    OUTPUT_PATH="$4/rosbag_${NAMESPACE}_$(date +%Y-%m-%d_%H-%M-%S)"
elif [ "$#" -gt 2 ]; then
    usage
fi

# Source ROS setup
source /opt/ros/humble/setup.bash

# Change to the training data directory
cd $HOME/training_data

# Define the topics with the namespace replaced
TOPICS=(
    "/RHCViz_${NAMESPACE}_HandShake"
    "/RHCViz_${NAMESPACE}_hl_refs"
    "/RHCViz_${NAMESPACE}_rhc_actuated_jointnames"
    "/RHCViz_${NAMESPACE}_rhc_q"
    "/RHCViz_${NAMESPACE}_rhc_refs"
    "/RHCViz_${NAMESPACE}_robot_actuated_jointnames"
    "/RHCViz_${NAMESPACE}_robot_q"
)

# Record the topics
ros2 bag record --use-sim-time "${TOPICS[@]}" -o "$OUTPUT_PATH"
