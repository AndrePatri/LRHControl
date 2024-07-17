#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 --ns <namespace>"
    exit 1
}

# Check if the argument is provided
if [ "$#" -ne 2 ] || [ "$1" != "--ns" ]; then
    usage
fi

# Extract the namespace from the arguments
NAMESPACE=$2

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
ros2 bag record --use-sim-time "${TOPICS[@]}"