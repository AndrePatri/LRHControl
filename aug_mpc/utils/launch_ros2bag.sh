#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 --ns <namespace> --id <bag_id> --xbot <0/1> [--output_path <path>]"
    exit 1
}

# Function to handle SIGINT
sigint_handler() {
    echo "[launch_rosbag.sh]:SIGINT received, exiting..."
    exit 0
}

source_ros2_setup() {
    local distro
    for distro in jazzy humble iron rolling; do
        if [ -f "/opt/ros/${distro}/setup.bash" ]; then
            echo "[launch_rosbag.sh]: sourcing ROS 2 ${distro}"
            # shellcheck source=/dev/null
            source "/opt/ros/${distro}/setup.bash"
            return 0
        fi
    done

    echo "[launch_rosbag.sh]: no ROS 2 setup.bash found under /opt/ros" >&2
    return 1
}
# Set the trap to catch SIGINT and call sigint_handler
trap sigint_handler SIGINT

# Ensure the required arguments are provided
if [ "$#" -lt 8 ] || [ "$1" != "--ns" ] || [ "$3" != "--id" ] || [ "$5" != "--xbot" ]; then
    usage
fi

# Extract the namespace from the arguments
NAMESPACE=$2
BAG_ID=$4
ADD_XBOT_TOPICS=$6

# Validate ADD_XBOT_TOPICS input
if [[ "$ADD_XBOT_TOPICS" != "0" && "$ADD_XBOT_TOPICS" != "1" ]]; then
    echo "Error: --xbot must be 0 or 1, provided $ADD_XBOT_TOPICS"
    usage
fi

# Default output path to /tmp with namespace and date-time
OUTPUT_PATH="/tmp/rosbag_${NAMESPACE}_$(date +%Y-%m-%d_%H-%M-%S)_${BAG_ID}"
# Check for optional --output_path argument
if [ "$#" -eq 8 ] && [ "$7" == "--output_path" ]; then
    OUTPUT_PATH="$8/rosbag_${NAMESPACE}_$(date +%Y-%m-%d_%H-%M-%S)_${BAG_ID}"
fi

# Source ROS setup
source_ros2_setup

# Change to the training data directory
cd "$HOME/training_data"

# Define the topics with the namespace replaced
TOPICS=(
    "/MPCViz_${NAMESPACE}_HandShake"
    "/MPCViz_${NAMESPACE}_hl_refs"
    "/MPCViz_${NAMESPACE}_rhc_actuated_jointnames"
    "/MPCViz_${NAMESPACE}_rhc_q"
    "/MPCViz_${NAMESPACE}_rhc_refs"
    "/MPCViz_${NAMESPACE}_rhc_contacts"
    "/MPCViz_${NAMESPACE}_root_wrench"
    "/MPCViz_${NAMESPACE}_root_wrench_point"
    "/MPCViz_${NAMESPACE}_root_wrench_marker"
    "/MPCViz_${NAMESPACE}_robot_actuated_jointnames"
    "/MPCViz_${NAMESPACE}_robot_q"
    "/MPCViz_${NAMESPACE}_heightmap"
)

# Add XBot topics if requested
if [ "$ADD_XBOT_TOPICS" = "1" ]; then
    XBOT_TLIST=(
        "/xbotcore/command"
        "/xbotcore/imu/imu_link"
        "/xbotcore/joint_device_info"
        "/xbotcore/joint_states"
        "/xbotcore/lifecycle_events"
        "/xbotcore/statistics"
        "/xbotcore/status"
    )
    TOPICS+=("${XBOT_TLIST[@]}")
fi

# Record the topics. Use exec so SIGINT reaches rosbag2 and bag_dumper waits on the actual recorder.
trap - SIGINT
exec ros2 bag record --compression-mode file --compression-format zstd --use-sim-time "${TOPICS[@]}" -o "$OUTPUT_PATH"
