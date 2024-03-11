#!/bin/bash

# Set Italian keyboard layout
export XMODIFIERS=@im=ibus
export GTK_IM_MODULE=ibus
export QT_IM_MODULE=ibus

SLEEP_FOR=0.5
WS_NAME="RlWorkspace"
WORKING_DIR="$HOME/RL_ws/hhcm/src/KyonRLStepping/kyonrlstepping/scripts"
MAMBAENVNAME="KyonRLSteppingIsaac2023.1.0"

# Array of directories
directories=(
    "$HOME/RL_ws/hhcm/src/KyonRLStepping"
    "$HOME/RL_ws/hhcm/src/CoClusterBridge"
    "$HOME/RL_ws/hhcm/src/OmniRoboGym"
    "$HOME/RL_ws/hhcm/src/horizon"
    "$HOME/RL_ws/hhcm/build/horizon"
    "$HOME/RL_ws/hhcm/src/phase_manager"
    "$HOME/RL_ws/hhcm/build/phase_manager"
    "$HOME/RL_ws/hhcm/src/SharsorIPCpp"
    "$HOME/RL_ws/hhcm/build/SharsorIPCpp"
    # Add more directories as needed
)

# Function to execute common commands in Terminator terminal
execute_command() {
    xdotool type --delay 1 "$1"
    xdotool key Return
    sleep $SLEEP_FOR
    xdotool key Ctrl+L
}

prepare_command() {
    xdotool type --delay 1 "$1"
    sleep $SLEEP_FOR
}

source_mamba_env() {

    execute_command "mamba activate ${MAMBAENVNAME}"

}

split_h() {

    xdotool key Ctrl+Shift+o 

}

split_v() {

    xdotool key Ctrl+Shift+e

}

new_tab() {

    byobu new-window

}

# Function to navigate to a directory and split Terminator horizontally
cd_and_split() {

    execute_command "cd $1"
    
    # Check if it's the last directory before splitting
    if [ "$1" != "${directories[-1]}" ]; then
    
        split_h

    fi
}

# launch terminator window
terminator -m -T ${WS_NAME} --working-directory=${WORKING_DIR}

# tab 0
source_mamba_env
execute_command "source ~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/setup_conda_env.sh"

prepare_command "reset && python KyonPlay.py"

split_v

source_mamba_env
prepare_command "reset && python RunControlCluster.py"

split_h

source_mamba_env
prepare_command "reset && python RunDebuggerGUI.py"

split_h

source_mamba_env
prepare_command "reset && python RunKeyboardCmds.py"

xdotool key Alt+Left

split_h

execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source ~/RL_ws/hhcm/setup.bash"
source_mamba_env

prepare_command "reset && python RunRhc2RosBridge.py kyon0"

# tab 1

new_tab
execute_command "source /opt/ros/noetic/setup.bash"
execute_command "roscore"

split_h

execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source ~/RL_ws/hhcm/setup.bash"

prepare_command "reset && python3 run_rhc2viz.py --nodes_perc 10 --robot_type kyon --robot_name kyon0"

# tab 2

new_tab

execute_command "htop"

split_h

execute_command "nvtop"
xdotool key Return

# tab 3

new_tab

# Loop through directories and navigate to each one
for dir in "${directories[@]}"; do
    cd_and_split "$dir"
done
