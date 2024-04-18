#!/bin/bash

# Set Italian keyboard layout
export XMODIFIERS=@im=ibus
export GTK_IM_MODULE=ibus
export QT_IM_MODULE=ibus

SLEEP_FOR=0.1
BYOBU_WS_NAME="RlWorkspace"
WS_ROOT="$HOME/RL_ws/hhcm"
WORKING_DIR="$WS_ROOT/src/LRHControl/lrhc_control/scripts"
MAMBAENVNAME="LRHControlMambaEnv"
N_FILES=14336 # to allow more open files

# Array of directories
directories=(
    "$WS_ROOT/src/LRHControl"
    "$WS_ROOT/src/CoClusterBridge"
    "$WS_ROOT/src/OmniRoboGym"
    "$WS_ROOT/src/SharsorIPCpp"
    "$WS_ROOT/src/horizon"
    # Add more directories as needed
)

press_enter() {

    byobu send-keys Enter
    sleep $SLEEP_FOR
}

# Function to execute common commands in Terminator terminal
execute_command() {
    byobu send-keys "$1"
    press_enter
    sleep $SLEEP_FOR
}

prepare_command() {
    byobu send-keys "$1"
    sleep $SLEEP_FOR
}

go_to_pane() {

    byobu select-pane -t $1

}

go_to_window() {

    byobu select-window -t $1

}

attach_to_session() {

    byobu attach-session -t ${BYOBU_WS_NAME} 

}

source_mamba_env() {

    execute_command "mamba activate ${MAMBAENVNAME}"

}

clear_terminal() {

    execute_command "clear"

}

increase_file_limits_locally() {

    # for shared memory

    execute_command "ulimit -n ${N_FILES}"

}

split_h() {

    byobu split-window -p 50 -v

}

split_v() {

    byobu split-window -p 50 -h

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
byobu kill-session -t ${BYOBU_WS_NAME}

byobu new-session -d -s ${BYOBU_WS_NAME} -c ${WORKING_DIR} -n ${BYOBU_WS_NAME} # -d "detached" session

# tab 0
execute_command "cd ${WORKING_DIR}"
source_mamba_env
#execute_command "source ~/.local/share/ov/pkg/isaac_sim-2023.1.1/setup_conda_env.sh"
execute_command "source /isaac-sim/setup_conda_env.sh"
execute_command "source $WS_ROOT/setup.bash"
increase_file_limits_locally 
clear_terminal
prepare_command "reset && python launch_sim_env.py --headless --remote_stepping --robot_name {} --robot_pkg_name {} --num_envs {}"

split_v
execute_command "cd ${WORKING_DIR}"
source_mamba_env
execute_command "source $WS_ROOT/setup.bash"
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_control_cluster.py --enable_debug --ns {} --robot_pkg_name {} --size {}"

split_h
execute_command "cd ${WORKING_DIR}"
source_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_GUI.py --cores {} --ns {}"

split_h
execute_command "cd ${WORKING_DIR}"
source_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_keyboard_cmds.py --ns {}"

go_to_pane 0 

split_h
execute_command "cd ${WORKING_DIR}"
source_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_train_env.py --ns {} --run_name {} --drop_dir $HOME/results --comment "" "

split_h
execute_command "cd ${WORKING_DIR}"
# execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source /opt/ros/humble/setup.bash"
execute_command "source $WS_ROOT/setup.bash"
source_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_rhc2ros_bridge.py --ros2 --ns {} --cores {} --dt {}"

# tab 1
new_tab
execute_command "cd ${WORKING_DIR}"

split_h
execute_command "cd ${WORKING_DIR}"
# execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source /opt/ros/humble/setup.bash"
source_mamba_env
clear_terminal
prepare_command "reset && python3 launch_rhcviz.py --ns {} --dpath {} --nodes_perc 10 --cores {}"

# tab2
new_tab
execute_command "htop"

split_h
execute_command "cd ${WORKING_DIR}"
execute_command "nvtop"
clear_terminal

# tab 3

new_tab

# Loop through directories and navigate to each one
for dir in "${directories[@]}"; do
    cd_and_split "$dir"
done

# we attach to the detached session
attach_to_session