#!/bin/bash

source /opt/ros/humble/setup.bash
cd $HOME/training_data
ros2 bag record --use-sim-time /RHCViz_kyon0_HandShake /RHCViz_kyon0_hl_refs /RHCViz_kyon0_rhc_actuated_jointnames /RHCViz_kyon0_rhc_q /RHCViz_kyon0_rhc_refs /RHCViz_kyon0_robot_actuated_jointnames /RHCViz_kyon0_robot_q 
