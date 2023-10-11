# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of LRhcExamples and distributed under the General Public License version 2 license.
# 
# LRhcExamples is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# LRhcExamples is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with LRhcExamples.  If not, see <http://www.gnu.org/licenses/>.
# 
import os
script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

from lrhc_examples.controllers.rhc.rhc import RHC
from lrhc_examples.controllers.rhc.rhc_cluster_srvr import RHClusterSrvr

import torch

from perf_sleep.pyperfsleep import PerfSleep

def generate_controllers(robot_name: str):

    # create controllers
    cluster_controllers = []
    for i in range(0, control_cluster_srvr.cluster_size):

        cluster_controllers.append(RHC(
                                    controller_index = i,
                                    cluster_size=control_cluster_srvr.cluster_size,
                                    srdf_path=control_cluster_srvr._srdf_path,
                                    robot_name=robot_name,
                                    verbose = verbose, 
                                    debug = debug,
                                    array_dtype = dtype))
    
    return cluster_controllers

verbose = True
debug = True

perf_timer = PerfSleep()

dtype = torch.float32 # this has to be the same wrt the cluster client, otherwise
# messages are not read properly

robot_name = "aliengo0"
control_cluster_srvr = RHClusterSrvr(robot_name) # this blocks until connection with the client is established
controllers = generate_controllers(robot_name)

for i in range(0, control_cluster_srvr.cluster_size):
    
    # we add the controllers

    result = control_cluster_srvr.add_controller(controllers[i])

control_cluster_srvr.start() # spawns the controllers on separate processes

try:

    while True:
        
        nsecs = int(0.1 * 1e9)
        perf_timer.clock_sleep(nsecs) # we don't want to drain all the CPU
        # with a busy wait

        pass

except KeyboardInterrupt:

    # This block will execute when Control-C is pressed
    print(f"[{script_name}]" + "[info]: KeyboardInterrupt detected. Cleaning up...")

    control_cluster_srvr.terminate() # closes all processes

    import sys
    sys.exit()
