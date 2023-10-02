import os
script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

from lrhc_examples.controllers.aliengo_rhc.rhc import AliengoRHC
from lrhc_examples.controllers.aliengo_rhc.rhc_cluster_srvr import AliengoRHClusterSrvr

import torch

from perf_sleep.pyperfsleep import PerfSleep

def generate_controllers(robot_name: str):

    # create controllers
    cluster_controllers = []
    for i in range(0, control_cluster_srvr.cluster_size):

        cluster_controllers.append(AliengoRHC(
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
control_cluster_srvr = AliengoRHClusterSrvr(robot_name) # this blocks until connection with the client is established
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
