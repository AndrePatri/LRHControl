from control_cluster_bridge.utilities.debugger_gui.cluster_debugger import RtClusterDebugger
from control_cluster_bridge.utilities.debugger_gui.gui_exts import JntImpMonitor
from lrhc_control.utils.gui_exts.training_env import TrainingEnvData

import os
import argparse

# Function to set CPU affinity
def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

if __name__ == "__main__":  
    
    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set affinity to')
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--dt_data', type=float, help='Dt at which data on shared memory is sampled',
                default=0.01)
    parser.add_argument('--dt_plot', type=float, help='Dt at which plots are updated',
                default=0.01)
    parser.add_argument('--w_length', type=float, help='length [s] of the data window into the past',
                default=5.0)
    
    args = parser.parse_args()
    
    # Set CPU affinity if cores are provided
    if args.cores:
        set_affinity(args.cores)

    data_update_dt = args.dt_data
    plot_update_dt = args.dt_plot

    window_length = args.w_length
    window_buffer_factor = 1

    cluster_debugger = RtClusterDebugger(data_update_dt=data_update_dt,
                            plot_update_dt=plot_update_dt,
                            window_length=window_length, 
                            window_buffer_factor=window_buffer_factor, 
                            verbose=True, 
                            namespace=args.ns)
    
    # adding some of the available extensions to the GUI
    gui_extensions = []
    gui_extensions.append(JntImpMonitor(update_data_dt = data_update_dt,
            update_plot_dt = plot_update_dt,
            window_duration = window_length,
            window_buffer_factor = window_buffer_factor,
            namespace = args.ns,
            verbose = True))
    gui_extensions.append(TrainingEnvData(update_data_dt = data_update_dt,
            update_plot_dt = plot_update_dt,
            window_duration = window_length,
            window_buffer_factor = window_buffer_factor,
            namespace = args.ns,
            verbose = True,
            add_settings_tab = True))

    cluster_debugger.add_spawnable_tab(gui_extensions)

    cluster_debugger.run()
