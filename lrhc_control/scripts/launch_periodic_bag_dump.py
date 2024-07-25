import argparse
import os
import signal
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper

from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererClnt,RemoteTriggererSrvr
import rosbag2_py

from SharsorIPCpp.PySharsorIPC import StringTensorClient
from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo

from perf_sleep.pyperfsleep import PerfSleep

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set affinity to')
    parser.add_argument('--dt', type=float, default=0.01, help='Update interval in seconds, default is 0.01')
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, default is False')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Enable verbose mode, default is True')
    parser.add_argument('--ros2', action=argparse.BooleanOptionalAction, default=True, help='Use ROS 2')
    parser.add_argument('--with_agent_refs', action=argparse.BooleanOptionalAction, default=False, help='also forward agent refs to rhcviz')
    parser.add_argument('--rhc_refs_in_h_frame', type=bool, default=True, help='set to true if rhc refs are \
                        specified in the horizontal frame')
    parser.add_argument('--agent_refs_in_h_frame', type=bool, default=False, help='set to true if agent refs are \
                        specified in the horizontal frame')
    parser.add_argument('--env_idx', type=int, help='env index of which data is to be published', default=-1)
    parser.add_argument('--stime_trgt', type=float, default=None, help='sim time for which this bridge runs (None -> indefinetly)')
    parser.add_argument('--srdf_path', type=str, help='path to SRDF path specifying homing configuration, to be used for missing joints', default=None)
    parser.add_argument('--dump_rosbag', action=argparse.BooleanOptionalAction, default=False, help='whether to dump a rosbag of the published topics')
    parser.add_argument('--dump_path', type=str, default="/tmp", help='where bag will be dumped')
    parser.add_argument('--use_shared_drop_dir', action=argparse.BooleanOptionalAction, default=False, 
        help='if true use the shared drop dir to drop the data where all the other training data is dropeer')

    args = parser.parse_args()

    periodic_bag_dumper=RosBagDumper()

    shared_info=SharedRLAlgorithmInfo(is_server=False,
                    namespace=args.ns, 
                    verbose=True, 
                    vlevel=VLevel.V2)
    shared_info.run()
    shared_info_names=shared_info.dynamic_info.get()
    is_done_idx=shared_info_names.index("is_done")

    training_done=False
    while not training_done:
        
        periodic_bag_dumper.record()

        algo_data = shared_info.get().flatten()
        training_done=algo_data[is_done_idx]>0.5
    
    shared_info.close()
    periodic_bag_dumper.close()