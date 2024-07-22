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

from perf_sleep.pyperfsleep import PerfSleep

def launch_rosbag(namespace: str, dump_path: str, run_for_sec:float):
        
    import multiprocess as mp

    retry_kill=20
    additional_secs=5.0
    term_trigger=RemoteTriggererClnt(namespace=namespace+f"SharedTerminator",
                            verbose=True,
                            vlevel=VLevel.V1)
    term_trigger.run()

    Journal.log("launch_rhc2ros_bridge.py",
            "launch_rosbag",
            "launching rosbag recording",
            LogType.INFO)

    command = ["./launch_rosbag.sh", "--ns", namespace, "--output_path", dump_path]
    ctx = mp.get_context('forkserver')
    proc = ctx.Process(target=os.system, args=(' '.join(command),))
    proc.start()

    if not term_trigger.wait(240000):
        Journal.log("launch_rhc2ros_bridge.py",
            "launch_rosbag",
            "Didn't receive any termination req within timeout! Will terminate anyway",
            LogType.WARN)
    
    Journal.log("launch_rhc2ros_bridge.py",
            "launch_rosbag",
            f"terminating rosbag recording. Dump base-path is: {dump_path}",
            LogType.INFO)

    term_trigger.close()

    # os.killpg(os.getpgid(proc.pid), signal.SIGINT)  # Send SIGINT to the whole pro
    # proc.send_signal(signal.SIGINT)  # Gracefully interrupt bag collection
    try: 
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except:
        pass
    proc.join()

    Journal.log("launch_rhc2ros_bridge.py",
            "launch_rosbag",
            f"successfully terminated rosbag recording process",
            LogType.INFO)


if __name__ == '__main__':

    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--cores', nargs='+', type=int, help='List of CPU cores to set affinity to')
    parser.add_argument('--dt', type=float, default=0.01, help='Update interval in seconds, default is 0.01')
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, default is False')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Enable verbose mode, default is True')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')
    parser.add_argument('--with_agent_refs', action='store_true', help='also forward agent refs to rhcviz')
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

    # Use the provided robot name and update interval
    update_dt = args.dt
    debug = args.debug
    verbose = args.verbose
    dump_rosbag=args.dump_rosbag
    stime_trgt=args.stime_trgt
    if stime_trgt is None and dump_rosbag:
        # set a default stime trgt, otherwise the bag file could become of gigantic size
        stime_trgt=60.0
    shared_drop_dir=None
    shared_drop_dir_val=[""]
    if args.use_shared_drop_dir:
        shared_drop_dir=StringTensorClient(basename="SharedTrainingDropDir", 
                        name_space=args.ns,
                        verbose=True, 
                        vlevel=VLevel.V2)
        shared_drop_dir.run()
        shared_drop_dir_val=[""]*shared_drop_dir.length()
        while not self.shared_datanames.read_vec(shared_drop_dir_val, 0):
            ns=1000000000
            PerfSleep.thread_sleep(ns)
            continue
    
    bridge = None
    if not args.ros2:

        from lrhc_control.utils.rhc_viz.rhc2viz import RhcToVizBridge

        bridge = RhcToVizBridge(namespace=args.ns, 
                        verbose=verbose,
                        rhcviz_basename="RHCViz", 
                        robot_selector=[0, None],
                        with_agent_refs=args.with_agent_refs,
                        rhc_refs_in_h_frame=args.rhc_refs_in_h_frame,
                        agent_refs_in_h_frame=args.agent_refs_in_h_frame)
    else:

        from lrhc_control.utils.rhc_viz.rhc2viz2 import RhcToViz2Bridge

        bridge = RhcToViz2Bridge(namespace=args.ns, 
                        verbose=verbose,
                        rhcviz_basename="RHCViz", 
                        robot_selector=[0, None],
                        with_agent_refs=args.with_agent_refs,
                        rhc_refs_in_h_frame=args.rhc_refs_in_h_frame,
                        agent_refs_in_h_frame=args.agent_refs_in_h_frame,
                        env_idx=args.env_idx,
                        sim_time_trgt=stime_trgt,
                        srdf_homing_file_path=args.srdf_path)

    # spawn a process to record bag if required
    bag_proc=None
    term_trigger=None
    if dump_rosbag:

        term_trigger=RemoteTriggererSrvr(namespace=args.ns+f"SharedTerminator",
                                            verbose=verbose,
                                            vlevel=VLevel.V1,
                                            force_reconnection=True)
        term_trigger.run()

        import multiprocess as mp
        ctx = mp.get_context('forkserver')
        bag_proc=ctx.Process(target=launch_rosbag, 
                            name="rosbag_recorder_"+f"{args.ns}",
                            args=(args.ns,args.dump_path,stime_trgt))
        bag_proc.start()
        
    bridge.run(update_dt=update_dt)

    if bag_proc is not None:
        term_trigger.trigger()

        bag_proc.join()
    
    if term_trigger is not None:
        term_trigger.close()

