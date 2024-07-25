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

class BagDumper():

    def __init__(self,
            ns:str,
            dt: float,
            debug: bool=False,
            verbose: bool=False,
            dump_path: str = "/tmp",
            stime_trgt: float=60.0,
            use_shared_drop_dir:bool=True,
            ros2:bool=True,
            env_idx:int=0,
            srdf_path:str=None):

        self._ns=ns

        self._srdf_path=None

        self._with_agent_refs=True
        self._rhc_refs_in_h_frame=True
        self._agent_refs_in_h_frame=False

        self._env_idx=env_idx

        timeout_ms=240000
        self._dump_path=dump_path
        self._verbose=verbose
        self._debug=debug
        self._dt=dt
        self._stime_trgt=stime_trgt
        self._ros2=ros2

        self._shared_drop_dir=None
        self._shared_drop_dir_val=[""]
        if use_shared_drop_dir:
            self._shared_drop_dir=StringTensorClient(basename="SharedTrainingDropDir", 
                            name_space=args.ns,
                            verbose=True, 
                            vlevel=VLevel.V2)
            self._shared_drop_dir.run()
            self._shared_drop_dir_val=[""]*self._shared_drop_dir.length()
            while not self._shared_drop_dir.read_vec(self._shared_drop_dir_val, 0):
                ns=1000000000
                PerfSleep.thread_sleep(ns)
                continue
            self._dump_path=self._shared_drop_dir_val[0]
        
        self._bridge = None
        if not self._ros2:
            from lrhc_control.utils.rhc_viz.rhc2viz import RhcToVizBridge
            bridge = RhcToVizBridge(namespace=self._ns, 
                            verbose=self._verbose,
                            rhcviz_basename="RHCViz", 
                            robot_selector=[0, None],
                            with_agent_refs=self._with_agent_refs,
                            rhc_refs_in_h_frame=self._rhc_refs_in_h_frame,
                            agent_refs_in_h_frame=self._agent_refs_in_h_frame)
        else:

            from lrhc_control.utils.rhc_viz.rhc2viz2 import RhcToViz2Bridge

            bridge = RhcToViz2Bridge(namespace=self._ns, 
                            verbose=self._verbose,
                            rhcviz_basename="RHCViz", 
                            robot_selector=[0, None],
                            with_agent_refs=self._with_agent_refs,
                            rhc_refs_in_h_frame=self._rhc_refs_in_h_frame,
                            agent_refs_in_h_frame=self._agent_refs_in_h_frame
                            env_idx=self.._env_idx,
                            sim_time_trgt=self._stime_trgt,
                            srdf_homing_file_path=self._srdf_path)

        # spawn a process to record bag if required
        self._bag_proc=None
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
                                args=(args.ns,dump_path,timeout_ms))
            bag_proc.start()
        
        bridge.run(update_dt=update_dt)

        if bag_proc is not None:
            term_trigger.trigger()
            bag_proc.join()
        
        if term_trigger is not None:
            term_trigger.close()

        if shared_drop_dir is not None:
            shared_drop_dir.close()

    def launch_rosbag(namespace: str, dump_path: str, timeout_sec:float):
        
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

        timeout_ms = int(timeout_sec*1e3)
        if not term_trigger.wait(timeout_ms):
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

