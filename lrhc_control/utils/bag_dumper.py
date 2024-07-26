
import signal
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype

class RosBagDumper():

    def __init__(self,
            ns:str,
            ros_bridge_dt: float,
            bag_sdt: float,
            debug: bool=False,
            verbose: bool=False,
            dump_path: str = "/tmp",
            use_shared_drop_dir:bool=True,
            ros2:bool=True,
            env_idx:int=0,
            srdf_path:str=None,
            abort_wallmin:float=5.0,
            with_agent_refs:bool=True,
            rhc_refs_in_h_frame:bool=True,
            agent_refs_in_h_frame:bool=False):

        self._closed=False


        self._ns=ns

        self._srdf_path=None

        self._with_agent_refs=with_agent_refs
        self._rhc_refs_in_h_frame=rhc_refs_in_h_frame
        self._agent_refs_in_h_frame=agent_refs_in_h_frame

        self._env_idx=env_idx

        self._timeout_ms=240000
        self._dump_path=dump_path
        self._verbose=verbose
        self._debug=debug
        self._ros_bridge_dt=ros_bridge_dt
        self._bag_sdt=bag_sdt

        self._ros2=ros2

        self._shared_info=None
        self._use_shared_drop_dir=use_shared_drop_dir

        self._bridge = None

        # spawn a process to record bag if required
        self._bag_proc=None
        self._term_trigger=None

        self._is_done_idx=None
        self._training_done=False
        
        self._abort_wallmin=abort_wallmin

        self._initialize()

    def __del__(self):
        self.close()

    def _initialize(self):
        
        import multiprocess as mp
        ctx = mp.get_context('forkserver')
        self._bag_proc=ctx.Process(target=self._launch_rosbag, 
            name="rosbag_recorder_"+f"{self._ns}",
            args=(self._ns,self._dump_path,self._timeout_ms,self._use_shared_drop_dir))
        self._bag_proc.start()

        # for detecting when training is finished
        from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo

        self._shared_info=SharedRLAlgorithmInfo(is_server=False,
                    namespace=self._ns, 
                    verbose=True, 
                    vlevel=VLevel.V2)
        self._shared_info.run()
        shared_info_names=self._shared_info.dynamic_info.get()
        self._is_done_idx=shared_info_names.index("is_done")

        # bridge from rhc shared data to ROS
        if not self._ros2:
            # DEPRECATED
            from lrhc_control.utils.rhc_viz.rhc2viz import RhcToVizBridge
            self._bridge = RhcToVizBridge(namespace=self._ns, 
                            verbose=self._verbose,
                            rhcviz_basename="RHCViz", 
                            robot_selector=[0, None],
                            with_agent_refs=self._with_agent_refs,
                            rhc_refs_in_h_frame=self._rhc_refs_in_h_frame,
                            agent_refs_in_h_frame=self._agent_refs_in_h_frame)
        else:
            from lrhc_control.utils.rhc_viz.rhc2viz2 import RhcToViz2Bridge
            self._bridge = RhcToViz2Bridge(namespace=self._ns, 
                verbose=self._verbose,
                rhcviz_basename="RHCViz", 
                robot_selector=[0, None],
                with_agent_refs=self._with_agent_refs,
                rhc_refs_in_h_frame=self._rhc_refs_in_h_frame,
                agent_refs_in_h_frame=self._agent_refs_in_h_frame,
                env_idx=self._env_idx,
                sim_time_trgt=self._bag_sdt,
                srdf_homing_file_path=self._srdf_path,
                abort_wallmin=self._abort_wallmin)

        # actual process recording bag
        from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererSrvr
        self._term_trigger=RemoteTriggererSrvr(namespace=self._ns+f"SharedTerminator",
                                            verbose=self._verbose,
                                            vlevel=VLevel.V1,
                                            force_reconnection=True)
    
    def run(self):

        self._term_trigger.run()

        self._bridge.run(update_dt=self._ros_bridge_dt)
        self._training_done = self._shared_info.get().flatten()[self._is_done_idx]>0.5
                
    def training_done(self):
        
        self._training_done = self._shared_info.get().flatten()[self._is_done_idx]>0.5
        return self._training_done

    def _launch_rosbag(self, 
            namespace: str, dump_path: str, timeout_sec:float, use_shared_drop_dir: bool = True):
        
        import multiprocess as mp
        import os

        # using a shared drop dir if enabled
        from SharsorIPCpp.PySharsorIPC import StringTensorClient
        from perf_sleep.pyperfsleep import PerfSleep

        if use_shared_drop_dir:
            shared_drop_dir=StringTensorClient(basename="SharedTrainingDropDir", 
                            name_space=self._ns,
                            verbose=True, 
                            vlevel=VLevel.V2)
            shared_drop_dir.run()
            shared_drop_dir_val=[""]*shared_drop_dir.length()
            while not shared_drop_dir.read_vec(shared_drop_dir_val, 0):
                ns=1000000000
                PerfSleep.thread_sleep(ns)
                continue
            dump_path=shared_drop_dir_val[0] # overwrite

        retry_kill=20
        additional_secs=5.0
        from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererClnt
        term_trigger=RemoteTriggererClnt(namespace=namespace+f"SharedTerminator",
                                verbose=True,
                                vlevel=VLevel.V1)
        term_trigger.run()
        this_dir_path=os.path.dirname(__file__)

        shell=False
        if not shell:
            command = [f"{this_dir_path}/launch_rosbag.sh", "--ns", namespace, "--output_path", dump_path]
        else:
            command = f"{this_dir_path}/launch_rosbag.sh --ns {namespace} --output_path {dump_path}"

        import subprocess
        proc = subprocess.Popen(command, shell=False, 
            preexec_fn=os.setsid # crucial -> all childs will have the same ID
            )
        # Set the process group ID to the subprocess PID
        # os.setpgid(proc.pid, proc.pid)

        Journal.log("launch_rhc2ros_bridge.py",
                "launch_rosbag",
                f"launching rosbag recording with PID {proc.pid}",
                LogType.INFO)

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

        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        # proc.send_signal(signal.SIGINT)
            
        try:
            proc.wait(timeout=1.0)
        except:
            proc.kill()

        Journal.log("launch_rhc2ros_bridge.py",
                "launch_rosbag",
                f"successfully terminated rosbag recording process",
                LogType.INFO)

    def close(self):
        if not self._closed:
            if self._bag_proc is not None:
                self._term_trigger.trigger()
                self._bag_proc.join()
            if self._term_trigger is not None:
                self._term_trigger.close()
            self._closed=True
