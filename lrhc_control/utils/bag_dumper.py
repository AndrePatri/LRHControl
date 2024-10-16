
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
            remap_ns: str = None,
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
            agent_refs_in_h_frame:bool=False,
            use_static_idx: bool = True):

        self._closed=False
        
        self._ns=ns
        self._remap_ns=remap_ns
        if self._remap_ns is None: # allow to publish with different namespace (to allow
            # support for multiple bags at once and multiple rhcviz instances)
            self._remap_ns=self._ns

        self._srdf_path=None

        self._with_agent_refs=with_agent_refs
        self._rhc_refs_in_h_frame=rhc_refs_in_h_frame
        self._agent_refs_in_h_frame=agent_refs_in_h_frame

        self._env_idx=env_idx
        self._use_static_idx=use_static_idx

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
        
        self._abort_wallmin=abort_wallmin

        self._initialize()

    def __del__(self):
        self.close()

    def _initialize(self):
        
        import multiprocess as mp
        ctx = mp.get_context('forkserver')
        self._bag_proc=ctx.Process(target=self._launch_rosbag, 
            name="rosbag_recorder_"+f"{self._ns}",
            args=(self._remap_ns,self._dump_path,self._timeout_ms,self._use_shared_drop_dir))
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
                remap_ns=self._remap_ns,
                verbose=self._verbose,
                rhcviz_basename="RHCViz", 
                robot_selector=[0, None],
                with_agent_refs=self._with_agent_refs,
                rhc_refs_in_h_frame=self._rhc_refs_in_h_frame,
                agent_refs_in_h_frame=self._agent_refs_in_h_frame,
                env_idx=self._env_idx,
                sim_time_trgt=self._bag_sdt,
                srdf_homing_file_path=self._srdf_path,
                abort_wallmin=self._abort_wallmin,
                use_static_idx=self._use_static_idx)

        # actual process recording bag
        from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererSrvr
        self._term_trigger=RemoteTriggererSrvr(namespace=self._remap_ns+f"SharedTerminator",
                                            verbose=self._verbose,
                                            vlevel=VLevel.V1,
                                            force_reconnection=True)
    
    def init(self):

        self._term_trigger.run()
        self._bridge.init(update_dt=self._ros_bridge_dt)
    
    def step(self):
        return self._bridge.step()

    def training_done(self):
        
        return self._shared_info.get().flatten()[self._is_done_idx]>0.5

    def _launch_rosbag(self, 
            namespace: str, dump_path: str, timeout_sec:float, use_shared_drop_dir: bool = True):
        
        import os

        Journal.log(self.__class__.__name__,
            "launch_rosbag",
            f"launch_rosbag PID is {os.getpid()}",
            LogType.INFO)

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

        shared_drop_dir.close()
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
        proc = subprocess.Popen(command, shell=shell,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid
            # preexec_fn=os.setsid # crucial -> all childs will have the same ID
            )
        # Set the process group ID to the subprocess PID
        # os.setpgid(proc.pid, proc.pid)

        Journal.log(self.__class__.__name__,
                "launch_rosbag",
                f"launching rosbag recording with PID {proc.pid}",
                LogType.INFO)

        timeout_ms = int(timeout_sec*1e3)
        if not term_trigger.wait(timeout_ms):
            Journal.log(self.__class__.__name__,
                "launch_rosbag",
                "Didn't receive any termination req within timeout! Will terminate anyway",
                LogType.WARN)
        
        Journal.log(self.__class__.__name__,
                "launch_rosbag",
                f"terminating rosbag recording. Dump base-path is: {dump_path}",
                LogType.INFO)

        term_trigger.close()

        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        # os.kill(proc.pid, signal.SIGINT)
        # proc.send_signal(signal.SIGINT)
            
        try:
            proc.wait(timeout=1.0)
        except:
            proc.kill()

        Journal.log(self.__class__.__name__,
                "launch_rosbag",
                f"successfully terminated rosbag recording process",
                LogType.INFO)

    def close(self):
        if not self._closed:
            if self._bag_proc is not None:
                self._term_trigger.trigger() # triggering process termination and joining
                ret=self._bag_proc.join(5) # waits some time 
                if ret is not None:
                    if self._bag_proc.exitcode is None: # process not terminated yet
                        Journal.log(self.__class__.__name__,
                            "close",
                            f"forcibly terminating bag process",
                            LogType.WARN)
                        self._bag_proc.terminate()
            if self._term_trigger is not None:
                self._term_trigger.close()
            
            self._bridge.close()
            self._closed=True
