from aug_mpc.utils.shared_data.agent_refs import AgentRefs

from mpc_hive.utilities.shared_data.rhc_data import RobotState
from mpc_hive.utilities.math_utils import world2base_frame_twist

from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import dtype

import math

import numpy as np

class AgentRefsFromJoy:

    def __init__(self, 
                namespace: str, 
                verbose = False,
                agent_refs_world: bool = True,
                env_idx: int = None):

        self._env_idx=env_idx

        self._verbose = verbose

        self._agent_refs_world=agent_refs_world
        
        self.namespace = namespace

        self._closed = False
        
        self.enable_linvel = False
        self.enable_omega = False
        self.enable_omega_roll = False
        self.enable_omega_pitch = False
        self.enable_omega_yaw = False

        self.enable_pos = False

        self.dpos = 0.1 # [m]
        self.dxy = 0.05 # [m/s]
        self.dvxyz = 0.05 # [m/s]
        self.dheading=0.05
        self._dtwist = 1.0 * math.pi / 180.0 # [rad]

        self._v_magnitude=0.0
        self._heading_lat=0.0
        self._heading_frontal=0.0
        self._heading=0.0
        self.agent_refs = None

        self._max_vxy_magn=1.5 # [m/s]
        self._max_vz_magn=0.0
        self._max_pitch_rate=0.0 # [rad/s]
        self._max_roll_rate=0.0 # [rad/s]
        self._max_yaw_rate=0.8 # [rad/s]

        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._twist_null = None

        self._init_shared_data()

    def _init_shared_data(self):
        
        self.env_index=None
        if self._env_idx is None:
            self.env_index = SharedTWrapper(namespace = self.namespace,
                    basename = "EnvSelector",
                    is_server = False, 
                    verbose = True, 
                    vlevel = VLevel.V2,
                    safe = False,
                    dtype=dtype.Int)
            
            self.env_index.run()
        
        self._init_rhc_ref_subscriber()
        
        self._current_twist_ref_world = np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np), 
                fill_value=0.0).reshape(-1)
        self._current_twist_ref_base=np.full_like(self._current_twist_ref_world, fill_value=0.0).reshape(1, -1)

        self._current_pos_ref = np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="p", robot_idxs=self.cluster_idx_np), 
                fill_value=0.0).reshape(-1)
        
        self._robot_state = RobotState(namespace=self.namespace,
                            is_server=False, 
                            safe=False,
                            verbose=True,
                            vlevel=VLevel.V2)
        self._robot_state.run()            

    def _init_rhc_ref_subscriber(self):

        self.agent_refs = AgentRefs(namespace=self.namespace,
                                is_server=False, 
                                safe=True, 
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                with_gpu_mirror=False,
                                with_torch_view=False)

        self.agent_refs.run()

        self._twist_null = self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np)
        self._twist_null[:]=0.0

        q0=np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="q", robot_idxs=self.cluster_idx_np),fill_value=0.0)
        q0[0]=1.0
        self.agent_refs.rob_refs.root_state.set(data_type="q",data=q0,
                                        robot_idxs=self.cluster_idx_np)
        
    def __del__(self):

        if not self._closed:
            self._close()
    
    def _close(self):
        
        if self.agent_refs is not None:
            self.agent_refs.close()
        if self._robot_state is not None:
            self._robot_state.close()

        self._closed = True
    
    def _synch(self, joy):
        
        if self.env_index is not None:
            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self._env_idx=env_index[0, 0].item()
        self.cluster_idx = self._env_idx
        self.cluster_idx_np = self.cluster_idx    
        
        self._set_omega(joy)    
        self._set_linvel(joy)    
        self._set_position(joy)     

    def _set_omega(self, 
                joy):
        
        if joy.face[1]:
            self.enable_omega = not self.enable_omega
            info = f"Twist change enabled: {self.enable_omega}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if not self.enable_omega:
            self._current_twist_ref_world[3:] = 0.0
               
    def _set_linvel(self,
                joy):
        if joy.face[0]:
            self.enable_linvel = not self.enable_linvel
            info = f"High level navigation enabled: {self.enable_linvel}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_linvel:
            self._current_twist_ref_world[0:3] = 0.0
        else:
            twist_ref=self._current_twist_ref_world
            self._v_magnitude=np.clip(self._v_magnitude, a_min=0.0, a_max=self._max_vxy_magn)
            twist_ref[0] = self._v_magnitude*np.cos(self._heading)
            twist_ref[1] = self._v_magnitude*np.sin(self._heading)
            twist_ref[2]=np.clip(twist_ref[2], a_min=0.0, a_max=self._max_vz_magn)
            twist_ref[3]=np.clip(twist_ref[3], a_min=0.0, a_max=self._max_roll_rate)
            twist_ref[4]=np.clip(twist_ref[4], a_min=0.0, a_max=self._max_pitch_rate)
            twist_ref[5]=np.clip(twist_ref[5], a_min=0.0, a_max=self._max_yaw_rate)
            
    def _set_position(self,joy):

        if joy.face[2]:
            self.enable_pos = not self.enable_pos
            info = f"High level pos reference change: {self.enable_pos}"
            Journal.log(self.__class__.__name__,
                "set_position",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_pos: # reset
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2]=0.0
            self._current_pos_ref[:]=robot_p

    def _write_to_shared_mem(self):

        self.agent_refs.rob_refs.root_state.synch_all(read=True)
        self._robot_state.root_state.synch_all(read = True, retry = True) # read robot state        
        
        if self.enable_pos:
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2]=0.0
            # self.agent_refs.rob_refs.root_state.set(data_type="p",data=self._current_pos_ref-robot_p,
            #                                 robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.set(data_type="p",data=self._current_pos_ref,
                                            robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                        n_rows=1, n_cols=3,
                                        read=False)
            
        if self.enable_omega or self.enable_linvel: # angular velocity or linear
            if self._agent_refs_world:
                # ref was set in world frame -> we need to move it in base frame before setting it to the agent
                robot_q = self._robot_state.root_state.get(data_type="q")[self.cluster_idx_np, :].reshape(1, -1)
                world2base_frame_twist(t_w=self._current_twist_ref_world.reshape(1, -1), 
                    q_b=robot_q, 
                    t_out=self._current_twist_ref_base)
                    
                self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._current_twist_ref_base,
                                                robot_idxs=self.cluster_idx_np)
            else:
                self._current_twist_ref_base[:, :]=self._current_twist_ref_world.reshape(1, -1)
            self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._current_twist_ref_base,
                                            robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=7, 
                                        n_rows=1, n_cols=6,
                                        read=False)
              
    def run(self, connect, topic, poll_interval ):

        info = f"Ready. Starting to listen for commands..."

        def on_message(payload):
            pass
        
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
                
        import time

        self.agent_refs.run()
        
        from mpc_hive.utilities.joy.joy_zmq_listener import JoyListenerZMQ

        joy_listener=JoyListenerZMQ(connect=connect, topic=topic, poll_interval=poll_interval, on_message=on_message)
        joy_listener.start()

        try:
            while not joy_listener.done:
                self._synch(joy_listener) # set refs
                self._write_to_shared_mem() # write refs
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("[AgentRefsFromJoy][run]: Exiting...")
        finally:
            joy_listener.stop()
