from pynput import keyboard

from lrhc_control.utils.shared_data.agent_refs import AgentRefs

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType
from SharsorIPCpp.PySharsorIPC import dtype

import math

import numpy as np

class AgentRefsFromKeyboard:

    def __init__(self, 
                namespace: str, 
                verbose = False):

        self._verbose = verbose

        self.namespace = namespace

        self._closed = False
        
        self.enable_navigation = False

        self.dxy = 0.05 # [m]
        self._dtwist = 1.0 * math.pi / 180.0 # [rad]

        self.agent_refs = None

        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._twist_null = None

        self._init_shared_data()

    def _init_shared_data(self):

        self.env_index = SharedTWrapper(namespace = self.namespace,
                basename = "EnvSelector",
                is_server = False, 
                verbose = True, 
                vlevel = VLevel.V2,
                safe = False,
                dtype=dtype.Int)
        
        self.env_index.run()
        
        self._init_rhc_ref_subscriber()

    def _init_rhc_ref_subscriber(self):

        self.agent_refs = AgentRefs(namespace=self.namespace,
                                is_server=False, 
                                safe=False, 
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                with_gpu_mirror=False)

        self.agent_refs.run()

        self._twist_null = self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np)
        self._twist_null.zero_()

    def __del__(self):

        if not self._closed:

            self._close()
    
    def _close(self):
        
        if self.agent_refs is not None:

            self.agent_refs.close()

        self._closed = True
    
    def _synch(self, 
            read = True):
        
        if read:

            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self.cluster_idx = env_index[0, 0].item()
            self.cluster_idx_np = self.cluster_idx
        
            self.agent_refs.rob_refs.synch_from_shared_mem()
        
        else:
            
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self.agent_refs.rob_refs.root_state.n_cols,
                                                read=False)
    
    def _update_navigation(self, 
                    type: str = "",
                    increment = True,
                    reset: bool = False):

        current_lin_v_ref = self.agent_refs.rob_refs.root_state.get(data_type="v", robot_idxs=self.cluster_idx_np)
        current_omega_ref = self.agent_refs.rob_refs.root_state.get(data_type="omega", robot_idxs=self.cluster_idx_np)

        if not reset:
            if type=="lateral_lin" and increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] - self.dxy
            if type=="lateral_lin" and not increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] + self.dxy
            if type=="frontal_lin" and not increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] - self.dxy
            if type=="frontal_lin" and increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] + self.dxy
            if type=="twist_roll" and increment:
                # rotate counter-clockwise
                current_omega_ref[0] = current_omega_ref[0] + self._dtwist 
            if type=="twist_roll" and not increment:
                current_omega_ref[0] = current_omega_ref[0] - self._dtwist 
            if type=="twist_pitch" and increment:
                # rotate counter-clockwise
                current_omega_ref[1] = current_omega_ref[1] + self._dtwist 
            if type=="twist_pitch" and not increment:
                current_omega_ref[1] = current_omega_ref[1] - self._dtwist 
            if type=="twist_yaw" and increment:
                # rotate counter-clockwise
                current_omega_ref[2] = current_omega_ref[2] + self._dtwist 
            if type=="twist_yaw" and not increment:
                current_omega_ref[2] = current_omega_ref[2] - self._dtwist 
        else:
            current_omega_ref[:] = 0
            current_lin_v_ref[:] = 0

        self.agent_refs.rob_refs.root_state.set(data_type="v",data=current_lin_v_ref,
                                    robot_idxs=self.cluster_idx_np)
        self.agent_refs.rob_refs.root_state.set(data_type="omega",data=current_omega_ref,
                                    robot_idxs=self.cluster_idx_np)

    def _set_navigation(self,
                key):
        if key.char == "n":
            self.enable_navigation = not self.enable_navigation
            info = f"High level navigation enabled: {self.enable_navigation}"
            Journal.log(self.__class__.__name__,
                "_set_navigation",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_navigation:
            self._update_navigation(reset = True)
            
        if key.char == "6" and self.enable_navigation:
            self._update_navigation(type="lateral_lin", 
                            increment = True)
        if key.char == "4" and self.enable_navigation:
            self._update_navigation(type="lateral_lin",
                            increment = False)
        if key.char == "8" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = True)
        if key.char == "2" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = False)
                
    def _on_press(self, key):
            
        self._synch(read=True) # updates  data like
        # current cluster index

        if hasattr(key, 'char'):
            self._set_navigation(key)

        self._synch(read=False)

    def _on_release(self, key):
            
        if hasattr(key, 'char'):
            
            # nullify vel ref
            self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._twist_null,
                                robot_idxs=self.cluster_idx_np)

            if key == keyboard.Key.esc:

                self._close()

        self._synch(read=False)

    def run(self):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True)
    
        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

if __name__ == "__main__":  

    keyb_cmds = AgentRefsFromKeyboard(namespace="kyon0", 
                            verbose=True)

    keyb_cmds.run()