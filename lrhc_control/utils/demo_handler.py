from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.keyboard_cmds import AgentRefsFromKeyboard

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState

from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import dtype

import math
import numpy as np
import time
import matplotlib.pyplot as plt

from typing import Dict

class DemoRunner(AgentRefsFromKeyboard):
    def __init__(self, 
        namespace: str, 
        verbose = False,
        agent_refs_world: bool = False,
        env_idx: int = None,
        opts: Dict):

        self._demo_opts = opts

        self._n_waypoints = 4
        self._edge_length = 4.0  # [m]
        self._edge_max_v_norm = 0.5  # [m/s]
        
        if "n_waypoints" in self._demo_opts:
            self._n_waypoints = self._demo_opts["n_waypoints"]
        if "edge_length" in self._demo_opts:
            self._edge_length = self._demo_opts["edge_length"]
        if "edge_max_v_norm" in self._demo_opts:
            self._edge_max_v_norm = self._demo_opts["edge_max_v_norm"]

        self._edge_dt = self._edge_length / self._edge_max_v_norm
        self._waypoints = np.zeros((2, self._n_waypoints))

        # super().__init__(namespace=namespace,
        #     verbose=verbose,
        #     agent_refs_world=agent_refs_world,
        #     env_idx=env_idx)
    
    def _write_to_shared_mem(self):
        self.enable_navigation = False
        self._set_waypoint()
        super()._write_to_shared_mem()

    def _compute_waypoints(self):
        radius = (self._edge_length / (2 * math.sin(math.pi / self._n_waypoints)))
        angles = np.linspace(0, 2 * math.pi, self._n_waypoints, endpoint=False)
        self._waypoints[0, :] = radius * np.cos(angles)
        self._waypoints[1, :] = radius * np.sin(angles)
    
    def visualize_waypoints(self):
        plt.figure(figsize=(6, 6))
        plt.plot(self._waypoints[0, :], self._waypoints[1, :], 'bo-', label='Waypoints')
        plt.plot([self._waypoints[0, -1], self._waypoints[0, 0]], 
                 [self._waypoints[1, -1], self._waypoints[1, 0]], 'bo-')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Waypoint Visualization')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

    def _set_waypoint(self):
        if self.enable_pos:
            elapsed = time.monotonic() - self._time_now
            if elapsed >= self._edge_dt:
                self._current_pos_ref[0:2] = self._starting_pos + self._waypoints[:, self._idx].reshape(-1)
                self._idx += 1
                self._time_now = time.monotonic()
        else:
            self._idx = 0
            self._update_starting_pos()

    def _on_press(self, key):
        super()._on_press()

    def _on_release(self, key):
        super()._on_release()

    def _update_starting_pos(self):
        self._robot_state.root_state.synch_all(read=True, retry=True)  # Read robot state        
        robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
        self._starting_pos = robot_p[0:2] 

    def run(self, read_from_stdin: bool = False, release_timeout: float = 0.1):
        self._idx = 0
        self._update_starting_pos()
        self._compute_waypoints()  # Compute waypoint trajectory
        # self._time_now = time.monotonic()

        # super().run(read_from_stdin=read_from_stdin,
        #     release_timeout=release_timeout)

if __name__ == "__main__":  

    keyb_cmds = DemoRunner(namespace="kyon0")

    keyb_cmds.run()

    keyb_cmds.visualize_waypoints()