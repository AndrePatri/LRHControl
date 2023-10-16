# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of LRhcExamples and distributed under the General Public License version 2 license.
# 
# LRhcExamples is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# LRhcExamples is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with LRhcExamples.  If not, see <http://www.gnu.org/licenses/>.
# 
from control_cluster_bridge.controllers.rhc import RHController

from lrhc_examples.utils.homing import RobotHomer

from lrhc_examples.controllers.rhc.rhc_taskref import RhcTaskRef

import numpy as np

import torch

class RHC(RHController):

    def __init__(self, 
            controller_index: int,
            cluster_size: int, # needed by shared mem manager
            srdf_path: str,
            robot_name: str = "aliengo",
            verbose = False, 
            debug = False, 
            array_dtype = torch.float32):

        self._homer: RobotHomer = None
        
        self.robot_name = robot_name

        super().__init__(controller_index = controller_index, 
                        cluster_size = cluster_size,
                        srdf_path = srdf_path,
                        namespace = self.robot_name,
                        verbose = verbose, 
                        debug = debug,
                        array_dtype = array_dtype)
        
        self.add_data_lenght = 2 
    
    def _init_problem(self):
        
        print(f"[{self.__class__.__name__}" + str(self.controller_index) + "]" + \
              f"[{self.journal.status}]" + ": initializing controller...")

        # any initializations of the controllers should be performed here
        
        self._homer = RobotHomer(srdf_path=self.srdf_path, 
                                jnt_names_prb=None)
        
        self._assign_server_side_jnt_names(self._get_robot_jnt_names())

        self.n_dofs = self._get_ndofs() # after loading the URDF and creating the controller we
        # know n_dofs -> we assign it (by default = None)

        self.n_contacts = 4

        print(f"[{self.__class__.__name__}" + str(self.controller_index) + "]" +  f"[{self.journal.status}]" + "controller initialized.")

    def _init_rhc_task_cmds(self) -> RhcTaskRef:

        return RhcTaskRef(n_contacts=self.n_contacts, 
                    index=self.controller_index, 
                    q_remapping=self._quat_remap, 
                    dtype=self.array_dtype, 
                    verbose=self._verbose, 
                    namespace=self.robot_name)
    
    def _get_robot_jnt_names(self):

        return self._homer.jnt_names_prb
    
    def _get_ndofs(self):
        
        return len(self._homer.jnt_names_prb)

    def _get_cmd_jnt_q_from_sol(self):
    
        return torch.tensor(self._homer.get_homing()).reshape(1, 
                            self.robot_cmds.jnt_cmd.q.shape[1])
    
    def _get_cmd_jnt_v_from_sol(self):

        return torch.zeros((1, self.robot_cmds.jnt_cmd.v.shape[1]), 
                        dtype=self.array_dtype)

    def _get_cmd_jnt_eff_from_sol(self):
        
        return torch.zeros((1, self.robot_cmds.jnt_cmd.eff.shape[1]), 
                        dtype=self.array_dtype)
    
    def _get_additional_slvr_info(self):

        return torch.tensor([2.3e-4, 
                            100], 
                        dtype=self.array_dtype)
    
    def _update_open_loop(self):

        a = 1
    
    def _update_closed_loop(self):

        a = 2

    def _solve(self):
        
        # self._update_open_loop() # updates the TO ig and 
        # # initial conditions using data from the solution

        self._update_closed_loop() # updates the TO ig and 
        # # initial conditions using robot measurements
        
        self.perf_timer.clock_sleep(int(0.02 * 1e9)) # nanoseconds, simulated processing time
