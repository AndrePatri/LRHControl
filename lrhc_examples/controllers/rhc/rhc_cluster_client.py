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
from control_cluster_bridge.cluster_client.control_cluster_client import ControlClusterClient
from typing import List

class RHClusterClient(ControlClusterClient):

    def __init__(self, 
            cluster_size, 
            control_dt,
            cluster_dt,
            jnt_names,
            device, 
            np_array_dtype, 
            verbose, 
            debug,
            n_contact_sensors: int = -1,
            contact_linknames: List[str] = None, 
            robot_name = "aliengo"):

        self.robot_name = robot_name
                
        super().__init__(cluster_size= cluster_size, 
                        control_dt=control_dt, 
                        cluster_dt=cluster_dt, 
                        jnt_names=jnt_names,
                        n_contact_sensors = n_contact_sensors,
                        contact_linknames = contact_linknames,
                        device=device, 
                        np_array_dtype=np_array_dtype, 
                        verbose=verbose, 
                        debug=debug, 
                        namespace=self.robot_name)
    
    pass