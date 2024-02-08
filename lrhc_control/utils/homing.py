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
import numpy as np

import xml.etree.ElementTree as ET

from typing import List

from control_cluster_bridge.utilities.defs import Journal

class RobotHomer:

    def __init__(self, 
            srdf_path: str, 
            jnt_names_prb: List[str] = None):

        self.journal = Journal()

        self.srdf_path = srdf_path

        self.jnt_names_prb = jnt_names_prb
        
        # open srdf and parse the homing field
        
        with open(srdf_path, 'r') as file:
            
            self._srdf_content = file.read()

        try:
            self._srdf_root = ET.fromstring(self._srdf_content)

        except ET.ParseError as e:
        
            print(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + ": could not read SRDF properly!!")

        # Find all the 'joint' elements within 'group_state' with the name attribute and their values
        joints = self._srdf_root.findall(".//group_state[@name='home']/joint")

        self._homing_map = {}

        self.jnt_names_srdf = []
        self.homing_srdf = []
        for joint in joints:
            joint_name = joint.attrib['name']
            joint_value = joint.attrib['value']
            self.jnt_names_srdf.append(joint_name)
            self.homing_srdf.append(joint_value)

            self._homing_map[joint_name] =  float(joint_value)

        if self.jnt_names_prb is None:
            
            # we use the same joints in the SRDF

            self.jnt_names_prb = self.jnt_names_srdf

        self.jnt_names_prb = self._filter_jnt_names(self.jnt_names_prb)
        self.n_dofs = len(self.jnt_names_prb)

        self.joint_idx_map = {}
        for joint in range(0, self.n_dofs):

            self.joint_idx_map[self.jnt_names_prb[joint]] = joint 

        self._homing = np.full((1, self.n_dofs), 
                        0.0, 
                        dtype=np.float32) # homing configuration
        
        self._assign2homing()

    def _assign2homing(self):
        
        for joint in self.jnt_names_srdf:
            
            if joint in self.jnt_names_prb:
                
                self._homing[:, self.joint_idx_map[joint]] = self._homing_map[joint],
            
            else:

                self._homing[:, self.joint_idx_map[joint]] = 0.0
                                                            
    def get_homing(self):

        return self._homing.flatten()
    
    def get_homing_map(self):

        return self._homing_map
    
    def _filter_jnt_names(self, 
                        names: List[str]):

        to_be_removed = ["universe", 
                        "reference", 
                        "world", 
                        "floating", 
                        "floating_base"]
        
        for name in to_be_removed:

            if name in names:
                names.remove(name)

        return names
    