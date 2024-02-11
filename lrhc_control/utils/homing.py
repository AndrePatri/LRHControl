import numpy as np

import xml.etree.ElementTree as ET

from typing import List

from SharsorIPCpp.PySharsorIPC import Journal, LogType

class RobotHomer:

    def __init__(self, 
            srdf_path: str, 
            jnt_names_prb: List[str]):

        self.srdf_path = srdf_path

        self.jnt_names_prb = self._filter_jnt_names(jnt_names_prb)
        self.n_dofs = len(self.jnt_names_prb)

        self.joint_idx_map = {}
        for joint in range(0, self.n_dofs):

            self.joint_idx_map[self.jnt_names_prb[joint]] = joint 

        self._homing = np.full((1, self.n_dofs), 
                        0.0, 
                        dtype=np.float32) # homing configuration
        
        # open srdf and parse the homing field
        
        with open(srdf_path, 'r') as file:
            
            self._srdf_content = file.read()

        try:
            self._srdf_root = ET.fromstring(self._srdf_content)
            # Now 'root' holds the root element of the XML tree.
            # You can navigate through the XML tree to extract the tags and their values.
            # Example: To find all elements with a specific tag, you can use:
            # elements = root.findall('.//your_tag_name')

            # Example: If you know the specific structure of your .SRDF file, you can extract
            # the data accordingly, for instance:
            # for child in root:
            #     if child.tag == 'some_tag_name':
            #         tag_value = child.text
            #         # Do something with the tag value.
            #     elif child.tag == 'another_tag_name':
            #         # Handle another tag.

        except ET.ParseError as e:
            
            exception = "Could not read SRDF properly!!"

            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

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
    