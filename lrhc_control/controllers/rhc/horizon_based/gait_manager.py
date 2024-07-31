import numpy as np

from horizon.rhc.taskInterface import TaskInterface

from phase_manager import pymanager

from typing import List

class GaitManager:

    def __init__(self, 
            task_interface: TaskInterface, 
            phase_manager: pymanager.PhaseManager, 
            contact_map,
            injection_node: int = None):

        self.task_interface = task_interface
        self.phase_manager = phase_manager

        self._contact_timelines = dict()
        self._f_reg_timelines = dict()

        if injection_node is None:
            self._injection_node = round(self.task_interface.prb.getNNodes()/2.0)
        else:
            self._injection_node = injection_node

        self._timeline_names = []
        self._freg_timeline_names = []
        for contact_name, timeline_name in contact_map.items():
            self._contact_timelines[contact_name] = self.phase_manager.getTimelines()[timeline_name]
            self._f_reg_timelines[contact_name] = None
            try:
                self._f_reg_timelines[contact_name] = self.phase_manager.getTimelines()[timeline_name+"_f_reg"]
            except:
                pass
            
            self._timeline_names.append(contact_name)
            self._freg_timeline_names.append(contact_name)

        # self.zmp_timeline = self.phase_manager.getTimelines()['zmp_timeline']

    def reset(self):
        # self.phase_manager.clear()
        self.task_interface.reset()

    def add_stand(self, timeline_name):
        timeline = self._contact_timelines[timeline_name]
        timeline.addPhase(timeline.getRegisteredPhase(f'stance_{timeline_name}_short'))
        # f reg
        freg_timeline = self._f_reg_timelines[timeline_name]
        if freg_timeline is not None:
            freg_timeline.addPhase(freg_timeline.getRegisteredPhase(f'freg_{timeline_name}_short'))
    
    def add_flight(self, timeline_name):
        timeline = self._contact_timelines[timeline_name]
        timeline.addPhase(timeline.getRegisteredPhase(f'flight_{timeline_name}'), 
            pos=self._injection_node, absolute_position=True)
        # f reg
        freg_timeline = self._f_reg_timelines[timeline_name]
        if freg_timeline is not None:
            freg_timeline.addPhase(freg_timeline.getRegisteredPhase(f'freg_{timeline_name}_empty'),
                pos=self._injection_node, absolute_position=True)
