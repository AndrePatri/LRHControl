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

        if injection_node is None:
            self._injection_node = round(self.task_interface.prb.getNNodes()/2.0)
        else:
            self._injection_node = injection_node

        for contact_name, phase_name in contact_map.items():
            self._contact_timelines[contact_name] = self.phase_manager.getTimelines()[phase_name]

        # self.zmp_timeline = self.phase_manager.getTimelines()['zmp_timeline']

    def reset(self):
        # self.phase_manager.clear()
        self.task_interface.reset()

    def set_step(self, contact_flags: List[bool]):
        for flag_contact, contact_name in zip(contact_flags, self._contact_timelines.keys()):
            phase_i = self._contact_timelines[contact_name]
            if flag_contact == True:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}_short'), pos=self._injection_node)
            else:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'flight_{contact_name}'), pos=self._injection_node)

    def trot_jumped(self):

        #  diagonal 1 duration 4
        self._contact_timelines['ball_2'].addPhase(self._contact_timelines['ball_2'].getRegisteredPhase(f'flight_ball_2'))
        self._contact_timelines['ball_3'].addPhase(self._contact_timelines['ball_3'].getRegisteredPhase(f'flight_ball_3'))

        # diagonal 2 short stance 1 (3 times)
        self._contact_timelines['ball_1'].addPhase(self._contact_timelines['ball_1'].getRegisteredPhase(f'stance_ball_1_short'))
        self._contact_timelines['ball_1'].addPhase(self._contact_timelines['ball_1'].getRegisteredPhase(f'stance_ball_1_short'))
        self._contact_timelines['ball_1'].addPhase(self._contact_timelines['ball_1'].getRegisteredPhase(f'stance_ball_1_short'))
        self._contact_timelines['ball_4'].addPhase(self._contact_timelines['ball_4'].getRegisteredPhase(f'stance_ball_4_short'))
        self._contact_timelines['ball_4'].addPhase(self._contact_timelines['ball_4'].getRegisteredPhase(f'stance_ball_4_short'))
        self._contact_timelines['ball_4'].addPhase(self._contact_timelines['ball_4'].getRegisteredPhase(f'stance_ball_4_short'))

        #  diagonal 2 duration 4
        self._contact_timelines['ball_1'].addPhase(self._contact_timelines['ball_1'].getRegisteredPhase(f'flight_ball_1'))
        self._contact_timelines['ball_4'].addPhase(self._contact_timelines['ball_4'].getRegisteredPhase(f'flight_ball_4'))

        # diagonal 1 short stance 1 (3 times)
        self._contact_timelines['ball_2'].addPhase(self._contact_timelines['ball_2'].getRegisteredPhase(f'stance_ball_2_short'))
        self._contact_timelines['ball_2'].addPhase(self._contact_timelines['ball_2'].getRegisteredPhase(f'stance_ball_2_short'))
        self._contact_timelines['ball_2'].addPhase(self._contact_timelines['ball_2'].getRegisteredPhase(f'stance_ball_2_short'))
        self._contact_timelines['ball_3'].addPhase(self._contact_timelines['ball_3'].getRegisteredPhase(f'stance_ball_3_short'))
        self._contact_timelines['ball_3'].addPhase(self._contact_timelines['ball_3'].getRegisteredPhase(f'stance_ball_3_short'))
        self._contact_timelines['ball_3'].addPhase(self._contact_timelines['ball_3'].getRegisteredPhase(f'stance_ball_3_short'))


        # self._contact_timelines['ball_1'].addPhase(self._contact_timelines['ball_1'].getRegisteredPhase(f'stance_ball_1'))
        # self._contact_timelines['ball_2'].addPhase(self._contact_timelines['ball_2'].getRegisteredPhase(f'flight_ball_2'))
        # self._contact_timelines['ball_3'].addPhase(self._contact_timelines['ball_3'].getRegisteredPhase(f'stance_ball_3'))
        # self._contact_timelines['ball_4'].addPhase(self._contact_timelines['ball_4'].getRegisteredPhase(f'flight_ball_4'))

    def trot(self):
        cycle_list_1 = [0, 1, 1, 0]
        cycle_list_2 = [1, 0, 0, 1]
        self.cycle(cycle_list_1)
        self.cycle(cycle_list_2)
        # self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_empty_phase'))
        # self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_empty_phase'))

    def crawl(self):
        cycle_list_1 = [0, 1, 1, 1]
        cycle_list_2 = [1, 1, 1, 0]
        cycle_list_3 = [1, 0, 1, 1]
        cycle_list_4 = [1, 1, 0, 1]
        self.cycle(cycle_list_1)
        self.cycle(cycle_list_2)
        self.cycle(cycle_list_3)
        self.cycle(cycle_list_4)

    def leap(self):
        cycle_list_1 = [0, 0, 1, 1]
        cycle_list_2 = [1, 1, 0, 0]
        self.cycle(cycle_list_1)
        self.cycle(cycle_list_2)

    def walk(self):
        cycle_list_1 = [1, 0, 1, 0]
        cycle_list_2 = [0, 1, 0, 1]
        self.cycle(cycle_list_1)
        self.cycle(cycle_list_2)

    def jump(self):
        cycle_list = [0, 0, 0, 0]
        self.cycle(cycle_list)

    def wheelie(self):
        cycle_list = [0, 0, 1, 1]
        self.cycle(cycle_list)

    def stand(self):
        cycle_list = [1, 1, 1, 1]
        self.cycle(cycle_list)
        # self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_phase'))



