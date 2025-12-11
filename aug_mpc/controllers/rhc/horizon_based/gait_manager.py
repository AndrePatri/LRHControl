import numpy as np

from aug_mpc.controllers.rhc.horizon_based.horizon_imports import *

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType

from typing import Dict

class GaitManager:

    def __init__(self, 
            task_interface: TaskInterface, 
            phase_manager: pymanager.PhaseManager, 
            injection_node: int = None,
            keep_yaw_vert: bool = False,
            yaw_vertical_weight: float = None,
            vertical_landing: bool = False,
            landing_vert_weight: float = None,
            phase_force_reg: float = None,
            flight_duration: int = 15,
            post_flight_stance: int = 3,
            step_height: float = 0.1,
            dh: float = 0.0,
            custom_opts: Dict = {}):

        self._custom_opts=custom_opts

        self._is_open_loop=False
        if "is_open_loop" in self._custom_opts:
            self._is_open_loop=self._custom_opts["is_open_loop"]

        self.task_interface = task_interface
        self._n_nodes_prb=self.task_interface.prb.getNNodes()
        
        self._phase_manager = phase_manager
        self._model=self.task_interface.model
        self._q0=self._model.q0
        self._kin_dyn=self.task_interface.model.kd
        
        # phase weights and regs
        self._keep_yaw_vert=keep_yaw_vert
        self._yaw_vertical_weight=yaw_vertical_weight
        self._vertical_landing=vertical_landing
        self._landing_vert_weight=landing_vert_weight
        self._phase_force_reg=phase_force_reg
        self._total_weight = np.atleast_2d(np.array([0, 0, self._kin_dyn.mass() * 9.81])).T 
        
        self._f_reg_ref={}

        # flight parameters
        self._post_flight_stance=post_flight_stance
        self._flight_info_now=None 
        self._flight_info_size=5 # current pos, current length, nominal length, nom. apex, no. landing height 
        # duration bounds/defaults
        self._flight_duration_max=self._n_nodes_prb-(injection_node+1)
        self._flight_duration_min=3
        self._flight_duration_default=flight_duration 
        # apex bounds/defaults
        self._step_height_default=step_height
        self._step_height_min=0.0
        self._step_height_max=0.5
        # end height bounds/defaults
        self._dh_default=dh
        self._dh_min=0.0
        self._dh_max=0.5

        # timeline data
        self._contact_timelines = dict()
        self.timeline_names=[]

        self._flight_phases = {}
        self._touchdown_phases = {}
        self._contact_phases = {}
        self._fk_contacts = {}
        self._fkd_contacts = {}
        self._f_reg_ref = {}

        # reference traj
        self._tg = trajectoryGenerator.TrajectoryGenerator()
        self._traj_der= [None, 0, 0]
        self._traj_second_der=[None, 0, 0]
        self._third_traj_der=[None, None, 0]

        self._ref_trjs = {}
        self._ref_vtrjs = {}

        if injection_node is None:
            self._injection_node = round(self.task_interface.prb.getNNodes()/2.0)
        else:
            self._injection_node = injection_node
        
        self._init_contact_timelines()  
        
        self._reset_contact_timelines()

    def _init_contact_timelines(self):
        short_stance_duration=1
        flight_phase_short_duration=1

        self.n_contacts=len(self._model.cmap.keys())

        # current pos [c0, c1, ....], current length, nominal length, nom. apex, no. landing height 
        self._flight_info_now=np.zeros(shape=(5*self.n_contacts)) 

        self._name_to_idx_map={}

        j=0
        for contact in self._model.cmap.keys():
            
            self._fk_contacts[contact]=self._kin_dyn.fk(contact)
            self._fkd_contacts[contact]=self._kin_dyn.frameVelocity(contact, self._model.kd_frame)
            self.timeline_names.append(contact)
            self._contact_timelines[contact]=self._phase_manager.createTimeline(f'{contact}_timeline')
            # stances
            self._contact_phases[contact] = self._contact_timelines[contact].createPhase(short_stance_duration, 
                                    f'stance_{contact}_short')
                
            if self.task_interface.getTask(f'{contact}') is not None:
                self._contact_phases[contact].addItem(self.task_interface.getTask(f'{contact}'))
            else:
                Journal.log(self.__class__.__name__,
                    "_init_contact_timelines",
                    f"contact task {contact} not found",
                    LogType.EXCEP,
                    throw_when_excep=True)
            i=0
            self._f_reg_ref[contact]=[]
            for force in self._model.cmap[contact]:
                f_ref=self.task_interface.prb.createParameter(name=f"{contact}_force_reg_f{i}_ref",
                    dim=3)
                force_reg=self.task_interface.prb.createResidual(f'{contact}_force_reg_f{i}', self._phase_force_reg * (force - f_ref), 
                    nodes=[])
                self._f_reg_ref[contact].append(f_ref)
                self.set_f_reg(contact_name=contact, scale=4)                
                self._contact_phases[contact].addCost(force_reg, nodes=list(range(0, short_stance_duration)))
                i+=1
            
            # flights
            self._flight_phases[contact]=self._contact_timelines[contact].createPhase(flight_phase_short_duration, 
                                    f'flight_{contact}_short')

            # ref traj for contacts
            self._ref_trjs[contact]=np.zeros(shape=[7, self.task_interface.prb.getNNodes()]) # allocate traj
            # of max length eual to number of nodes

            # sanity checks
            self._zpos_task_found=True
            self._zvel_task_found=True
            if self.task_interface.getTask(f'z_{contact}') is None:
                self._zpos_task_found=False
            if self.task_interface.getTask(f'vz_{contact}') is None:
                self._zvel_task_found=False
            if (not self._zpos_task_found) and (not self._zvel_task_found):
                Journal.log(self.__class__.__name__,
                    "_init_contact_timelines",
                    f"neither pos or vel task for contacts were found! Aborting.",
                    LogType.EXCEP,
                    throw_when_excep=True)
            if (not self._zpos_task_found) and self._is_open_loop:
                Journal.log(self.__class__.__name__,
                    "_init_contact_timelines",
                    f"Running in open loop, but no contact pos task found. Aborting.",
                    LogType.EXCEP,
                    throw_when_excep=True)
            # if (not self._zvel_task_found) and (not self._is_open_loop):
            #     Journal.log(self.__class__.__name__,
            #         "_init_contact_timelines",
            #         f"Running in closed loop, but contact vel task not found. Aborting",
            #         LogType.EXCEP,
            #         throw_when_excep=True)
            
            self._ref_trjs[contact]=None
            self._ref_vtrjs[contact]=None
            if self._zpos_task_found: # we use pos trajectory
                self._ref_trjs[contact]=np.zeros(shape=[7, self.task_interface.prb.getNNodes()])
                init_z_foot = self._fk_contacts[contact](q=self._q0)['ee_pos'].elements()[2]
                if self._is_open_loop:
                    self._ref_trjs[contact][2, :] = np.atleast_2d(init_z_foot)
                else:
                    self._ref_trjs[contact][2, :] = 0.0 # place foot at ground level initially ()
                    
                self._flight_phases[contact].addItemReference(self.task_interface.getTask(f'z_{contact}'), 
                    self._ref_trjs[contact][2, 0:1], 
                    nodes=list(range(0, flight_phase_short_duration)))
                self._touchdown_phases[contact]=None
            else: # foot traj in velocity
                # ref vel traj
                self._ref_vtrjs[contact]=np.zeros(shape=[7, self.task_interface.prb.getNNodes()]) # allocate traj
                # of max length eual to number of nodes
                self._ref_vtrjs[contact][2, :] = np.atleast_2d(0)
                self._flight_phases[contact].addItemReference(self.task_interface.getTask(f'vz_{contact}'), 
                    self._ref_vtrjs[contact][2, 0:1], 
                    nodes=list(range(0, flight_phase_short_duration)))
                self._touchdown_phases[contact]=self._contact_timelines[contact].createPhase(flight_phase_short_duration, 
                                    f'touchdown_{contact}_short')
                # self._touchdown_phases[contact].addItemReference(self.task_interface.getTask(f'vz_{contact}'), 
                #     self._ref_vtrjs[contact][2, 0:1], 
                #     nodes=list(range(0, flight_phase_short_duration)))
            
            # ee_vel=self._fkd_contacts[contact](q=self._model.q, 
            #             qdot=self._model.v)['ee_vel_linear']
            # cstr = self.task_interface.prb.createConstraint(f'{contact}_vert', ee_vel[0:2], [])
            # self._flight_phases[contact].addConstraint(cstr, nodes=[0, flight_phase_short_duration-1])
            if self._keep_yaw_vert:
                # keep ankle vertical
                c_ori = self._model.kd.fk(contact)(q=self._model.q)['ee_rot'][2, :]
                cost_ori = self.task_interface.prb.createResidual(f'{contact}_ori', self._yaw_vertical_weight * (c_ori.T - np.array([0, 0, 1])))
                # flight_phase.addCost(cost_ori, nodes=list(range(0, flight_duration+post_landing_stance)))
            
            if self._vertical_landing and self._touchdown_phases[contact] is not None:
                v_xy=self._fkd_contacts[contact](q=self._model.q, qdot=self._model.v)['ee_vel_linear'][0:2]
                vertical_landing=self.task_interface.prb.createResidual(f'{contact}_only_vert_v', 
                    self._landing_vert_weight * v_xy, 
                    nodes=[])
                self._touchdown_phases[contact].addCost(vertical_landing, nodes=list(range(0, short_stance_duration)))

            self._name_to_idx_map[contact]=j

            j+=1

        self.update()

    def _reset_contact_timelines(self):

        for contact in self._model.cmap.keys():
            
            idx=self._name_to_idx_map[contact]
            # we follow same order as in shm for more efficient writing 
            self._flight_info_now[idx]= -1.0 # pos [nodes]
            self._flight_info_now[idx+1*self.n_contacts]= -1.0 # duration (remaining) [nodes]
            self._flight_info_now[idx+2*self.n_contacts]=self._flight_duration_default # [nodes]
            self._flight_info_now[idx+3*self.n_contacts]=self._step_height_default
            self._flight_info_now[idx+4*self.n_contacts]=self._dh_default

            # fill timeline with stances
            contact_timeline=self._contact_timelines[contact]
            contact_timeline.clear() # remove phases
            short_stance_phase = contact_timeline.getRegisteredPhase(f'stance_{contact}_short')
            while contact_timeline.getEmptyNodes() > 0:
                contact_timeline.addPhase(short_stance_phase)   
            
            self.update()
            
    def reset(self):
        # self.phase_manager.clear()
        self.task_interface.reset()
        self._reset_contact_timelines()
    
    def set_f_reg(self, 
        contact_name,
        scale: float = 4.0):
        f_refs=self._f_reg_ref[contact_name]
        for force in f_refs:
            ref=self._total_weight/(scale*len(f_refs))
            force.assign(ref)
    
    def set_flight_duration(self, contact_name, val: float):
        self._flight_info_now[self._name_to_idx_map[contact_name]+2*self.n_contacts]=val
    
    def get_flight_duration(self, contact_name):
        return self._flight_info_now[self._name_to_idx_map[contact_name]+2*self.n_contacts]
    
    def set_step_apexdh(self, contact_name, val: float):
        self._flight_info_now[self._name_to_idx_map[contact_name]+3*self.n_contacts]=val
    
    def get_step_apexdh(self, contact_name):
        return self._flight_info_now[self._name_to_idx_map[contact_name]+3*self.n_contacts]
    
    def set_step_enddh(self, contact_name, val: float):
        self._flight_info_now[self._name_to_idx_map[contact_name]+4*self.n_contacts]=val
    
    def get_step_enddh(self, contact_name):
        return self._flight_info_now[self._name_to_idx_map[contact_name]+4*self.n_contacts]
    
    def add_stand(self, contact_name):
        # always add stand at the end of the horizon
        timeline = self._contact_timelines[contact_name]
        if timeline.getEmptyNodes() > 0:
            timeline.addPhase(timeline.getRegisteredPhase(f'stance_{contact_name}_short'))
    
    def add_flight(self, contact_name,
        robot_q: np.ndarray):

        timeline = self._contact_timelines[contact_name]

        flights_on_horizon=self._contact_timelines[contact_name].getPhaseIdx(self._flight_phases[contact_name]) 
        
        last_flight_idx=self._injection_node-1 # default to make things work
        if not len(flights_on_horizon)==0: # some flight phases are there
            last_flight_idx=flights_on_horizon[-1]+self._post_flight_stance

        if last_flight_idx<self._injection_node: # allow injecting
            
            flight_duration_req=int(self._flight_info_now[self._name_to_idx_map[contact_name]+2*self.n_contacts])
            flight_apex_req=self._flight_info_now[self._name_to_idx_map[contact_name]+3*self.n_contacts]
            flight_enddh_req=self._flight_info_now[self._name_to_idx_map[contact_name]+4*self.n_contacts]
            
            if not flight_duration_req>1:
                Journal.log(self.__class__.__name__,
                    "add_flight",
                    f"Got flight duration {flight_duration_req} < 1!",
                    LogType.WARN,
                    throw_when_excep=True)

            # process requests to ensure flight params are valid
            # duration
            if flight_duration_req<self._flight_duration_min:
                flight_duration_req=self._flight_duration_min
            if flight_duration_req>self._flight_duration_max:
                flight_duration_req=self._flight_duration_max
            # apex height
            if flight_apex_req<self._step_height_min:
                flight_apex_req=self._step_height_min
            if flight_apex_req>self._step_height_max:
                flight_apex_req=self._step_height_max
            # landing height
            if flight_enddh_req<self._dh_min:
                flight_enddh_req=self._dh_min
            if flight_enddh_req>self._dh_max:
                flight_enddh_req=self._dh_max

            if self._ref_vtrjs[contact_name] is not None and \
                self._ref_trjs[contact_name] is not None: # only allow one mode (pos/velocity traj)
                Journal.log(self.__class__.__name__,
                    "add_flight",
                    f"Both pos and vel traj for contact {contact_name} are not None! This is not allowed, aborting.",
                    LogType.EXCEP,
                    throw_when_excep=True)
                
            # inject pos traj if pos mode
            if self._ref_trjs[contact_name] is not None:
                # recompute trajectory online (just needed if using pos traj)
                starting_pos=self._fk_contacts[contact_name](q=robot_q)['ee_pos'].elements()[2] # compute foot traj
                # starting_pos=0.0
                self._ref_trjs[contact_name][2, 0:flight_duration_req]=np.atleast_2d(self._tg.from_derivatives(
                    flight_duration_req, 
                    p_start=starting_pos, 
                    p_goal=starting_pos+flight_enddh_req, 
                    clearance=flight_apex_req,
                    derivatives=self._traj_der,
                    second_der=self._traj_second_der,
                    third_der=self._third_traj_der)
                    )
                for i in range(flight_duration_req):
                    res, phase_token=timeline.addPhase(self._flight_phases[contact_name], 
                        pos=self._injection_node+i, 
                        absolute_position=True)
                    phase_token.setItemReference(f'z_{contact_name}',
                        self._ref_trjs[contact_name][:, i])
                if self._touchdown_phases[contact_name] is not None:
                    # add touchdown phase for forcing vertical landing
                    res, phase_token=timeline.addPhase(self._touchdown_phases[contact_name], 
                            pos=self._injection_node+flight_duration_req, 
                            absolute_position=True)                
                    
            # inject vel traj if vel mode
            if self._ref_vtrjs[contact_name] is not None:
                self._ref_vtrjs[contact_name][2, 0:flight_duration_req]=np.atleast_2d(self._tg.derivative_of_trajectory(
                    flight_duration_req, 
                    p_start=0.0, 
                    p_goal=flight_enddh_req, 
                    clearance=flight_apex_req,
                    derivatives=self._traj_der,
                    second_der=self._traj_second_der,
                    third_der=self._third_traj_der))
                for i in range(flight_duration_req):
                    res, phase_token=timeline.addPhase(self._flight_phases[contact_name], 
                        pos=self._injection_node+i, 
                        absolute_position=True)
                    phase_token.setItemReference(f'vz_{contact_name}',
                        self._ref_vtrjs[contact_name][2:3, i:i+1])
                if self._touchdown_phases[contact_name] is not None:
                    # add touchdown phase for forcing vertical landing
                    res, phase_token=timeline.addPhase(self._touchdown_phases[contact_name], 
                            pos=self._injection_node+self._flight_durations[contact_name], 
                            absolute_position=True)       

        if timeline.getEmptyNodes() > 0: # fill empty nodes at the end of the horizon, if any, with stance
            timeline.addPhase(timeline.getRegisteredPhase(f'stance_{contact_name}_short'))
    
    def update(self):
        self._phase_manager.update()
        
    def update_flight_info(self, timeline_name):

        # get current position and remaining duration of flight phases over the horizon for a single contact

        # phase indexes over timeline
        phase_idxs=self._contact_timelines[timeline_name].getPhaseIdx(self._flight_phases[timeline_name])
        
        if not len(phase_idxs)==0: # at least one flight phase on horizon -> read info from timeline

            # all active phases on timeline
            active_phases=self._contact_timelines[timeline_name].getActivePhases()

            phase_idx_start=phase_idxs[0]
            # active_nodes_start=active_phases[phase_idx_start].getActiveNodes()
            pos_start=active_phases[phase_idx_start].getPosition()
            # n_nodes_start=active_phases[phase_idx_start].getNNodes()

            phase_idx_last=phase_idxs[-1] # just get info for last phase on the horizon
            # active_nodes_last=active_phases[phase_idx_last].getActiveNodes()
            pos_last=active_phases[phase_idx_last].getPosition()
            # n_nodes_last=active_phases[phase_idx_last].getNNodes()
            
            # write to info
            self._flight_info_now[self._name_to_idx_map[timeline_name]+0*self.n_contacts]=pos_last
            self._flight_info_now[self._name_to_idx_map[timeline_name]+1*self.n_contacts]=pos_last - pos_start

            return True
        
        return False
    
    def get_flight_info(self, timeline_name):
        return (self._flight_info_now[self._name_to_idx_map[timeline_name]+0*self.n_contacts], 
            self._flight_info_now[self._name_to_idx_map[timeline_name]+1*self.n_contacts],
            self._flight_info_now[self._name_to_idx_map[timeline_name]+2*self.n_contacts],
            self._flight_info_now[self._name_to_idx_map[timeline_name]+3*self.n_contacts],
            self._flight_info_now[self._name_to_idx_map[timeline_name]+4*self.n_contacts])
    
    def get_flight_info_all(self):
        return self._flight_info_now
    
    def set_ref_pos(self,
        timeline_name:str,
        ref_height: np.array = None,
        threshold: float = 0.05):
        
        if ref_height is not None:
            self._ref_trjs[timeline_name][2, :]=ref_height
            if ref_height>threshold:
                self.add_flight(timeline_name=timeline_name)
                this_flight_token_idx=self._contact_timelines[timeline_name].getPhaseIdx(self._flight_phases[timeline_name])[-1]
                active_phases=self._contact_timelines[timeline_name].getActivePhases()
                active_phases[this_flight_token_idx].setItemReference(f'z_{timeline_name}',
                    self._ref_trjs[timeline_name])
            else:
                self.add_stand(timeline_name=timeline_name)

    def set_force_feedback(self,
        timeline_name: str,
        force_norm: float):
        
        flight_tokens=self._contact_timelines[timeline_name].getPhaseIdx(self._flight_phases[timeline_name])
        contact_tokens=self._contact_phases[timeline_name].getPhaseIdx(self._contact_phases[timeline_name])
        if not len(flight_tokens)==0:
            first_flight=flight_tokens[0]
            first_flight
    
    def check_horizon_full(self,
        timeline_name):
        timeline = self._contact_timelines[timeline_name]
        if timeline.getEmptyNodes() > 0:
            error = f"Empty nodes detected over the horizon for timeline {timeline}! Make sure to fill the whole horizon with valid phases!!"
            Journal.log(self.__class__.__name__,
                "check_horizon_full",
                error,
                LogType.EXCEP,
                throw_when_excep = True)