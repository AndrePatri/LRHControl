from aug_mpc.utils.sys_utils import PathsGetter
from aug_mpc.envs.lrhc_training_env_base import LRhcTrainingEnvBase

from mpc_hive.utilities.shared_data.rhc_data import RobotState, RhcStatus
from mpc_hive.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

import torch

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType

import os
from aug_mpc.utils.episodic_data import EpisodicData
from aug_mpc.utils.signal_smoother import ExponentialSignalSmoother
import math

from aug_mpc.utils.math_utils import check_capsize
from aug_mpc.envs.variable_flights_baseline import VariableFlightsBaseline

from typing import Dict

class FakePosEnvVariableFlights(VariableFlightsBaseline):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000,
            env_opts: Dict = {}):

        self._add_env_opt(env_opts, "max_distance", default=5.0) # [m] 
        self._add_env_opt(env_opts, "min_distance", default=0.0) # [m]
        self._add_env_opt(env_opts, "max_vref", default=1.0) # [m/s]
        self._add_env_opt(env_opts, "max_dp", default=5.0) # [m] after this, v ref saturates
        self._add_env_opt(env_opts, "max_dt", default=env_opts["max_dp"]/ env_opts["max_vref"])

        VariableFlightsBaseline.__init__(self, 
            namespace=namespace,
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms,
            env_opts=env_opts)

    def get_file_paths(self):
        paths=VariableFlightsBaseline.get_file_paths(self)
        paths.append(os.path.abspath(__file__))        
        return paths
    
    def _custom_post_init(self):
        VariableFlightsBaseline._custom_post_init(self)
        
        # position targets to be reached (wrt robot's pos at ep start)
        self._p_trgt_w=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2].detach().clone()
        self._p_delta_w=self._p_trgt_w.detach().clone()
        self._dp_norm=torch.zeros((self._n_envs, 1),dtype=self._dtype,device=self._device)
        self._dp_versor=self._p_trgt_w.detach().clone()

        self._trgt_d=torch.zeros((self._n_envs, 1),dtype=self._dtype,device=self._device)
        self._trgt_theta=torch.zeros((self._n_envs, 1),dtype=self._dtype,device=self._device)

    def _update_loc_twist_refs(self):
        # this is called at each env substep
        
        self._compute_twist_ref_w()
    
        if not self._override_agent_refs:
            agent_p_ref_current=self._agent_refs.rob_refs.root_state.get(data_type="p",
            gpu=self._use_gpu)
            agent_p_ref_current[:, 0:2]=self._p_trgt_w

        # then convert it to base ref local for the agent
        robot_q = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        # rotate agent ref from world to robot base
        world2base_frame(t_w=self._agent_twist_ref_current_w, q_b=robot_q, 
            t_out=self._agent_twist_ref_current_base_loc)
        # write it to agent refs tensors
        self._agent_refs.rob_refs.root_state.set(data_type="twist", data=self._agent_twist_ref_current_base_loc,
                                            gpu=self._use_gpu)

    def _compute_twist_ref_w(self, env_indxs: torch.Tensor = None):
        
        # angular refs are not altered
        if env_indxs is None:
            # we update the position error using the current base position
            self._p_delta_w[:, :]=self._p_trgt_w-\
                self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2]
                
            self._dp_norm[:, :]=self._p_delta_w.norm(dim=1,keepdim=True)+1e-6
            self._dp_versor[:, :]=self._p_delta_w/self._dp_norm

            # apply for vref saturation
            to_be_saturated=self._dp_norm[:, :]>self._env_opts["max_dp"]
            self._dp_norm[to_be_saturated.flatten(), :]=self._env_opts["max_dp"]

            # we compute the twist refs for the agent depending of the position error
            self._agent_twist_ref_current_w[:, 0:2]=self._dp_norm*self._dp_versor/self._env_opts["max_dt"]
            self._agent_twist_ref_current_w[:, 2:3]=0 # no vertical vel

            # apply pof0 using last value of bernoully coeffs
            self._agent_twist_ref_current_w[:, 0:3] = self._agent_twist_ref_current_w[:, 0:3]*self._bernoulli_coeffs_linvel # linvel
            self._agent_twist_ref_current_w[:, 3:6] = self._agent_twist_ref_current_w[:, 3:6]*self._bernoulli_coeffs_omega # omega
        else:
            self._p_delta_w[env_indxs, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[env_indxs, 0:2] -\
                self._p_trgt_w[env_indxs, :]
            
            # apply for vref saturation
            to_be_saturated=torch.logical_and((self._dp_norm[:, :]>self._env_opts["max_dp"]).flatten(),env_indxs)
            self._dp_norm[to_be_saturated.flatten(), :]=self._env_opts["max_dp"]

            self._dp_norm[env_indxs, :]=self._p_delta_w[env_indxs, :].norm(dim=1,keepdim=True)+1e-6
            self._dp_versor[env_indxs, :]=self._p_delta_w[env_indxs, :]/self._dp_norm[env_indxs, :]

            self._agent_twist_ref_current_w[env_indxs, 0:2]=self._dp_norm[env_indxs, :]*self._dp_versor[env_indxs, :]/self._env_opts["max_dt"]        
            self._agent_twist_ref_current_w[env_indxs, 2:3]=0 # no vertical vel

            # apply pof0 using last value of bernoully coeffs
            self._agent_twist_ref_current_w[env_indxs, 0:3] = self._agent_twist_ref_current_w[env_indxs, 0:3]*self._bernoulli_coeffs_linvel[env_indxs, :]
            self._agent_twist_ref_current_w[env_indxs, 3:6] = self._agent_twist_ref_current_w[env_indxs, 3:6]*self._bernoulli_coeffs_omega[env_indxs, :] # omega

    def _override_refs(self,
            env_indxs: torch.Tensor = None):
        
        # runs at every post_step
        self._agent_refs.rob_refs.root_state.synch_all(read=True,retry=True) # first read from mem
        if self._use_gpu:
            # copies latest refs to GPU 
            self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=False,non_blocking=False) 

        agent_p_ref_current=self._agent_refs.rob_refs.root_state.get(data_type="p",
                gpu=self._use_gpu)
        
        agent_yaw_omega_ref_current=self._agent_refs.rob_refs.root_state.get(data_type="omega",
                gpu=self._use_gpu)
        
        # self._p_trgt_w[:, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2] + \
        #     agent_p_ref_current[:, 0:2]
        self._p_trgt_w[:, :]=agent_p_ref_current[:, 0:2] # set p target target from shared mem

        self._agent_twist_ref_current_w[:, 5:6]=agent_yaw_omega_ref_current[:, 2:3] # set yaw ang. vel target from shared mem
    
    def _debug_agent_refs(self):
        if self._use_gpu:
            self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=False)
        self._agent_refs.rob_refs.root_state.synch_all(read=False, retry = True)

    def _randomize_task_refs(self,
        env_indxs: torch.Tensor = None):

        # we randomize the target position/omega in world frame
        if env_indxs is None:
            self._trgt_d.uniform_(self._env_opts["min_distance"], self._env_opts["max_distance"])
            self._trgt_theta.uniform_(0.0, 2*torch.pi)

            self._p_trgt_w[:, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2] +\
                torch.cat((self._trgt_d*torch.cos(self._trgt_theta)
                           ,self._trgt_d*torch.sin(self._trgt_theta)), dim=1)
            
            # randomize just omega
            random_uniform=torch.full_like(self._agent_twist_ref_current_w[:, 3:6], fill_value=0.0)
            torch.nn.init.uniform_(random_uniform, a=-1, b=1)
            self._agent_twist_ref_current_w[:, 3:6] = random_uniform*self._twist_ref_scale[:, 3:6] + self._twist_ref_offset[:, 3:6]

            # sample for all envs pof0
            if self._env_opts["use_pof0"]: # sample from bernoulli distribution and update coefficients
                torch.bernoulli(input=self._pof1_b_linvel,out=self._bernoulli_coeffs_linvel) # by default bernoulli_coeffs are 1 if not self._env_opts["use_pof0"]
                torch.bernoulli(input=self._pof1_b_omega,out=self._bernoulli_coeffs_omega)
                           
        else:

            if env_indxs.any():
                integer_idxs=torch.nonzero(env_indxs).flatten()
                
                trgt_d_selected=self._trgt_d[integer_idxs, :]
                trgt_d_selected.uniform_(self._env_opts["min_distance"], self._env_opts["max_distance"])
                self._trgt_d[integer_idxs, :]=trgt_d_selected

                trgt_theta_selected=self._trgt_theta[integer_idxs, :]
                trgt_theta_selected.uniform_(0.0, 2*torch.pi)
                self._trgt_theta[integer_idxs, :]=trgt_theta_selected

                self._p_trgt_w[integer_idxs, 0:1]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[integer_idxs, 0:1] +\
                    self._trgt_d[integer_idxs, :]*torch.cos(self._trgt_theta[integer_idxs, :])
                self._p_trgt_w[integer_idxs, 1:2]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[integer_idxs, 1:2] +\
                    self._trgt_d[integer_idxs, :]*torch.sin(self._trgt_theta[integer_idxs, :])

                # randomize just omega
                random_uniform=torch.full_like(self._agent_twist_ref_current_w[env_indxs, 3:6], fill_value=0.0)
                torch.nn.init.uniform_(random_uniform, a=-1, b=1)
                self._agent_twist_ref_current_w[env_indxs, 3:6] = random_uniform*self._twist_ref_scale[:, 3:6] + self._twist_ref_offset[:, 3:6]

                # sample for all envs pof0, then reset to 1 for envs which are not to be randomized
                if self._env_opts["use_pof0"]: # sample from bernoulli distribution and update coefficients
                    torch.bernoulli(input=self._pof1_b_linvel,out=self._bernoulli_coeffs_linvel) # by default bernoulli_coeffs are 1 if not self._env_opts["use_pof0"]
                    torch.bernoulli(input=self._pof1_b_omega,out=self._bernoulli_coeffs_omega)
                    self._bernoulli_coeffs_linvel[~env_indxs, :]=1
                    self._bernoulli_coeffs_omega[~env_indxs, :]=1

        self._compute_twist_ref_w(env_indxs=env_indxs) # update linear vel twist refs based on pos error

