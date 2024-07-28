import torch

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcCmds
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus

from lrhc_control.utils.shared_data.remote_stepping import RemoteStepperSrvr
from lrhc_control.utils.shared_data.remote_stepping import RemoteResetSrvr
from lrhc_control.utils.shared_data.remote_stepping import RemoteResetRequest

from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from lrhc_control.utils.shared_data.training_env import SharedTrainingEnvInfo

from lrhc_control.utils.shared_data.training_env import Observations, NextObservations
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import Rewards
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Terminations
from lrhc_control.utils.shared_data.training_env import Truncations
from lrhc_control.utils.shared_data.training_env import EpisodesCounter,TaskRandCounter,SafetyRandResetsCounter

from lrhc_control.utils.episodic_rewards import EpisodicRewards
from lrhc_control.utils.episodic_data import EpisodicData
from lrhc_control.utils.episodic_data import MemBuffer

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from perf_sleep.pyperfsleep import PerfSleep

from abc import abstractmethod

import os
from typing import List

class LRhcTrainingEnvBase():

    """Base class for a remote training environment tailored to Learning-based Receding Horizon Control"""

    def __init__(self,
            namespace: str,
            obs_dim: int,
            actions_dim: int,
            episode_timeout_lb: int,
            episode_timeout_ub: int,
            n_steps_task_rand_lb: int,
            n_steps_task_rand_ub: int,
            action_repeat: int = 1,
            env_name: str = "",
            n_preinit_steps: int = 0,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            debug: bool = True,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000,
            rescale_rewards: bool= True,
            srew_drescaling: bool = True,
            srew_tsrescaling: bool = False,
            use_act_mem_bf: bool = False,
            act_membf_size: int = 3,
            use_random_safety_reset: bool = True,
            random_reset_freq: int = None,
            vec_ep_freq_metrics_db: int = 1):
        
        self._vec_ep_freq_metrics_db = vec_ep_freq_metrics_db # update single env metrics every
        # n episodes
        
        self._this_path = os.path.abspath(__file__)

        self._use_random_safety_reset = use_random_safety_reset

        self.custom_db_data = None
        
        self.custom_db_info = {}
        
        self._rescale_rewards=rescale_rewards
        self._srew_drescaling = srew_drescaling
        self._srew_tsrescaling = srew_tsrescaling
        
        self._action_repeat = action_repeat
        if self._action_repeat <=0: 
            self._action_repeat = 1
        
        self._use_act_mem_bf = use_act_mem_bf
        self._act_membf_size = act_membf_size
        
        self._closed = False

        self._override_agent_refs = override_agent_refs

        self._episode_timeout_lb = round(episode_timeout_lb/self._action_repeat) 
        self._episode_timeout_ub = round(episode_timeout_ub/self._action_repeat)

        self._n_steps_task_rand_lb = round(n_steps_task_rand_lb/self._action_repeat)
        self._n_steps_task_rand_ub = round(n_steps_task_rand_ub/self._action_repeat)
        
        self._random_rst_freq=random_reset_freq
        if self._random_rst_freq is None:
            self._random_rst_freq=self._episode_timeout_ub
        else:
            self._random_rst_freq=round(random_reset_freq/self._action_repeat)

        self._namespace = namespace
        self._with_gpu_mirror = True
        self._safe_shared_mem = False

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim

        self._use_gpu = use_gpu

        self._dtype = dtype

        self._verbose = verbose
        self._vlevel = vlevel

        self._is_debug = debug

        self._env_name = env_name

        self._robot_state = None
        self._rhc_cmds = None
        self._rhc_refs = None
        self._rhc_status = None

        self._remote_stepper = None
        self._remote_resetter = None
        self._remote_reset_req = None

        self._agent_refs = None

        self._n_envs = 0

        self._n_preinit_steps = n_preinit_steps
        
        self._ep_timeout_counter = None
        self._task_rand_counter = None
        self._rand_safety_reset_counter = None

        self._obs = None
        self._next_obs = None
        self._actions = None
        self._actions_ub = None
        self._actions_lb = None
        self._tot_rewards = None
        self._sub_rewards = None
        self._terminations = None
        self._truncations = None
        self._act_mem_buffer = None

        self._episodic_rewards_metrics = None
        
        self._timeout = timeout_ms

        self._attach_to_shared_mem()

        self._init_obs()
        self._init_actions(actions_dim)
        self._init_rewards()
        self._init_infos()
        self._init_terminations()
        self._init_truncations()
        
        self._custom_post_init()

        # self._wait_for_sim_env()

        self._init_step()

    def __del__(self):

        self.close()

    def _get_this_file_path(self):

        return self._this_path
    
    def episode_timeout_bounds(self):
        return self._episode_timeout_lb, self._episode_timeout_ub
    
    def task_rand_timeout_bounds(self):
        return self._n_steps_task_rand_lb, self._n_steps_task_rand_ub
    
    def n_action_reps(self):
        return self._action_repeat
    
    def get_file_paths(self):
        empty_list = []
        return empty_list

    def get_aux_dir(self):
        empty_list = []
        return empty_list
    
    def _init_step(self):
        
        self._check_controllers_registered(retry=True)
        self._activate_rhc_controllers()

        # just an auxiliary tensor
        initial_reset_aux = self._terminations.get_torch_mirror(gpu=self._use_gpu).clone()
        initial_reset_aux[:, :] = True # we reset all sim envs first
        self._remote_sim_step() 
        self._remote_reset(reset_mask=initial_reset_aux) 
        for i in range(self._n_preinit_steps): # perform some
            # dummy remote env stepping to make sure to have meaningful 
            # initializations (doesn't increment step counter)
            self._remote_sim_step() # 1 remote sim. step
            self._send_remote_reset_req() # fake reset request 

        self.reset()

    def _debug(self):

        if self._use_gpu:
            self._obs.synch_mirror(from_gpu=True) # copy data from gpu to cpu view
            self._next_obs.synch_mirror(from_gpu=True)
            self._actions.synch_mirror(from_gpu=True)
            self._tot_rewards.synch_mirror(from_gpu=True)
            self._sub_rewards.synch_mirror(from_gpu=True)
            self._truncations.synch_mirror(from_gpu=True) 
            self._terminations.synch_mirror(from_gpu=True)

        self._obs.synch_all(read=False, retry=True) # copies data on CPU shared mem
        self._next_obs.synch_all(read=False, retry=True)
        self._actions.synch_all(read=False, retry=True) 
        self._tot_rewards.synch_all(read=False, retry=True)
        self._sub_rewards.synch_all(read=False, retry=True)
        self._truncations.synch_all(read=False, retry = True) # writes on shared mem
        self._terminations.synch_all(read=False, retry = True) # writes on shared mem

    def _remote_sim_step(self):

        self._remote_stepper.trigger() # triggers simulation + RHC
        if not self._remote_stepper.wait_ack_from(1, self._timeout):
            Journal.log(self.__class__.__name__,
            "_remote_sim_step",
            "Remote sim. env step ack not received within timeout",
            LogType.EXCEP,
            throw_when_excep = False)
            return False
        return True

    def _remote_reset(self,
                reset_mask: torch.Tensor = None):

        reset_reqs = self._remote_reset_req.get_torch_mirror()
        if reset_mask is None: # just send the signal to allow stepping, but do not reset any of
            # the remote envs
            reset_reqs[:, :] = False
        else:
            reset_reqs[:, :] = reset_mask # remotely reset envs corresponding to
            # the mask (True--> to be reset)
        self._remote_reset_req.synch_all(read=False, retry=True) # write on shared buffer
        remote_reset_ok = self._send_remote_reset_req() # process remote request

        if reset_mask is not None:
            self._synch_obs(gpu=self._use_gpu) # if some env was reset, we use _obs
            # to hold the states, including resets, while _next_obs will always hold the 
            # state right after stepping the sim env
            # (could be a bit more efficient, since in theory we only need to read the envs
            # corresponding to the reset_mask)
            obs = self._obs.get_torch_mirror(gpu=self._use_gpu)
            self._fill_obs(obs)
            self._clamp_obs(obs)
        
        return remote_reset_ok
    
    def _send_remote_reset_req(self):

        self._remote_resetter.trigger()
        if not self._remote_resetter.wait_ack_from(1, self._timeout): # remote reset completed
            Journal.log(self.__class__.__name__,
                "_post_step",
                "Remote reset did not complete within the prescribed timeout!",
                LogType.EXCEP,
                throw_when_excep = False)
            return False
        return True

    def step(self, 
            action):

        self._pre_step()

        # set action from agent
        actions = self._actions.get_torch_mirror(gpu=self._use_gpu)
        actions[:, :] = action # writes actions
        
        if self._act_mem_buffer is not None:
            self._act_mem_buffer.update(new_data=self.get_actions(clone=True))

        self._apply_actions_to_rhc() # apply agent actions to rhc controller

        stepping_ok = True
        tot_rewards = self._tot_rewards.get_torch_mirror(gpu=self._use_gpu)
        tot_rewards[:, :] = 0 # reset total rewards before each action repeat loop
        for i in range(0, self._action_repeat): # remove env substepping
            stepping_ok = stepping_ok and self._check_controllers_registered(retry=False) # does not make sense to run training
            # if we lost some controllers
            stepping_ok = stepping_ok and self._remote_sim_step() # blocking, 
            # no sim substepping is allowed to fail
            self._synch_obs(gpu=self._use_gpu) # read obs from shared mem (done in substeps also, 
            # since substeps rewards will need updated substep obs)
            next_obs = self._next_obs.get_torch_mirror(gpu=self._use_gpu)
            obs = self._obs.get_torch_mirror(gpu=self._use_gpu)
            self._fill_obs(next_obs) # update next obs
            self._clamp_obs(next_obs) # good practice
            self._compute_sub_rewards(obs,next_obs)
            self._assemble_rewards() # includes rewards clipping
            obs[:, :] = next_obs # start from next observation, unless reset (handled in post_step())

            if not i==(self._action_repeat-1): # just sends reset signal to complete remote step sequence,
                # but does not reset any remote env
                stepping_ok = stepping_ok and self._remote_reset(reset_mask=None) 
            if not stepping_ok:
                return False
            
        stepping_ok =  stepping_ok and self._post_step() # post sub-stepping operations
        # (if action_repeat > 1, then just the db data at the last substep is logged)
        # also, if a reset of an env occurs, obs will hold the reset state

        return stepping_ok 
    
    def _post_step(self):
        
        self._ep_timeout_counter.increment() # first increment counters
        self._task_rand_counter.increment()
        if self._rand_safety_reset_counter is not None:
            self._rand_safety_reset_counter.increment()

        # check truncation and termination conditions 
        self._check_truncations() 
        self._check_terminations()
        terminated = self._terminations.get_torch_mirror(gpu=self._use_gpu)
        truncated = self._truncations.get_torch_mirror(gpu=self._use_gpu)
        if self._rand_safety_reset_counter is not None:
            lets_do_a_random_reset=self._rand_safety_reset_counter.time_limits_reached()
            truncated[:, :]=torch.logical_or(truncated,lets_do_a_random_reset.cuda()) # add random reset 
            # to truncation
        episode_finished = torch.logical_or(terminated,
                            truncated)

        if self._act_mem_buffer is not None:
            self._act_mem_buffer.reset(to_be_reset=episode_finished.flatten(),
                            init_data=self._defaut_bf_action)

        # debug step if required (IMPORTANT: must be before remote reset so that we always db
        # actual data from the step and not after reset)
        if self._is_debug:
            episode_finished_cpu = episode_finished.cpu()
            self._debug() # copies db data on shared memory
            self._update_custom_db_data(episode_finished=episode_finished_cpu)
            self._episodic_rewards_metrics.update(rewards = self._sub_rewards.get_torch_mirror(gpu=False),
                            ep_finished=episode_finished_cpu)

        # remotely reset envs only if terminated or if a timeout is reached
        
        to_be_reset=self._to_be_reset()
        rm_reset_ok = self._remote_reset(reset_mask=to_be_reset)
        
        self._custom_post_step(episode_finished=episode_finished) # any additional logic from child env  
        # here after reset, so that is can access all states post reset if necessary      

        # synchronize and reset counters for finished episodes
        self._ep_timeout_counter.reset(to_be_reset=episode_finished,randomize_limits=True)# reset and randomize duration 
        self._task_rand_counter.reset(to_be_reset=episode_finished,randomize_limits=True)# reset and randomize duration 
        # safety reset counter is only when it reches its reset interval (just to keep
        # the counter bounded)
        self._rand_safety_reset_counter.reset(to_be_reset=self._rand_safety_reset_counter.time_limits_reached())

        return rm_reset_ok
    
    def _to_be_reset(self):
        terminated = self._terminations.get_torch_mirror(gpu=self._use_gpu)
        # can be overriden by child -> defines the logic for when to reset envs
        to_be_reset=torch.logical_or(terminated.cpu(), # if terminal
            self._ep_timeout_counter.time_limits_reached() # episode timeouted
            )
        
        if self._rand_safety_reset_counter is not None:
            to_be_reset[:, :]=torch.logical_or(to_be_reset,
                self._rand_safety_reset_counter.time_limits_reached())

        return to_be_reset

    def _update_custom_db_data(self,
                    episode_finished):

        # update defaults
        self.custom_db_data["RhcRefsFlag"].update(new_data=self._rhc_refs.contact_flags.get_torch_mirror(gpu=False), 
                                    ep_finished=episode_finished) # before potentially resetting the flags, get data
        self.custom_db_data["Actions"].update(new_data=self._actions.get_torch_mirror(gpu=False), 
                                    ep_finished=episode_finished)
                
        self._get_custom_db_data(episode_finished=episode_finished)

    def reset_custom_db_data(self, keep_track: bool = False):
        # to be called periodically to reset custom db data stat. collection 
        for custom_db_data in self.custom_db_data.values():
            custom_db_data.reset(keep_track=keep_track)

    def _assemble_rewards(self):
        
        tot_rewards = self._tot_rewards.get_torch_mirror(gpu=self._use_gpu)
        sub_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)
        self._clamp_rewards(sub_rewards) # clipping rewards in a user-defined range
        
        scale=self._action_repeat
        if self._rescale_rewards and self._srew_drescaling: # scale rewards depending on the n of subrewards
            scale*=sub_rewards.shape[1] # n. dims rescaling
        if self._rescale_rewards and self._srew_tsrescaling: # scale rewards depending on the n of subrewards
            scale*=self._n_steps_task_rand_ub # scale using task rand ub (not using episode timeout since the scale
            # can then be excessively aggressive)

        # average over substeps depending on scale
        tot_rewards[:, :] = tot_rewards + torch.sum(sub_rewards, dim=1, keepdim=True)/scale

    def randomize_task_refs(self,
                env_indxs: torch.Tensor = None):
                    
        if self._override_agent_refs:
            self._override_refs(gpu=self._use_gpu)
        else:
            self._randomize_task_refs(env_indxs=env_indxs)
            
    def reset(self):
        
        self.randomize_task_refs(env_indxs=None) # randomize all refs across envs

        self._obs.reset()
        self._actions.reset()
        self._next_obs.reset()
        self._sub_rewards.reset()
        self._tot_rewards.reset()
        self._terminations.reset()
        self._truncations.reset()

        self._ep_timeout_counter.reset()
        self._task_rand_counter.reset()
        if self._rand_safety_reset_counter is not None:
            self._rand_safety_reset_counter.reset(randomize_limits=False)

        if self._act_mem_buffer is not None:
            self._act_mem_buffer.reset_all()

        self._synch_obs(gpu=self._use_gpu) # read obs from shared mem
        obs = self._obs.get_torch_mirror(gpu=self._use_gpu)
        next_obs = self._next_obs.get_torch_mirror(gpu=self._use_gpu)
        self._fill_obs(obs) # initialize observations 
        self._clamp_obs(obs) # to avoid bad things
        self._fill_obs(next_obs) # and next obs
        self._clamp_obs(next_obs)

        self.reset_custom_db_data(keep_track=False)
        self._episodic_rewards_metrics.reset(keep_track=False)

    def close(self):
        
        if not self._closed:

            # close all shared mem. clients
            self._robot_state.close()
            self._rhc_cmds.close()
            self._rhc_refs.close()
            self._rhc_status.close()
            
            self._remote_stepper.close()
            
            self._ep_timeout_counter.close()
            self._task_rand_counter.close()
            if self._rand_safety_reset_counter is not None:
                self._rand_safety_reset_counter.close()

            # closing env.-specific shared data
            self._obs.close()
            self._next_obs.close()
            self._actions.close()
            self._sub_rewards.close()
            self._tot_rewards.close()

            self._terminations.close()
            self._truncations.close()

            self._closed = True

    def get_obs(self, clone:bool=False):
        if clone:
            return self._obs.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._obs.get_torch_mirror(gpu=self._use_gpu)

    def get_next_obs(self, clone:bool=False):
        if clone:
            return self._next_obs.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._next_obs.get_torch_mirror(gpu=self._use_gpu)
        
    def get_actions(self, clone:bool=False):
        if clone:
            return self._actions.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._actions.get_torch_mirror(gpu=self._use_gpu)
            
    def get_rewards(self, clone:bool=False):
        if clone:
            return self._tot_rewards.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._tot_rewards.get_torch_mirror(gpu=self._use_gpu)
        
    def get_terminations(self, clone:bool=False):
        if clone:
            return self._terminations.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._terminations.get_torch_mirror(gpu=self._use_gpu)
    
    def get_truncations(self, clone:bool=False):
        if clone:
            return self._truncations.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._truncations.get_torch_mirror(gpu=self._use_gpu)
        
    def obs_dim(self):

        return self._obs_dim
    
    def actions_dim(self):

        return self._actions_dim
    
    def ep_rewards_metrics(self):

        return self._episodic_rewards_metrics
    
    def using_gpu(self):

        return self._use_gpu

    def name(self):

        return self._env_name

    def n_envs(self):

        return self._n_envs

    def dtype(self):
                                    
        return self._dtype 
    
    def obs_names(self):
        return self._get_obs_names()
    
    def action_names(self):
        return self._get_action_names()

    def sub_rew_names(self):
        return self._get_rewards_names()

    def _get_obs_names(self):
        # to be overridden by child class
        return None
    
    def _get_action_names(self):
        # to be overridden by child class
        return None
    
    def _get_rewards_names(self):
        # to be overridden by child class
        return None
    
    def _get_custom_db_data(self, episode_finished):
        # to be overridden by child class
        pass

    def _init_obs(self):
        
        obs_threshold_default = 10.0
        self._obs_threshold_lb = -obs_threshold_default # used for clipping observations
        self._obs_threshold_ub = obs_threshold_default

        if not self._obs_dim==len(self._get_obs_names()):
            error=f"obs dim {self._obs_dim} does not match obs names length {len(self._get_obs_names())}!!"
            Journal.log(self.__class__.__name__,
                "_init_obs",
                error,
                LogType.EXCEP,
                throw_when_excep = True)

        self._obs = Observations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            obs_dim=self._obs_dim,
                            obs_names=self._get_obs_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._next_obs = NextObservations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            obs_dim=self._obs_dim,
                            obs_names=self._get_obs_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)

        self._obs.run()
        self._next_obs.run()
        
    def _init_actions(self, actions_dim: int):
        
        device = "cuda" if self._use_gpu else "cpu"
        # action scalings to be applied to agent's output
        self._actions_ub = torch.full((1, actions_dim), dtype=self._dtype, device=device,
                                        fill_value=1.0) 
        self._actions_lb = torch.full((1, actions_dim), dtype=self._dtype, device=device,
                                        fill_value=-1.0)

        if not self._actions_dim==len(self._get_action_names()):
            error=f"action dim {self._actions_dim} does not match action names length {len(self._get_action_names())}!!"
            Journal.log(self.__class__.__name__,
                "_init_actions",
                error,
                LogType.EXCEP,
                throw_when_excep = True)
        self._actions = Actions(namespace=self._namespace,
                            n_envs=self._n_envs,
                            action_dim=self._actions_dim,
                            action_names=self._get_action_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)

        self._actions.run()

        if self._use_act_mem_bf:
            self._act_mem_buffer=MemBuffer(name="ActionMemBuf",
                data_tensor=self._actions.get_torch_mirror(),
                data_names=self._get_action_names(),
                debug=self._debug,
                horizon=self._act_membf_size,
                dtype=self._dtype,
                use_gpu=self._use_gpu)
            self._defaut_bf_action = torch.full_like(input=self.get_actions(),fill_value=0.0)

    def _init_rewards(self):
        
        reward_thresh_default = 1.0
        n_sub_rewards = len(self._get_rewards_names())
        device = "cuda" if self._use_gpu else "cpu"
        self._reward_thresh_lb = torch.full((1, n_sub_rewards), dtype=self._dtype, fill_value=-reward_thresh_default, device=device) # used for clipping rewards
        self._reward_thresh_ub = torch.full((1, n_sub_rewards), dtype=self._dtype, fill_value=reward_thresh_default, device=device) 

        self._sub_rewards = Rewards(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_rewards=n_sub_rewards,
                            reward_names=self._get_rewards_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._tot_rewards = TotRewards(namespace=self._namespace,
                            n_envs=self._n_envs,
                            reward_names=["total_reward"],
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._sub_rewards.run()
        self._tot_rewards.run()

        self._episodic_rewards_metrics = EpisodicRewards(reward_tensor=self._sub_rewards.get_torch_mirror(),
                                        reward_names=self._get_rewards_names(),
                                        max_episode_length=self._episode_timeout_ub,
                                        ep_freq=self.n_envs())
        self._set_ep_rewards_scaling(scaling=self._n_steps_task_rand_ub)
        
    def _set_ep_rewards_scaling(self,
                        scaling: int):
        
        self._episodic_rewards_metrics.set_constant_data_scaling(scaling=scaling)
        
    def _init_infos(self):

        self.custom_db_data = {}
        # by default always log this contact data
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror()
        contact_names = self._rhc_refs.rob_refs.contact_names()
        stepping_data = EpisodicData("RhcRefsFlag", rhc_latest_contact_ref, contact_names,
            ep_freq=self.n_envs())
        self._add_custom_db_info(db_data=stepping_data)
        # log also action data
        actions = self._actions.get_torch_mirror()
        action_names = self._get_action_names()
        action_data = EpisodicData("Actions", actions, action_names,
            ep_freq=self.n_envs())
        self._add_custom_db_info(db_data=action_data)

    def _add_custom_db_info(self, db_data: EpisodicData):
        self.custom_db_data[db_data.name()] = db_data

    def _init_terminations(self):

        # Boolean array indicating whether each environment episode has terminated after 
        # the current step. An episode termination could occur based on predefined conditions
        # in the environment, such as reaching a goal or exceeding a time limit.

        self._terminations = Terminations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False) 
        
        self._terminations.run()
    
    def _init_truncations(self):
        
        self._truncations = Truncations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False) 
        
        self._truncations.run()
        
    def _attach_to_shared_mem(self):

        # runs shared mem clients for getting observation and setting RHC commands
    
        self._robot_state = RobotState(namespace=self._namespace,
                                is_server=False, 
                                safe=self._safe_shared_mem,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                with_gpu_mirror=self._use_gpu,
                                with_torch_view=True)
        
        self._rhc_cmds = RhcCmds(namespace=self._namespace,
                                is_server=False, 
                                safe=self._safe_shared_mem,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                with_gpu_mirror=self._use_gpu,
                                with_torch_view=True)

        self._rhc_refs = RhcRefs(namespace=self._namespace,
                            is_server=False,
                            safe=self._safe_shared_mem,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            with_gpu_mirror=self._use_gpu,
                            with_torch_view=True)

        self._rhc_status = RhcStatus(namespace=self._namespace,
                                is_server=False,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                with_gpu_mirror=self._use_gpu,
                                with_torch_view=True)
        
        self._robot_state.run()
        self._rhc_cmds.run()
        self._rhc_refs.run()
        self._rhc_status.run()

        self._n_envs = self._robot_state.n_robots()
        self._n_jnts = self._robot_state.n_jnts()
        self._n_contacts = self._robot_state.n_contacts()

        # run server for agent commands
        self._agent_refs = AgentRefs(namespace=self._namespace,
                                is_server=True,
                                n_robots=self._n_envs,
                                n_jnts=self._robot_state.n_jnts(),
                                n_contacts=self._robot_state.n_contacts(),
                                contact_names=self._robot_state.contact_names(),
                                q_remapping=None,
                                with_gpu_mirror=self._use_gpu,
                                force_reconnection=True,
                                safe=False,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                fill_value=0)
        self._agent_refs.run()
        
        # remote stepping data
        self._remote_stepper = RemoteStepperSrvr(namespace=self._namespace,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            force_reconnection=True)
        self._remote_stepper.run()
        self._remote_resetter = RemoteResetSrvr(namespace=self._namespace,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            force_reconnection=True)
        self._remote_resetter.run()
        self._remote_reset_req = RemoteResetRequest(namespace=self._namespace,
                                            is_server=False, 
                                            verbose=self._verbose,
                                            vlevel=self._vlevel,
                                            safe=True)
        self._remote_reset_req.run()

        # episode steps counters (for detecting episode truncations for 
        # time limits) 
        self._ep_timeout_counter = EpisodesCounter(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_steps_lb=self._episode_timeout_lb,
                            n_steps_ub=self._episode_timeout_ub,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=False) # handles step counter through episodes and through envs
        self._ep_timeout_counter.run()
        self._task_rand_counter = TaskRandCounter(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_steps_lb=self._n_steps_task_rand_lb,
                            n_steps_ub=self._n_steps_task_rand_ub,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=False) # handles step counter through episodes and through envs
        self._task_rand_counter.run()
        if self._use_random_safety_reset:
            rand_range=round(5.0/100.0*self._random_rst_freq)
            self._rand_safety_reset_counter=SafetyRandResetsCounter(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_steps_lb=self._random_rst_freq-rand_range,
                            n_steps_ub=self._random_rst_freq+rand_range,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=False)
            self._rand_safety_reset_counter.run()

        # debug data servers
        traing_env_param_dict = {}
        traing_env_param_dict["use_gpu"] = self._use_gpu
        traing_env_param_dict["debug"] = self._is_debug
        traing_env_param_dict["n_preinit_steps"] = self._n_preinit_steps
        traing_env_param_dict["n_preinit_steps"] = self._n_envs
        
        self._training_sim_info = SharedTrainingEnvInfo(namespace=self._namespace,
                is_server=True, 
                training_env_params_dict=traing_env_param_dict,
                safe=False,
                force_reconnection=True,
                verbose=self._verbose,
                vlevel=self._vlevel)
        self._training_sim_info.run()
        
    def _activate_rhc_controllers(self):
        self._rhc_status.activation_state.get_torch_mirror()[:, :] = True
        self._rhc_status.activation_state.synch_all(read=False, retry=True) # activates all controllers
    
    def _synch_obs(self,
            gpu=True):

        # read from shared memory on CPU
        # robot state
        self._robot_state.root_state.synch_all(read = True, retry = True)
        self._robot_state.jnts_state.synch_all(read = True, retry = True)
        # rhc cmds
        self._rhc_cmds.contact_wrenches.synch_all(read = True, retry = True)
        # refs for root link and contacts
        self._rhc_refs.rob_refs.root_state.synch_all(read = True, retry = True)
        self._rhc_refs.contact_flags.synch_all(read = True, retry = True)
        # rhc cost
        self._rhc_status.rhc_cost.synch_all(read = True, retry = True)
        # rhc constr. violations
        self._rhc_status.rhc_constr_viol.synch_all(read = True, retry = True)
        # failure states
        self._rhc_status.fails.synch_all(read = True, retry = True)
        # tot cost and cnstr viol on nodes + step variable
        self._rhc_status.rhc_nodes_cost.synch_all(read = True, retry = True)
        self._rhc_status.rhc_nodes_constr_viol.synch_all(read = True, retry = True)
        self._rhc_status.rhc_step_var.synch_all(read = True, retry = True)
        self._rhc_status.rhc_fail_idx.synch_all(read = True, retry = True)
        if gpu:
            # copies data to "mirror" on GPU
            self._robot_state.root_state.synch_mirror(from_gpu=False) # copies shared data on GPU
            self._robot_state.jnts_state.synch_mirror(from_gpu=False)
            self._rhc_cmds.contact_wrenches.synch_mirror(from_gpu=False)
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=False)
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=False)
            self._rhc_status.rhc_cost.synch_mirror(from_gpu=False)
            self._rhc_status.rhc_constr_viol.synch_mirror(from_gpu=False)
            self._rhc_status.fails.synch_mirror(from_gpu=False)
            self._rhc_status.rhc_nodes_cost.synch_mirror(from_gpu=False)
            self._rhc_status.rhc_nodes_constr_viol.synch_mirror(from_gpu=False)
            self._rhc_status.rhc_step_var.synch_mirror(from_gpu=False)
            self._rhc_status.rhc_fail_idx.synch_mirror(from_gpu=False)
            #torch.cuda.synchronize() # ensuring that all the streams on the GPU are completed \
            # before the CPU continues execution

    def _synch_refs(self,
            gpu=True):

        if gpu:
            # copies latest refs from GPU to CPU shared mem for debugging
            self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=True) 
        self._agent_refs.rob_refs.root_state.synch_all(read=False, retry = True) # write on shared mem
    
    def _override_refs(self,
            gpu=True):

        # just used for setting agent refs externally (i.e. from shared mem on CPU)
        self._agent_refs.rob_refs.root_state.synch_all(read=True, retry = True) # first read from mem
        if gpu:
            # copies latest refs to GPU 
            self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=False) 

    def _clamp_obs(self, 
            obs: torch.Tensor):
        if self._is_debug:
            self._check_finite(obs, "observations", False)
        torch.nan_to_num(input=obs, out=obs, nan=torch.inf, posinf=None, neginf=None) # prevent nans
        obs.clamp_(self._obs_threshold_lb, self._obs_threshold_ub)
    
    def _clamp_rewards(self, 
            rewards: torch.Tensor):
        if self._is_debug:
            self._check_finite(rewards, "rewards", False)
        torch.nan_to_num(input=rewards, out=rewards, nan=torch.inf, posinf=None, neginf=None) # prevent nans
        rewards.clamp_(self._reward_thresh_lb, self._reward_thresh_ub)

    def get_actions_lb(self):
        return self._actions_lb

    def get_actions_ub(self):
        return self._actions_ub
    
    def _check_finite(self, 
                tensor: torch.Tensor,
                name: str, 
                throw: bool = False):
        if not torch.isfinite(tensor).all().item():
            exception = f"Found nonfinite elements in {name} tensor!!"
            if throw:
                print(tensor)
            Journal.log(self.__class__.__name__,
                "_check_finite",
                exception,
                LogType.EXCEP,
                throw_when_excep = throw)
            
    def _check_controllers_registered(self, 
                retry: bool = False):

        if retry:
            self._rhc_status.controllers_counter.synch_all(read=True, retry=True)
            n_connected_controllers = self._rhc_status.controllers_counter.get_torch_mirror()[0, 0].item()
            while not (n_connected_controllers == self._n_envs):
                warn = f"Expected {self._n_envs} controllers to be active during training, " + \
                    f"but got {n_connected_controllers}. Will wait for all to be connected..."
                Journal.log(self.__class__.__name__,
                    "_check_controllers_registered",
                    warn,
                    LogType.WARN,
                    throw_when_excep = False)
                nsecs = int(2 * 1000000000)
                PerfSleep.thread_sleep(nsecs) 
                self._rhc_status.controllers_counter.synch_all(read=True, retry=True)
                n_connected_controllers = self._rhc_status.controllers_counter.get_torch_mirror()[0, 0].item()
            info = f"All {n_connected_controllers} controllers connected!"
            Journal.log(self.__class__.__name__,
                "_check_controllers_registered",
                info,
                LogType.INFO,
                throw_when_excep = False)
            return True
        else:
            self._rhc_status.controllers_counter.synch_all(read=True, retry=True)
            n_connected_controllers = self._rhc_status.controllers_counter.get_torch_mirror()[0, 0].item()
            if not (n_connected_controllers == self._n_envs):
                exception = f"Expected {self._n_envs} controllers to be active during training, " + \
                    f"but got {n_connected_controllers}. Aborting..."
                Journal.log(self.__class__.__name__,
                    "_check_controllers_registered",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = False)
                return False
            return True
    
    @abstractmethod
    def _check_truncations(self):
        # default behaviour-> to be overriden by child
        truncations = self._truncations.get_torch_mirror(gpu=self._use_gpu)
        time_limits_reached = self._ep_timeout_counter.time_limits_reached()
        # truncate when episode timeout occurs
        truncations[:, :] = time_limits_reached

    @abstractmethod
    def _check_terminations(self):
        # default behaviour-> to be overriden by child
        terminations = self._terminations.get_torch_mirror(gpu=self._use_gpu)
        # terminate upon controller failure
        terminations[:, :] = self._rhc_status.fails.get_torch_mirror(gpu=self._use_gpu)
    
    @abstractmethod
    def _pre_step(self):
        pass
    
    @abstractmethod
    def _custom_post_step(self,episode_finished):
        pass

    @abstractmethod
    def _apply_actions_to_rhc(self):
        pass

    @abstractmethod
    def _compute_sub_rewards(self,
            obs: torch.Tensor,
            next_obs: torch.Tensor):
        pass

    @abstractmethod
    def _fill_obs(self,
            obs: torch.Tensor):
        pass

    @abstractmethod
    def _randomize_task_refs(self,
                env_indxs: torch.Tensor = None):
        pass

    def _custom_post_init(self):
        pass 
