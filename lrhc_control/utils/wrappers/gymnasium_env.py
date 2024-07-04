from lrhc_control.utils.shared_data.training_env import Observations, NextObservations
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Terminations
from lrhc_control.utils.shared_data.training_env import Truncations

from lrhc_control.utils.episodic_rewards import EpisodicRewards

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

import torch
import numpy  as np

class Gymnasium2LRHCEnv():

    def __init__(self,
            gymnasium_env,
            namespace: str = "Gymnasium2LRHCEnv",
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            debug: bool = True,
            use_gpu: bool = True):
        
        # dtype mapping dictionary
        self._dtype_mapping = {
            'float32': torch.float32,
            'float64': torch.float64
        }

        self._env = gymnasium_env
        
        self._namespace = namespace
        self._action_repeat = 1
        if self._action_repeat <=0: 
            self._action_repeat = 1

        self._with_gpu_mirror = True
        self._safe_shared_mem = False
        
        self._closed = False

        self._obs_dim = torch.tensor(self._env.single_observation_space.shape).prod().item()
        self._actions_dim = torch.tensor(self._env.single_action_space.shape).prod().item()

        self._use_gpu = use_gpu
        
        self._dtype = self._dtype_mapping[str(self._env.single_observation_space.dtype)]
        
        self._verbose = verbose
        self._vlevel = vlevel
        self._is_debug = debug
        
        self._n_envs = self._env.num_envs

        device = "cuda" if self._use_gpu else "cpu"
        # action scalings to be applied to agent's output
        self._actions_ub = torch.full((1, self._actions_dim), dtype=self._dtype, device=device,
                                        fill_value=1.0)
        self._actions_lb = torch.full((1, self._actions_dim), dtype=self._dtype, device=device,
                                        fill_value=-1.0)
        
        reward_thresh_default = 1.0
        device = "cuda" if self._use_gpu else "cpu"
        self._reward_thresh_lb = torch.full((1, 1), dtype=self._dtype, fill_value=-reward_thresh_default, device=device) # used for clipping rewards
        self._reward_thresh_ub = torch.full((1, 1), dtype=self._dtype, fill_value=reward_thresh_default, device=device) 

        self._obs = None
        self._next_obs = None
        self._actions = None
        self._tot_rewards = None
        self._terminations = None
        self._truncations = None
        
        self.custom_db_data = {}

        self._episodic_rewards_getter = None
        self._episode_duration_ub = 100

        self._episode_timeout_lb = self._episode_duration_ub
        self._episode_timeout_ub = self._episode_duration_ub
        
        self._n_steps_task_rand_lb = -1
        self._n_steps_task_rand_ub = -1

        self._init_shared_data()

    def __del__(self):
        self.close()

    def close(self):
        
        if not self._closed:
            
            self._next_obs.close()
            self._obs.close()
            self._actions.close()
            self._tot_rewards.close()

            self._terminations.close()
            self._truncations.close()

            self._closed = True

    def _init_shared_data(self):
        
        obs_names = [""]*self._obs_dim
        for i in range(len(obs_names)):
            obs_names[i] = f"obs_n.{i}"
        
        action_names = [""]*self._actions_dim
        for i in range(len(action_names)):
            action_names[i] = f"action_n.{i}"

        self._obs = Observations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            obs_dim=self._obs_dim,
                            obs_names=obs_names,
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
                            obs_names=obs_names,
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._actions = Actions(namespace=self._namespace,
                            n_envs=self._n_envs,
                            action_dim=self._actions_dim,
                            action_names=action_names,
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
        
        self._terminations = Terminations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False) 
        
        self._truncations = Truncations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False)

        self._obs.run()
        self._next_obs.run()
        self._actions.run()
        self._tot_rewards.run()
        self._truncations.run()
        self._terminations.run()
        
        self._episodic_rewards_getter = EpisodicRewards(reward_tensor=self._tot_rewards.get_torch_mirror(),
                                        reward_names=["total_reward"],
                                        max_episode_length=self._episode_duration_ub)
        self._episodic_rewards_getter.set_constant_data_scaling(scaling=self._episode_duration_ub)
    
    def ep_reward_getter(self):
        return self._episodic_rewards_getter
    
    def dtype(self):
        return self._dtype 
    
    def n_envs(self):
        return self._n_envs
    
    def obs_dim(self):
        return self._obs_dim
    
    def actions_dim(self):
        return self._actions_dim
    
    def get_actions_lb(self):
        return self._actions_lb

    def get_actions_ub(self):
        return self._actions_ub

    def name(self):
        return self._namespace
    
    def episode_timeout_bounds(self):
        return self._episode_timeout_lb, self._episode_timeout_ub
    
    def task_rand_timeout_bounds(self):
        return self._n_steps_task_rand_lb, self._n_steps_task_rand_ub
    
    def n_action_reps(self):
        return self._action_repeat
    
    def using_gpu(self):
        return self._use_gpu
    
    def get_file_paths(self):
        empty_list = []
        return empty_list
    
    def get_aux_dir(self):
        empty_list = []
        return empty_list

    def reset(self):
        
        self._actions.reset()
        self._obs.reset()
        self._next_obs.reset()
        self._tot_rewards.reset()

        self._terminations.reset()
        self._truncations.reset()

        # fill obs from gymansium env

    def step(self, 
            action):
        
        stepping_ok = True

        # step gymnasium env using the given action
        
        self.post_step()

        return stepping_ok
    
    def post_step(self):
        if self._is_debug:
            terminated = self._terminations.get_torch_mirror(gpu=self._use_gpu)
            truncated = self._truncations.get_torch_mirror(gpu=self._use_gpu)
            episode_finished = torch.logical_or(terminated,
                            truncated)
            episode_finished_cpu = episode_finished.cpu()
            self._debug() # copies db data on shared memory
            self._update_custom_db_data(episode_finished=episode_finished_cpu)
            self._episodic_rewards_getter.update(rewards = self._tot_rewards.get_torch_mirror(gpu=False),
                            ep_finished=episode_finished_cpu)

if __name__ == "__main__":  

    from lrhc_control.training_algs.sac.sac import SAC

    import gymnasium as gym

    import argparse
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--db', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable local data logging for the algorithm (reward metrics, etc..)')
    parser.add_argument('--env_db', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable env db data logging on \
                            shared mem (e.g.reward metrics are not available for reading anymore)')
    parser.add_argument('--rmdb', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable remote debug (e.g. data logging on remote servers)')
    parser.add_argument('--obs_norm', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable the use of running normalizer in agent')
    parser.add_argument('--run_name', type=str, help='Name of training run', default="GymnasiumEnvTest")
    parser.add_argument('--drop_dir', type=str, help='Directory root where all run data will be dumped',default="/tmp")
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--seed', type=int, help='seed', default=1)
    parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=False, help='Whether to perform an evaluation run')
    parser.add_argument('--mpath', type=str, help='Model path to be used for policy evaluation',default=None)
    parser.add_argument('--n_evals', type=int, help='N. of evaluation rollouts to be performed', default=None)
    parser.add_argument('--n_timesteps', type=int, help='Toal n. of timesteps for each evaluation rollout', default=None)
    parser.add_argument('--dump_checkpoints', action=argparse.BooleanOptionalAction, default=True, help='Whether to dump model checkpoints during training')
    parser.add_argument('--use_cpu', action=argparse.BooleanOptionalAction, default=False, help='If set, all the training (data included) will be perfomed on CPU')

    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory', default="Gymnasium2LRHCEnv")
    parser.add_argument('--num_envs', type=int, help='seed', default=1)

    args = parser.parse_args()

    env = gym.make_vec('InvertedPendulum-v4', num_envs=args.num_envs)
    env_wrapper = Gymnasium2LRHCEnv(gymnasium_env=env,
                        namespace=args.ns,
                        verbose=True,
                        vlevel=VLevel.V2,
                        debug=args.env_db,
                        use_gpu=not args.use_cpu)

    algo = SAC(env=env_wrapper, 
            debug=args.db, 
            remote_db=args.rmdb,
            seed=args.seed)
    algo.setup(run_name=args.run_name, 
        verbose=True,
        drop_dir_name=args.drop_dir,
        custom_args = {},
        comment=args.comment,
        eval=args.eval,
        model_path=args.mpath,
        n_evals=args.n_evals,
        n_timesteps_per_eval=args.n_timesteps,
        dump_checkpoints=args.dump_checkpoints,
        norm_obs=args.obs_norm)
    
    try:
        while not algo.is_done():
            if not args.eval:
                if not algo.learn():
                    algo.done()
            else: # eval phase
                if not algo.eval():
                    algo.done()
    except KeyboardInterrupt:
        algo.done() # in case it's interrupted by user