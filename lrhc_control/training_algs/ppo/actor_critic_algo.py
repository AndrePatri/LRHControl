from lrhc_control.agents.actor_critic.ppo_tanh import ActorCriticTanh
from lrhc_control.agents.actor_critic.ppo_lrelu import ActorCriticLRelu

from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo
import torch 
import torch.optim as optim
import torch.nn as nn

import random

from typing import Dict

import os
import shutil

import time

import wandb

from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import VLevel

from abc import ABC, abstractmethod

class ActorCriticAlgoBase():

    # base class for actor-critic RL algorithms
     
    def __init__(self,
            env, 
            debug = False,
            seed: int = 1):

        self._env = env 
        self._seed = seed

        self._eval = False
        self._agent = ActorCriticTanh(obs_dim=self._env.obs_dim(),
                        actions_dim=self._env.actions_dim(),
                        actor_std=0.01,
                        critic_std=1.0)
        # self._agent = ActorCriticLRelu(obs_dim=self._env.obs_dim(),
        #                 actions_dim=self._env.actions_dim(),
        #                 actor_std=0.01,
        #                 critic_std=1.0)

        self._debug = debug

        self._optimizer = None

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._dbinfo_drop_dir = None
        self._model_path = None

        self._hyperparameters = {}
        
        self._episodic_reward_getter = self._env.ep_reward_getter()

        self._init_dbdata()
        
        self._init_params()
        
        self._setup_done = False

        self._verbose = False

        self._is_done = False
        
        self._shared_algo_data = None

        self._this_child_path = None
        self._this_basepath = os.path.abspath(__file__)
    
    def __del__(self):

        self.done()

    def setup(self,
            run_name: str,
            custom_args: Dict = {},
            verbose: bool = False,
            drop_dir_name: str = None,
            eval: bool = False,
            comment: str = ""):

        self._verbose = verbose

        self._eval = eval

        self._run_name = run_name
        from datetime import datetime
        self._time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')
        self._unique_id = self._time_id + "-" + self._run_name
        self._init_algo_shared_data(static_params=self._hyperparameters) # can only handle dicts with
        # numeric values
        self._hyperparameters.update(custom_args)

        # create dump directory + copy important files for debug
        self._init_drop_dir(drop_dir_name)

        if self._eval: # load pretrained model
            self._load_model(self._model_path)
            
        # seeding + deterministic behavior for reproducibility
        self._set_all_deterministic()

        if (self._debug):
            
            torch.autograd.set_detect_anomaly(self._debug)
            job_type = "evaluation" if eval else "training"
            wandb.init(
                project="LRHControl",
                group=self._run_name,
                name=self._unique_id,
                id=self._unique_id,
                job_type=job_type,
                # tags=None,
                notes=comment,
                resume="never", # do not allow runs with the same unique id
                mode="online", # "online", "offline" or "disabled"
                entity=None,
                sync_tensorboard=True,
                config=self._hyperparameters,
                monitor_gym=True,
                save_code=True,
                dir=self._drop_dir
            )
            wandb.watch(self._agent.critic)
            wandb.watch(self._agent.actor_mean)

        self._torch_device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")

        self._agent.to(self._torch_device) # move agent to target device

        self._optimizer = optim.Adam(self._agent.parameters(), 
                                lr=self._base_learning_rate, 
                                eps=1e-5 # small constant added to the optimization
                                )
        self._init_buffers()
        
        # self._env.reset()
        
        self._setup_done = True

        self._is_done = False

        self._start_time_tot = time.perf_counter()

        self._start_time = time.perf_counter()
    
    def is_done(self):

        return self._is_done 
    
    def learn(self):
        
        self._start_time = time.perf_counter()

        if not self._setup_done:
        
            self._should_have_called_setup()

        # annealing the learning rate if enabled (may improve convergence)
        if self._anneal_lr:

            frac = 1.0 - (self._it_counter - 1.0) / self._iterations_n
            self._learning_rate_now = frac * self._base_learning_rate
            self._optimizer.param_groups[0]["lr"] = self._learning_rate_now

        self._episodic_reward_getter.reset() # necessary, we don't want to accumulate 
        # debug rewards from previous rollout

        self._play(self._env_timesteps)

        # after rolling out policy, we get the episodic reward for the current policy
        self._episodic_rewards = self._episodic_reward_getter.get_total()
        self._episodic_rewards_detail = self._episodic_reward_getter.get()
        self._episodic_rewards_detail_env_avrg = self._episodic_reward_getter.get_env_avrg()
        self._episodic_rewards_env_avrg = self._episodic_reward_getter.get_total_env_avrg()

        self._bootstrap()

        self._improve_policy()

        self._post_step()

    def eval(self, 
        n_timesteps: int):

        self._start_time = time.perf_counter()

        if not self._setup_done:
        
            self._should_have_called_setup()

        self._play(n_timesteps)

        self._eval_post_step()

    @abstractmethod
    def _play(self):
        pass
    
    @abstractmethod
    def _bootstrap(self):
       pass
    
    @abstractmethod
    def _improve_policy(self):
        pass

    def done(self):
        
        if not self._is_done:

            if self._save_model:
                
                info = f"Saving model and other data to {self._model_path}"

                Journal.log(self.__class__.__name__,
                    "done",
                    info,
                    LogType.INFO,
                    throw_when_excep = True)

                torch.save(self._agent.state_dict(), self._model_path)

                info = f"Done."

                Journal.log(self.__class__.__name__,
                    "done",
                    info,
                    LogType.INFO,
                    throw_when_excep = True)
            
            self._dump_dbinfo_to_file()

            if self._shared_algo_data is not None:

                self._shared_algo_data.close() # close shared memory

            self._env.close()

            self._is_done = True

    def _dump_dbinfo_to_file(self):

        import h5py

        info = f"Dumping debug info at {self._dbinfo_drop_dir}"
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        print(self._hyperparameters)

        with h5py.File(self._dbinfo_drop_dir+".hdf5", 'w') as hf:
            # hf.create_dataset('numpy_data', data=numpy_data)
            # Write dictionaries to HDF5 as attributes
            for key, value in self._hyperparameters.items():
                if value is None:
                    value = "None"
                hf.attrs[key] = value
        
        info = f"done."
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)

    def _load_model(self,
            model_path: str):
        
        info = f"Loading model at {self._model_path}"

        Journal.log(self.__class__.__name__,
            "_load_model",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._agent.load_state_dict(torch.load(model_path, 
                            map_location=self._torch_device))
        self._agent.eval()

    def _set_all_deterministic(self):

        random.seed(self._seed) # python seed
        torch.manual_seed(self._seed)
        torch.backends.cudnn.deterministic = self._torch_deterministic
        # torch.use_deterministic_algorithms(mode=True) # will throw excep. when trying to use non-det. algos
        import numpy as np
        np.random.seed(0)

    def _init_drop_dir(self,
                drop_dir_name: str = None):

        if drop_dir_name is None:
            # drop to current directory
            self._drop_dir = "./" + f"{self.__class__.__name__}/" + self._run_name + "/" + self._unique_id
        else:
            self._drop_dir = drop_dir_name + "/" + f"{self.__class__.__name__}/" + self._run_name + "/" + self._unique_id

        self._model_path = self._drop_dir + "/" + self._unique_id + "_model"

        if self._eval: # drop in same directory
            f = self._drop_dir + "/" + self._unique_id + "_evalrun"
        
        self._dbinfo_drop_dir = self._drop_dir + "/" + self._unique_id + "db_info"

        aux_drop_dir = self._drop_dir + "/other"
        os.makedirs(self._drop_dir)
        os.makedirs(aux_drop_dir)

        filepaths = self._env.get_file_paths() # envs implementation
        filepaths.append(self._this_basepath) # algorithm implementation
        filepaths.append(self._this_child_path)
        filepaths.append(self._agent.get_impl_path()) # agent implementation
        for file in filepaths:
            shutil.copy(file, self._drop_dir)

        aux_dirs = self._env.get_aux_dir()
        for aux_dir in aux_dirs:
            shutil.copytree(aux_dir, aux_drop_dir, dirs_exist_ok=True)

    def _post_step(self):

        self._debug_info()

        self._it_counter +=1 
        
        if self._it_counter == self._iterations_n:

            self.done()

        if self._verbose:
            
            info = f"\nN. PPO iterations performed: {self._it_counter}/{self._iterations_n}\n" + \
                f"N. policy updates performed: {self._it_counter * self._update_epochs * self._num_minibatches}/" + \
                f"{self._update_epochs * self._num_minibatches * self._iterations_n}\n" + \
                f"N. timesteps performed: {self._it_counter * self._batch_size}/{self._total_timesteps}\n" + \
                f"Elapsed minutes: {self._elapsed_min}\n" + \
                f"Estimated remaining training time: " + \
                f"{self._elapsed_min/60 * 1/self._it_counter * (self._iterations_n-self._it_counter)} hours\n" + \
                f"Average episodic reward across all environments: {self._episodic_rewards_env_avrg}\n" + \
                f"Average episodic rewards across all environments {self._reward_names}: {self._episodic_rewards_detail_env_avrg}\n"

            Journal.log(self.__class__.__name__,
                "_post_step",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
    def _eval_post_step(self):
        
        info = f"Evaluation of policy model {self._model_path} completed. Dropping evaluation info to {self._drop_dir}"

        Journal.log(self.__class__.__name__,
            "_post_step",
            info,
            LogType.INFO,
            throw_when_excep = True)
            
    def _should_have_called_setup(self):

        exception = f"setup() was not called!"

        Journal.log(self.__class__.__name__,
            "_should_have_called_setup",
            exception,
            LogType.EXCEP,
            throw_when_excep = True)
    
    def _debug_info(self):
        
        self._elapsed_min = (time.perf_counter() - self._start_time_tot) / 60
        self._env_step_fps = self._batch_size / self._bootstrap_dt
        
        if self._debug:

            # write debug info to shared memory    
            self._shared_algo_data.write(dyn_info_name=["current_batch_iteration", 
                                            "n_of_performed_policy_updates",
                                            "n_of_played_episodes", 
                                            "n_of_timesteps_done",
                                            "current_learning_rate",
                                            "env_step_fps",
                                            "boostrap_dt",
                                            "policy_update_dt",
                                            "learn_step_total_fps",
                                            "elapsed_min"
                                            ],
                                    val=[self._it_counter, 
                    (self._it_counter+1) * self._update_epochs * self._num_minibatches,
                    self._n_of_played_episodes, 
                    (self._it_counter+1) * self._batch_size,
                    self._learning_rate_now,
                    self._env_step_fps,
                    self._bootstrap_dt,
                    self._policy_update_dt,
                    self._learn_step_total_fps,
                    self._elapsed_min
                    ])

            d = {'tot_episodic_reward': wandb.Histogram(self._episodic_rewards.numpy()),
                'tot_episodic_reward_env_avrg': self._episodic_rewards_env_avrg,
                'ppo_iteration' : self._it_counter}
            d.update({f"sub_reward/{self._episodic_reward_getter.reward_names()[i]}_env_avrg":
                      self._episodic_rewards_detail_env_avrg[i] for i in range(len(self._episodic_reward_getter.reward_names()))})
            d.update({f"sub_reward/{self._episodic_reward_getter.reward_names()[i]}":
                      wandb.Histogram(self._episodic_rewards_detail.numpy()[:, i:i+1]) for i in range(len(self._episodic_reward_getter.reward_names()))})
            
            # write debug info to shared memory    
            wandb.log(d)
    
    def _init_dbdata(self):

        # initalize some debug data

        self._bootstrap_dt = 0.0
        self._gae_dt = 0.0

        self._env_step_fps = 0.0
        self._policy_update_dt = 0.0
        self._learn_step_total_fps = 0.0
        self._n_of_played_episodes = 0.0
        self._elapsed_min = 0

        self._episodic_rewards = None
        self._episodic_rewards_env_avrg = None
        self._episodic_rewards_detail = None
        self._episodic_rewards_detail_env_avrg = None
        self._reward_names = "[" + ', '.join(self._episodic_reward_getter.reward_names()) + "]"

    def _init_params(self):

        self._dtype = self._env.dtype()

        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()

        self._run_name = "DummyRun"

        self._use_gpu = self._env.using_gpu()
        self._torch_device = torch.device("cpu") # defaults to cpu
        self._torch_deterministic = True

        self._save_model = True
        self._env_name = self._env.name()

        self._iterations_n = 300
        self._env_timesteps = 256

        self._env_episode_n_steps = self._env.n_steps_per_episode()
        self._total_timesteps = self._iterations_n * (self._env_timesteps * self._num_envs)
        self._batch_size =int(self._num_envs * self._env_timesteps)
        self._num_minibatches = self._env.n_envs()
        self._minibatch_size = int(self._batch_size // self._num_minibatches)

        self._base_learning_rate = 3e-4
        self._learning_rate_now = self._base_learning_rate
        self._anneal_lr = True
        self._discount_factor = 0.99
        self._gae_lambda = 0.95
        
        self._update_epochs = 5
        self._norm_adv = True
        self._clip_coef = 0.2
        self._clip_vloss = True
        self._entropy_coeff = 0.01
        self._val_f_coeff = 0.5
        self._max_grad_norm = 0.5
        self._target_kl = None

        # write them to hyperparam dictionary for debugging
        self._hyperparameters["n_envs"] = self._num_envs
        self._hyperparameters["obs_dim"] = self._obs_dim
        self._hyperparameters["actions_dim"] = self._actions_dim
        self._hyperparameters["seed"] = self._seed
        self._hyperparameters["using_gpu"] = self._use_gpu
        self._hyperparameters["n_iterations"] = self._iterations_n
        self._hyperparameters["n_policy_updates_per_batch"] = self._update_epochs * self._num_minibatches
        self._hyperparameters["n_policy_updates_when_done"] = \
            self._iterations_n * self._update_epochs * self._num_minibatches
        self._hyperparameters["n steps per env. episode"] = self._env_episode_n_steps
        self._hyperparameters["n steps per env. rollout"] = self._env_timesteps
        self._hyperparameters["per-batch update_epochs"] = self._update_epochs
        self._hyperparameters["per-epoch policy updates"] = self._num_minibatches
        self._hyperparameters["total policy updates to be performed"] = self._update_epochs * self._num_minibatches * self._iterations_n
        self._hyperparameters["total_timesteps to be simulated"] = self._total_timesteps
        self._hyperparameters["batch_size"] = self._batch_size
        self._hyperparameters["minibatch_size"] = self._minibatch_size
        self._hyperparameters["total_timesteps"] = self._total_timesteps
        self._hyperparameters["base_learning_rate"] = self._base_learning_rate
        self._hyperparameters["anneal_lr"] = self._anneal_lr
        self._hyperparameters["discount_factor"] = self._discount_factor
        self._hyperparameters["gae_lambda"] = self._gae_lambda
        self._hyperparameters["norm_adv"] = self._norm_adv
        self._hyperparameters["clip_coef"] = self._clip_coef
        self._hyperparameters["clip_vloss"] = self._clip_vloss
        self._hyperparameters["entropy_coeff"] = self._entropy_coeff
        self._hyperparameters["val_f_coeff"] = self._val_f_coeff
        self._hyperparameters["max_grad_norm"] = self._max_grad_norm
        self._hyperparameters["target_kl"] = self._target_kl

        # small debug log
        info = f"\nUsing \n" + \
            f"batch_size {self._batch_size}\n" + \
            f"minibatch_size {self._minibatch_size}\n" + \
            f"num_minibatches {self._num_minibatches}\n" + \
            f"n steps per env. episode {self._env_episode_n_steps}\n" + \
            f"n steps per env. rollout {self._env_timesteps}\n" + \
            f"iterations_n {self._iterations_n}\n" + \
            f"per-batch update_epochs {self._update_epochs}\n" + \
            f"per-epoch policy updates {self._num_minibatches}\n" + \
            f"total policy updates to be performed {self._update_epochs * self._num_minibatches * self._iterations_n}\n" + \
            f"total_timesteps to be simulated {self._total_timesteps}\n"
        Journal.log(self.__class__.__name__,
            "_init_params",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._it_counter = 0

    def _init_buffers(self):

        self._obs = torch.full(size=(self._env_timesteps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._actions = torch.full(size=(self._env_timesteps, self._num_envs, self._actions_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._logprobs = torch.full(size=(self._env_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._rewards = torch.full(size=(self._env_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._dones = torch.full(size=(self._env_timesteps, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._values = torch.full(size=(self._env_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._advantages = torch.full(size=(self._env_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._returns = torch.full(size=(self._env_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)

    def _init_algo_shared_data(self,
                static_params: Dict):

        self._shared_algo_data = SharedRLAlgorithmInfo(namespace="CleanPPO",
                is_server=True, 
                static_params=static_params,
                verbose=self._verbose, 
                vlevel=VLevel.V2, 
                safe=False,
                force_reconnection=True)

        self._shared_algo_data.run()