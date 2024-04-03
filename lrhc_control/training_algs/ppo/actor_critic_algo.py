from lrhc_control.agents.actor_critic.ppo_tanh import ActorCriticTanh
from lrhc_control.agents.actor_critic.ppo_lrelu import ActorCriticLRelu
from lrhc_control.agents.actor_critic.ppo_tanhv2 import ActorCriticThB

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
        
        self._debug = debug

        self._optimizer = None

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._dbinfo_drop_dir = None
        self._model_path = None
        
        self._policy_update_db_data_dict =  {}
        self._hyperparameters = {}
        
        self._episodic_reward_getter = self._env.ep_reward_getter()
        
        self._init_params()
        
        self._init_dbdata()

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
            wandb.watch(self._agent, log="all")
            # wandb.watch(self._agent.actor_mean, log="all")
            # wandb.watch(self.actor_logstd, log="all")
            
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
        
        if not self._setup_done:
        
            self._should_have_called_setup()

        # annealing the learning rate if enabled (may improve convergence)
        if self._anneal_lr:

            frac = 1.0 - (self._it_counter - 1.0) / self._iterations_n
            self._learning_rate_now = frac * self._base_learning_rate
            self._optimizer.param_groups[0]["lr"] = self._learning_rate_now

        self._episodic_reward_getter.reset() # necessary, we don't want to accumulate 
        # debug rewards from previous rollout

        self._start_time = time.perf_counter()

        rollout_ok = self._play(self._env_timesteps)
        if not rollout_ok:
            return False
        # after rolling out policy, we get the episodic reward for the current policy
        self._episodic_rewards[self._it_counter, :, :] = self._episodic_reward_getter.get_total() # total ep. rewards across envs
        self._episodic_sub_rewards[self._it_counter, :, :] = self._episodic_reward_getter.get() # sub-episodic rewards across envs
        self._episodic_sub_rewards_env_avrg[self._it_counter, :] = self._episodic_reward_getter.get_env_avrg() # avrg over envs
        self._episodic_rewards_env_avrg[self._it_counter, :, :] = self._episodic_reward_getter.get_total_env_avrg() # avrg over envs
        self._rollout_t = time.perf_counter()

        self._compute_returns()
        self._gae_t = time.perf_counter()

        self._improve_policy()
        self._policy_update_t = time.perf_counter()

        self._post_step()

        return True

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
    def _compute_returns(self):
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
        
        with h5py.File(self._dbinfo_drop_dir+".hdf5", 'w') as hf:
            # hf.create_dataset('numpy_data', data=numpy_data)
            # Write dictionaries to HDF5 as attributes
            for key, value in self._hyperparameters.items():
                if value is None:
                    value = "None"
                hf.attrs[key] = value
            
            # rewards
            hf.create_dataset('sub_reward_names', data=self._reward_names, 
                dtype='S20') 
            hf.create_dataset('episodic_rewards', data=self._episodic_rewards.numpy())
            hf.create_dataset('episodic_sub_rewards', data=self._episodic_sub_rewards.numpy())
            hf.create_dataset('episodic_sub_rewards_env_avrg', data=self._episodic_sub_rewards_env_avrg.numpy())
            hf.create_dataset('episodic_rewards_env_avrg', data=self._episodic_rewards_env_avrg.numpy())

            # profiling data
            hf.create_dataset('rollout_dt', data=self._rollout_dt.numpy())
            hf.create_dataset('env_step_fps', data=self._env_step_fps.numpy())
            hf.create_dataset('env_step_rt_factor', data=self._env_step_rt_factor.numpy())
            hf.create_dataset('gae_dt', data=self._gae_dt.numpy())
            hf.create_dataset('policy_update_dt', data=self._policy_update_dt.numpy())
            hf.create_dataset('policy_update_fps', data=self._policy_update_fps.numpy())
            hf.create_dataset('n_of_played_episodes', data=self._n_of_played_episodes.numpy())
            hf.create_dataset('n_timesteps_done', data=self._n_timesteps_done.numpy())
            hf.create_dataset('n_policy_updates', data=self._n_policy_updates.numpy())
            hf.create_dataset('elapsed_min', data=self._elapsed_min.numpy())
            hf.create_dataset('learn_rates', data=self._learning_rates.numpy())

            # ppo iterations db data
            hf.create_dataset('tot_loss', data=self._tot_loss.numpy())
            hf.create_dataset('value_loss', data=self._value_loss.numpy())
            hf.create_dataset('policy_loss', data=self._policy_loss.numpy())
            hf.create_dataset('entropy_loss', data=self._entropy_loss.numpy())
            hf.create_dataset('old_approx_kl', data=self._old_approx_kl.numpy())
            hf.create_dataset('approx_kl', data=self._approx_kl.numpy())
            hf.create_dataset('clipfrac', data=self._clipfrac.numpy())
            hf.create_dataset('explained_variance', data=self._explained_variance.numpy())

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

        self._it_counter +=1 

        self._rollout_dt[self._it_counter-1] = self._rollout_t -self._start_time
        self._gae_dt[self._it_counter-1] = self._gae_t - self._rollout_t
        self._policy_update_dt[self._it_counter-1] = self._policy_update_t - self._gae_t
        
        self._n_of_played_episodes[self._it_counter-1] = self._episodic_reward_getter.get_n_played_episodes()
        self._n_timesteps_done[self._it_counter-1] = self._it_counter * self._batch_size
        self._n_policy_updates[self._it_counter-1] = self._it_counter * self._update_epochs * self._num_minibatches
        
        self._elapsed_min[self._it_counter-1] = (time.perf_counter() - self._start_time_tot) / 60
        
        self._learning_rates[self._it_counter-1] = self._learning_rate_now

        self._env_step_fps[self._it_counter-1] = self._batch_size / self._rollout_dt[self._it_counter-1]
        self._env_step_rt_factor[self._it_counter-1] = self._env_step_fps[self._it_counter-1] * self._hyperparameters["control_clust_dt"]
        self._policy_update_fps[self._it_counter-1] = self._update_epochs * self._num_minibatches / self._policy_update_dt[self._it_counter-1]

        self._log_info()

        if self._it_counter == self._iterations_n:

            self.done()
            
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
    
    def _log_info(self):
        
        if self._debug:

            info_names=["current_ppo_iteration", 
                "n_of_performed_policy_updates",
                "n_of_played_episodes", 
                "n_of_timesteps_done",
                "current_learning_rate",
                "rollout_dt",
                "return_dt",
                "policy_improv_dt",
                "env_step_fps",
                "env_step_rt_factor",
                "policy_improv_fps",
                "elapsed_min"
                ]
            info_data = [self._it_counter, 
                self._n_policy_updates[self._it_counter-1].item(),
                self._n_of_played_episodes[self._it_counter-1].item(), 
                self._n_timesteps_done[self._it_counter-1].item(),
                self._learning_rate_now,
                self._rollout_dt[self._it_counter-1].item(),
                self._gae_dt[self._it_counter-1].item(),
                self._policy_update_dt[self._it_counter-1].item(),
                self._env_step_fps[self._it_counter-1].item(),
                self._env_step_rt_factor[self._it_counter-1].item(),
                self._policy_update_fps[self._it_counter-1].item(),
                self._elapsed_min[self._it_counter-1].item()
                ]

            # write debug info to shared memory    
            self._shared_algo_data.write(dyn_info_name=info_names,
                                    val=info_data)

            wandb_d = {'tot_episodic_reward': wandb.Histogram(self._episodic_rewards[self._it_counter-1, :, :].numpy()),
                'tot_episodic_reward_env_avrg': self._episodic_rewards_env_avrg[self._it_counter-1, :, :].item(),
                'ppo_iteration' : self._it_counter}
            wandb_d.update(dict(zip(info_names, info_data)))
            wandb_d.update({f"sub_reward/{self._reward_names[i]}_env_avrg":
                      self._episodic_sub_rewards_env_avrg[self._it_counter-1, i] for i in range(len(self._reward_names))})
            wandb_d.update({f"sub_reward/{self._reward_names[i]}":
                      wandb.Histogram(self._episodic_sub_rewards.numpy()[self._it_counter-1, :, i:i+1]) for i in range(len(self._reward_names))})
            wandb_d.update(self._policy_update_db_data_dict)
        
            # write debug info to shared memory    
            wandb.log(wandb_d),

        if self._verbose:
            
            info = f"\nN. PPO iterations performed: {self._it_counter}/{self._iterations_n}\n" + \
                f"N. policy updates performed: {self._n_policy_updates[self._it_counter-1].item()}/" + \
                f"{self._update_epochs * self._num_minibatches * self._iterations_n}\n" + \
                f"N. timesteps performed: {self._it_counter * self._batch_size}/{self._total_timesteps}\n" + \
                f"Elapsed minutes: {self._elapsed_min[self._it_counter-1].item()}\n" + \
                f"Estimated remaining training time: " + \
                f"{self._elapsed_min[self._it_counter-1].item()/60 * 1/self._it_counter * (self._iterations_n-self._it_counter)} hours\n" + \
                f"Average episodic reward across all environments: {self._episodic_rewards_env_avrg[self._it_counter-1, :, :].item()}\n" + \
                f"Average episodic rewards across all environments {self._reward_names_str}: {self._episodic_sub_rewards_env_avrg[self._it_counter-1, :]}\n" + \
                f"Current rollout fps: {self._env_step_fps[self._it_counter-1].item()}, time for rollout {self._rollout_dt[self._it_counter-1].item()} s\n" + \
                f"Current rollout rt factor: {self._env_step_rt_factor[self._it_counter-1].item()}\n" + \
                f"Time to compute bootstrap {self._gae_dt[self._it_counter-1].item()} s\n" + \
                f"Current policy update fps: {self._policy_update_fps[self._it_counter-1].item()}, time for policy updates {self._policy_update_dt[self._it_counter-1].item()} s\n"
            Journal.log(self.__class__.__name__,
                "_post_step",
                info,
                LogType.INFO,
                throw_when_excep = True)

    def _init_dbdata(self):

        # initalize some debug data

        # rollout phase
        self._rollout_dt = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=-1.0, device="cpu")
        self._rollout_t = -1.0
        self._env_step_fps = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._env_step_rt_factor = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        # gae computation
        self._gae_t = -1.0
        self._gae_dt = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=-1.0, device="cpu")

        # ppo iteration
        self._policy_update_t = -1.0
        self._policy_update_dt = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=-1.0, device="cpu")
        self._policy_update_fps = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._n_of_played_episodes = torch.full((self._iterations_n, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_timesteps_done = torch.full((self._iterations_n, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_policy_updates = torch.full((self._iterations_n, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._elapsed_min = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0, device="cpu")
        self._learning_rates = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0, device="cpu")
        
        # reward db data
        tot_ep_rew_shape = self._episodic_reward_getter.get_total().shape
        subrep_ewards_shape = self._episodic_reward_getter.get().shape
        subrep_ewards_avrg_shape = self._episodic_reward_getter.get_env_avrg().shape
        self._reward_names = self._episodic_reward_getter.reward_names()
        self._reward_names_str = "[" + ', '.join(self._reward_names) + "]"
        self._episodic_rewards = torch.full((self._iterations_n, tot_ep_rew_shape[0], tot_ep_rew_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_rewards_env_avrg = torch.full((self._iterations_n, 1, 1), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_sub_rewards = torch.full((self._iterations_n, subrep_ewards_shape[0], subrep_ewards_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_sub_rewards_env_avrg = torch.full((self._iterations_n, subrep_ewards_avrg_shape[0]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")

        # ppo iteration db data
        self._tot_loss = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._value_loss = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._policy_loss = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._entropy_loss = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._old_approx_kl = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._approx_kl = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._clipfrac = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._explained_variance = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")

    def _init_params(self):

        self._dtype = self._env.dtype()

        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()

        self._run_name = "DefaultRun" # default
        self._env_name = self._env.name()
        self._env_episode_n_steps_lb, self._env_episode_n_steps_ub = self._env.n_steps_per_episode()

        self._use_gpu = self._env.using_gpu()
        self._torch_device = torch.device("cpu") # defaults to cpu
        self._torch_deterministic = True

        self._save_model = True

        # main algo settings
        self._iterations_n = 1500 # number of ppo iterations
        self._batch_size_nom = 8192 # 24576
        self._num_minibatches = 8
        self._env_timesteps = int(self._batch_size_nom / self._num_envs)
        self._batch_size = self._env_timesteps * self._num_envs
        self._minibatch_size = int(self._batch_size // self._num_minibatches)
        self._total_timesteps = self._iterations_n * self._batch_size
        
        self._base_learning_rate = 3e-4
        self._learning_rate_now = self._base_learning_rate
        self._anneal_lr = True
        self._discount_factor = 0.99
        self._gae_lambda = 0.95
        
        self._update_epochs = 10
        self._norm_adv = True
        self._clip_coef = 0.4
        self._clip_vloss = True
        self._entropy_coeff = 0.0 # 0.01
        self._val_f_coeff = 0.5
        self._max_grad_norm = 0.5
        self._target_kl = None

        self._n_policy_updates_to_be_done = self._update_epochs * self._num_minibatches * self._iterations_n

        # write them to hyperparam dictionary for debugging
        self._hyperparameters["n_envs"] = self._num_envs
        self._hyperparameters["obs_dim"] = self._obs_dim
        self._hyperparameters["actions_dim"] = self._actions_dim
        self._hyperparameters["seed"] = self._seed
        self._hyperparameters["using_gpu"] = self._use_gpu
        self._hyperparameters["n_iterations"] = self._iterations_n
        self._hyperparameters["n_policy_updates_per_batch"] = self._update_epochs * self._num_minibatches
        self._hyperparameters["n_policy_updates_when_done"] = self._n_policy_updates_to_be_done
        self._hyperparameters["n steps per env. episode lb"] = self._env_episode_n_steps_lb
        self._hyperparameters["n steps per env. episode ub"] = self._env_episode_n_steps_ub
        self._hyperparameters["n steps per env. rollout"] = self._env_timesteps
        self._hyperparameters["per-batch update_epochs"] = self._update_epochs
        self._hyperparameters["per-epoch policy updates"] = self._num_minibatches
        self._hyperparameters["total policy updates to be performed"] = self._update_epochs * self._num_minibatches * self._iterations_n
        self._hyperparameters["total_timesteps to be simulated"] = self._total_timesteps
        self._hyperparameters["batch_size"] = self._batch_size
        self._hyperparameters["batch_size_nom"] = self._batch_size_nom
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
            f"batch_size_nominal {self._batch_size_nom}\n" + \
            f"batch_size {self._batch_size}\n" + \
            f"num_minibatches {self._num_minibatches}\n" + \
            f"minibatch_size {self._minibatch_size}\n" + \
            f"per-batch update_epochs {self._update_epochs}\n" + \
            f"iterations_n {self._iterations_n}\n" + \
            f"n steps per env. rollout {self._env_timesteps}\n" + \
            f"max n steps per env. episode {self._env_episode_n_steps_ub}\n" + \
            f"min n steps per env. episode {self._env_episode_n_steps_lb}\n" + \
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
        self._values = torch.full(size=(self._env_timesteps, self._num_envs, 1),
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

        self._next_obs = torch.full(size=(self._env_timesteps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._next_values = torch.full(size=(self._env_timesteps, self._num_envs, 1),
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