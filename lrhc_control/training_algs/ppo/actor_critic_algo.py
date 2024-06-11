from lrhc_control.agents.actor_critic.ppo_tanh import ActorCriticTanh

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
        self._agent = None 
        
        self._debug = debug

        self._optimizer = None

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._dbinfo_drop_fname = None
        self._model_path = None
        
        self._policy_update_db_data_dict =  {}
        self._custom_env_data_db_dict = {}
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
            model_path: str = None,
            n_evals: int = None,
            n_timesteps_per_eval: int = None,
            comment: str = "",
            dump_checkpoints: bool = False,
            norm_obs: bool = True):

        self._verbose = verbose

        self._dump_checkpoints = dump_checkpoints
        
        self._eval = eval

        self._run_name = run_name
        from datetime import datetime
        self._time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')
        self._unique_id = self._time_id + "-" + self._run_name
        self._init_algo_shared_data(static_params=self._hyperparameters) # can only handle dicts with
        # numeric values
        self._hyperparameters.update(custom_args)

        self._torch_device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")

        self._agent = ActorCriticTanh(obs_dim=self._env.obs_dim(),
                        actions_dim=self._env.actions_dim(),
                        actor_std=0.01,
                        critic_std=1.0,
                        norm_obs=norm_obs,
                        device=self._torch_device,
                        dtype=self._dtype,
                        is_eval=self._eval)
        # self._agent.to(self._torch_device) # move agent to target device

        # load model if necessary 
        if self._eval: # load pretrained model
            if model_path is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, a model_path should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            elif n_timesteps_per_eval is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, n_timesteps_per_eval should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            elif n_evals is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, n_evals should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            else:
                self._model_path = model_path
                self._rollout_timesteps = int(n_timesteps_per_eval/self._num_envs) # overrides 
                self._iterations_n = n_evals
            self._load_model(self._model_path)

        # create dump directory + copy important files for debug
        self._init_drop_dir(drop_dir_name)
            
        # seeding + deterministic behavior for reproducibility
        self._set_all_deterministic()

        if (self._debug):
            
            torch.autograd.set_detect_anomaly(self._debug)
            job_type = "evaluation" if self._eval else "training"
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

        if not self._eval:
            self._optimizer = optim.Adam(self._agent.parameters(), 
                                    lr=self._base_lr_actor, 
                                    eps=1e-5 # small constant added to the optimization
                                    )
            # self._optimizer = optim.Adam([
            #     {'params': self._agent.actor_mean.parameters(), 'lr': self._base_lr_actor},
            #     {'params': self._agent.critic.parameters(), 'lr': self._base_lr_critic}, ],
            #     lr=self._base_lr_actor, # default to actor lr (e.g. lfor ogstd parameter)
            #     eps=1e-5 # small constant added to the optimization
            #     )
        self._init_buffers()
        
        # self._env.reset()
        self._env._set_ep_rewards_scaling(scaling=self._rollout_timesteps)

        self._setup_done = True

        self._is_done = False

        self._start_time_tot = time.perf_counter()

        self._start_time = time.perf_counter()
    
    def is_done(self):

        return self._is_done 
    
    def model_path(self):

        return self._model_path
    
    def learn(self):
        
        if not self._setup_done:
            self._should_have_called_setup()

        # annealing the learning rate if enabled (may improve convergence)
        if self._anneal_lr:
            frac = 1.0 - (self._it_counter - 1.0) / self._iterations_n
            self._lr_now_actor = frac * self._base_lr_actor
            self._lr_now_critic = frac * self._base_lr_critic
            self._optimizer.param_groups[0]["lr"] = self._lr_now_actor
            # self._optimizer.param_groups[1]["lr"] = self._lr_now_critic

        self._episodic_reward_getter.reset() # necessary, we don't want to accumulate 
        # debug rewards from previous rollouts
        self._env.reset_custom_db_data() # reset custom db stats for this iteration

        self._start_time = time.perf_counter()

        rollout_ok = self._play()
        if not rollout_ok:
            return False
        
        self._rollout_t = time.perf_counter()

        self._compute_returns()
        self._gae_t = time.perf_counter()

        self._improve_policy()
        self._policy_update_t = time.perf_counter()

        self._post_step()

        return True

    def eval(self):

        if not self._setup_done:
            self._should_have_called_setup()

        self._episodic_reward_getter.reset()

        self._start_time = time.perf_counter()

        rollout_ok = self._play()
        if not rollout_ok:
            return False

        self._rollout_t = time.perf_counter()

        self._post_step()

        return True

    @abstractmethod
    def _play(self):
        pass
    
    @abstractmethod
    def _compute_returns(self):
       pass
    
    @abstractmethod
    def _improve_policy(self):
        pass

    def _save_model(self,
            is_checkpoint: bool = False):

        path = self._model_path
        if is_checkpoint: # use iteration as id
            path = path + "_checkpoint" + str(self._it_counter)
        info = f"Saving model to {path}"
        Journal.log(self.__class__.__name__,
            "done",
            info,
            LogType.INFO,
            throw_when_excep = True)
        torch.save(self._agent.state_dict(), path) # saves whole agent state
        # torch.save(self._agent.parameters(), path) # only save agent parameters
        info = f"Done."
        Journal.log(self.__class__.__name__,
            "done",
            info,
            LogType.INFO,
            throw_when_excep = True)
                    
    def done(self):
        
        if not self._is_done:

            if not self._eval:
                self._save_model()
            
            self._dump_dbinfo_to_file()
            
            if self._shared_algo_data is not None:
                self._shared_algo_data.close() # close shared memory

            self._env.close()

            self._is_done = True

    def _dump_dbinfo_to_file(self):

        import h5py

        info = f"Dumping debug info at {self._dbinfo_drop_fname}"
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        with h5py.File(self._dbinfo_drop_fname+".hdf5", 'w') as hf:
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
            hf.create_dataset('tot_loss_mean', data=self._tot_loss_mean.numpy())
            hf.create_dataset('value_los_means', data=self._value_loss_mean.numpy())
            hf.create_dataset('policy_loss_mean', data=self._policy_loss_mean.numpy())
            hf.create_dataset('entropy_loss_mean', data=self._entropy_loss_mean.numpy())
            hf.create_dataset('tot_loss_grad_norm_mean', data=self._tot_loss_grad_norm_mean.numpy())
            hf.create_dataset('actor_loss_grad_norm_mean', data=self._actor_loss_grad_norm_mean.numpy())

            hf.create_dataset('tot_loss_std', data=self._tot_loss_std.numpy())
            hf.create_dataset('value_loss_std', data=self._value_loss_std.numpy())
            hf.create_dataset('policy_loss_std', data=self._policy_loss_std.numpy())
            hf.create_dataset('entropy_loss_std', data=self._entropy_loss_std.numpy())
            hf.create_dataset('tot_loss_grad_norm_std', data=self._tot_loss_grad_norm_std.numpy())
            hf.create_dataset('actor_loss_grad_norm_std', data=self._actor_loss_grad_norm_std.numpy())

            hf.create_dataset('old_approx_kl_mean', data=self._old_approx_kl_mean.numpy())
            hf.create_dataset('approx_kl_mean', data=self._approx_kl_mean.numpy())
            hf.create_dataset('old_approx_kl_std', data=self._old_approx_kl_std.numpy())
            hf.create_dataset('approx_kl_std', data=self._approx_kl_std.numpy())

            hf.create_dataset('clipfrac_mean', data=self._clipfrac_mean.numpy())
            hf.create_dataset('clipfrac_std', data=self._clipfrac_std.numpy())
            
            hf.create_dataset('explained_variance', data=self._explained_variance.numpy())

            hf.create_dataset('batch_returns_std', data=self._batch_returns_std.numpy())
            hf.create_dataset('batch_returns_mean', data=self._batch_returns_mean.numpy())
            hf.create_dataset('batch_adv_std', data=self._batch_adv_std.numpy())
            hf.create_dataset('batch_adv_mean', data=self._batch_adv_mean.numpy())
            hf.create_dataset('batch_val_std', data=self._batch_val_std.numpy())
            hf.create_dataset('batch_val_mean', data=self._batch_val_mean.numpy())

            # dump all custom env data
            db_data_names = list(self._env.custom_db_data.keys())
            for db_dname in db_data_names:
                data=self._custom_env_data[db_dname]
                subnames = list(data.keys())
                for subname in subnames:
                    var_name = db_dname + "_" + subname
                    hf.create_dataset(var_name, data=data[subname])
            db_info_names = list(self._env.custom_db_info.keys())
            for db_info in db_info_names:
                hf.create_dataset(db_info, data=self._env.custom_db_info[db_info])
        info = f"done."
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)

    def _load_model(self,
            model_path: str):
        
        info = f"Loading model at {model_path}"

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

        # main drop directory
        if drop_dir_name is None:
            # drop to current directory
            self._drop_dir = "./" + f"{self.__class__.__name__}/" + self._run_name + "/" + self._unique_id
        else:
            self._drop_dir = drop_dir_name + "/" + f"{self.__class__.__name__}/" + self._run_name + "/" + self._unique_id
        os.makedirs(self._drop_dir)

        # model
        if not self._eval:
            self._model_path = self._drop_dir + "/" + self._unique_id + "_model"
        else: # we copy the model under evaluation to the drop dir
            shutil.copy(self._model_path, self._drop_dir)

        # debug info
        self._dbinfo_drop_fname = self._drop_dir + "/" + self._unique_id + "db_info" # extension added later

        # other auxiliary db files
        aux_drop_dir = self._drop_dir + "/other"
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
        
        self._learning_rates[self._it_counter-1, 0] = self._lr_now_actor
        self._learning_rates[self._it_counter-1, 0] = self._lr_now_critic

        self._env_step_fps[self._it_counter-1] = self._batch_size / self._rollout_dt[self._it_counter-1]
        self._env_step_rt_factor[self._it_counter-1] = self._env_step_fps[self._it_counter-1] * self._hyperparameters["control_clust_dt"]
        self._policy_update_fps[self._it_counter-1] = self._update_epochs * self._num_minibatches / self._policy_update_dt[self._it_counter-1]

        # after rolling out policy, we get the episodic reward for the current policy
        self._episodic_rewards[self._it_counter-1, :, :] = self._episodic_reward_getter.get_rollout_avrg_total_reward() # total ep. rewards across envs
        self._episodic_rewards_env_avrg[self._it_counter-1, :, :] = self._episodic_reward_getter.get_rollout_avrg_total_reward_env_avrg() # tot, avrg over envs
        self._episodic_sub_rewards[self._it_counter-1, :, :] = self._episodic_reward_getter.get_rollout_avrg_reward() # sub-episodic rewards across envs
        self._episodic_sub_rewards_env_avrg[self._it_counter-1, :, :] = self._episodic_reward_getter.get_rollout_reward_env_avrg() # avrg over envs

        # fill env db info
        db_data_names = list(self._env.custom_db_data.keys())
        for dbdatan in db_data_names:
            self._custom_env_data[dbdatan]["rollout_stat"][self._it_counter-1, :, :] = self._env.custom_db_data[dbdatan].get_rollout_stat()
            self._custom_env_data[dbdatan]["rollout_stat_env_avrg"][self._it_counter-1, :, :] = self._env.custom_db_data[dbdatan].get_rollout_stat_env_avrg()
            self._custom_env_data[dbdatan]["rollout_stat_comp"][self._it_counter-1, :, :] = self._env.custom_db_data[dbdatan].get_rollout_stat_comp()
            self._custom_env_data[dbdatan]["rollout_stat_comp_env_avrg"][self._it_counter-1, :, :] = self._env.custom_db_data[dbdatan].get_rollout_stat_comp_env_avrg()

        self._log_info()

        if self._it_counter == self._iterations_n:
            self.done()
        else:
            if self._dump_checkpoints and (self._it_counter % self._m_checkpoint_freq == 0):
                self._save_model(is_checkpoint=True)
            
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
                "lr_now_actor",
                "lr_now_critic",
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
                self._lr_now_actor,
                self._lr_now_critic,
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
                      self._episodic_sub_rewards_env_avrg[self._it_counter-1, :, i:i+1] for i in range(len(self._reward_names))})
            wandb_d.update({f"sub_reward/{self._reward_names[i]}":
                      wandb.Histogram(self._episodic_sub_rewards.numpy()[self._it_counter-1, :, i:i+1]) for i in range(len(self._reward_names))})
            
            # add custom env db data
            db_data_names = list(self._env.custom_db_data.keys())
            for dbdatan in db_data_names: 
                data = self._custom_env_data[dbdatan]
                data_names = self._env.custom_db_data[dbdatan].data_names()
                self._custom_env_data_db_dict.update({f"{dbdatan}" + "_rollout_stat_comp": 
                        wandb.Histogram(data["rollout_stat_comp"][self._it_counter-1, :, :].numpy())})
                self._custom_env_data_db_dict.update({f"{dbdatan}" + "_rollout_stat_comp_env_avrg": 
                        data["rollout_stat_comp_env_avrg"][self._it_counter-1, :, :].item()})
                self._custom_env_data_db_dict.update({f"sub_env_dbdata/{dbdatan}-{data_names[i]}" + "_rollout_stat_env_avrg": 
                       data["rollout_stat_env_avrg"][self._it_counter-1, :, i:i+1] for i in range(len(data_names))})
                self._custom_env_data_db_dict.update({f"sub_env_dbdata/{dbdatan}-{data_names[i]}" + "_rollout_stat": 
                        wandb.Histogram(data["rollout_stat"].numpy()[self._it_counter-1, :, i:i+1]) for i in range(len(data_names))})

            # write debug info to shared memory    
            wandb_d.update(self._policy_update_db_data_dict)
            wandb_d.update(self._custom_env_data_db_dict)

            wandb.log(wandb_d)

        if self._verbose:
            
            info = f"\nN. PPO iterations performed: {self._it_counter}/{self._iterations_n}\n" + \
                f"N. policy updates performed: {self._n_policy_updates[self._it_counter-1].item()}/" + \
                f"{self._update_epochs * self._num_minibatches * self._iterations_n}\n" + \
                f"N. env steps performed: {self._it_counter * self._batch_size}/{self._total_timesteps}\n" + \
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
        self._learning_rates = torch.full((self._iterations_n, 2), 
                    dtype=torch.float32, fill_value=0, device="cpu")
        
        # reward db data
        tot_ep_rew_shape = self._episodic_reward_getter.get_rollout_avrg_total_reward().shape
        tot_ep_rew_shape_env_avrg_shape = self._episodic_reward_getter.get_rollout_avrg_total_reward_env_avrg().shape
        rollout_avrg_rew_shape = self._episodic_reward_getter.get_rollout_avrg_reward().shape
        rollout_avrg_rew_env_avrg_shape = self._episodic_reward_getter.get_rollout_reward_env_avrg().shape
        self._reward_names = self._episodic_reward_getter.reward_names()
        self._reward_names_str = "[" + ', '.join(self._reward_names) + "]"
        self._episodic_rewards = torch.full((self._iterations_n, tot_ep_rew_shape[0], tot_ep_rew_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_rewards_env_avrg = torch.full((self._iterations_n, tot_ep_rew_shape_env_avrg_shape[0], tot_ep_rew_shape_env_avrg_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_sub_rewards = torch.full((self._iterations_n, rollout_avrg_rew_shape[0], rollout_avrg_rew_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_sub_rewards_env_avrg = torch.full((self._iterations_n, rollout_avrg_rew_env_avrg_shape[0], rollout_avrg_rew_env_avrg_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")

        # ppo iteration db data
        self._tot_loss_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._value_loss_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._policy_loss_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._entropy_loss_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_loss_grad_norm_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._actor_loss_grad_norm_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._tot_loss_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._value_loss_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._policy_loss_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._entropy_loss_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_loss_grad_norm_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._actor_loss_grad_norm_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._old_approx_kl_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._approx_kl_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._old_approx_kl_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._approx_kl_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._clipfrac_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._clipfrac_std= torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._explained_variance = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._batch_returns_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_returns_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_adv_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_adv_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_val_std = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_val_mean = torch.full((self._iterations_n, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        # custom data from env
        self._custom_env_data = {}
        db_data_names = list(self._env.custom_db_data.keys())
        for dbdatan in db_data_names:
            self._custom_env_data[dbdatan] = {}
            rollout_stat=self._env.custom_db_data[dbdatan].get_rollout_stat()
            self._custom_env_data[dbdatan]["rollout_stat"] = torch.full((self._iterations_n, 
                                                                rollout_stat.shape[0], 
                                                                rollout_stat.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_env_avrg=self._env.custom_db_data[dbdatan].get_rollout_stat_env_avrg()
            self._custom_env_data[dbdatan]["rollout_stat_env_avrg"] = torch.full((self._iterations_n, 
                                                                        rollout_stat_env_avrg.shape[0], 
                                                                        rollout_stat_env_avrg.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_comp=self._env.custom_db_data[dbdatan].get_rollout_stat_comp()
            self._custom_env_data[dbdatan]["rollout_stat_comp"] = torch.full((self._iterations_n, 
                                                                    rollout_stat_comp.shape[0], 
                                                                    rollout_stat_comp.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_comp_env_avrg=self._env.custom_db_data[dbdatan].get_rollout_stat_comp_env_avrg()
            self._custom_env_data[dbdatan]["rollout_stat_comp_env_avrg"] = torch.full((self._iterations_n, rollout_stat_comp_env_avrg.shape[0], rollout_stat_comp_env_avrg.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")

    def _init_params(self):

        self._dtype = self._env.dtype()

        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()

        self._run_name = "DefaultRun" # default
        self._env_name = self._env.name()
        self._episode_timeout_lb, self._episode_timeout_ub = self._env.episode_timeout_bounds()
        self._task_rand_timeout_lb, self._task_rand_timeout_ub = self._env.task_rand_timeout_bounds()
        self._env_n_action_reps = self._env.n_action_reps()
        
        self._use_gpu = self._env.using_gpu()
        self._torch_device = torch.device("cpu") # defaults to cpu
        self._torch_deterministic = True

        self._m_checkpoint_freq = 50 # n ppo iterations after which a checkpoint model is dumped

        # policy rollout and return comp./adv estimation
        self._total_timesteps_nom = int(50e6) # atomic env steps (including substepping if action reps>1)
        self._total_timesteps_nom = self._total_timesteps_nom//self._env_n_action_reps # correct with n of action reps
        
        self._rollout_timesteps = 128 # numer of vectorized steps (does not include env substepping) 
        # to be done per policy rollout (influences adv estimation!!!)
        self._batch_size = self._rollout_timesteps * self._num_envs

        self._iterations_n = self._total_timesteps_nom//self._batch_size # number of ppo iterations
        self._total_timesteps = self._iterations_n*self._batch_size
        
        # policy update
        self._num_minibatches = 8
        self._minibatch_size = self._batch_size // self._num_minibatches
        
        self._base_lr_actor = 3e-4
        self._base_lr_critic = 3e-4
        self._lr_now_actor = self._base_lr_actor
        self._lr_now_critic= self._base_lr_critic
        self._anneal_lr = False

        self._discount_factor = 0.99
        self._gae_lambda = 0.95 # λ = 1 gives an unbiased estimate of the total reward (but high variance),
        # λ < 1 gives a biased estimate, but with less variance. 0.95
        
        self._update_epochs = 10
        self._norm_adv = True
        self._clip_vloss = False
        self._clip_coef_vf = 0.2 # IMPORTANT: this clipping depends on the reward scaling (only used if clip_vloss)
        self._clip_coef = 0.2
        self._entropy_coeff = 1e-4
        self._val_f_coeff = 0.5
        self._max_grad_norm_actor = 0.5
        self._max_grad_norm_critic = 0.5
        self._target_kl = None

        self._n_policy_updates_to_be_done = self._update_epochs * self._num_minibatches * self._iterations_n

        # write them to hyperparam dictionary for debugging
        self._hyperparameters["n_envs"] = self._num_envs
        self._hyperparameters["obs_dim"] = self._obs_dim
        self._hyperparameters["actions_dim"] = self._actions_dim
        # self._hyperparameters["critic_size"] = self._critic_size
        # self._hyperparameters["actor_size"] = self._actor_size
        self._hyperparameters["seed"] = self._seed
        self._hyperparameters["using_gpu"] = self._use_gpu
        self._hyperparameters["n_iterations"] = self._iterations_n
        self._hyperparameters["n_policy_updates_per_batch"] = self._update_epochs * self._num_minibatches
        self._hyperparameters["n_policy_updates_when_done"] = self._n_policy_updates_to_be_done
        self._hyperparameters["episodes timeout lb"] = self._episode_timeout_lb
        self._hyperparameters["episodes timeout ub"] = self._episode_timeout_ub
        self._hyperparameters["task rand timeout lb"] = self._task_rand_timeout_lb
        self._hyperparameters["task rand timeout ub"] = self._task_rand_timeout_ub
        self._hyperparameters["env_n_action_reps"] = self._env_n_action_reps
        self._hyperparameters["n steps per env. rollout"] = self._rollout_timesteps
        self._hyperparameters["per-batch update_epochs"] = self._update_epochs
        self._hyperparameters["per-epoch policy updates"] = self._num_minibatches
        self._hyperparameters["total policy updates to be performed"] = self._update_epochs * self._num_minibatches * self._iterations_n
        self._hyperparameters["total_timesteps to be simulated"] = self._total_timesteps
        self._hyperparameters["batch_size"] = self._batch_size
        self._hyperparameters["minibatch_size"] = self._minibatch_size
        self._hyperparameters["total_timesteps"] = self._total_timesteps
        self._hyperparameters["base_lr_actor"] = self._base_lr_actor
        self._hyperparameters["base_lr_critic"] = self._base_lr_critic
        self._hyperparameters["anneal_lr"] = self._anneal_lr
        self._hyperparameters["discount_factor"] = self._discount_factor
        self._hyperparameters["gae_lambda"] = self._gae_lambda
        self._hyperparameters["norm_adv"] = self._norm_adv
        self._hyperparameters["clip_coef"] = self._clip_coef
        self._hyperparameters["clip_coef_vf"] = self._clip_coef_vf
        self._hyperparameters["clip_vloss"] = self._clip_vloss
        self._hyperparameters["entropy_coeff"] = self._entropy_coeff
        self._hyperparameters["val_f_coeff"] = self._val_f_coeff
        self._hyperparameters["max_grad_norm_actor"] = self._max_grad_norm_actor
        self._hyperparameters["max_grad_norm_critic"] = self._max_grad_norm_critic
        self._hyperparameters["target_kl"] = self._target_kl

        # small debug log
        info = f"\nUsing \n" + \
            f"n vec. steps per policy rollout {self._rollout_timesteps}\n" + \
            f"batch_size {self._batch_size}\n" + \
            f"num_minibatches for policy update {self._num_minibatches}\n" + \
            f"minibatch_size {self._minibatch_size}\n" + \
            f"per-batch update_epochs {self._update_epochs}\n" + \
            f"iterations_n {self._iterations_n}\n" + \
            f"episode timeout max steps {self._episode_timeout_ub}\n" + \
            f"episode timeout min steps {self._episode_timeout_lb}\n" + \
            f"task rand. max n steps {self._task_rand_timeout_ub}\n" + \
            f"task rand. min n steps {self._task_rand_timeout_lb}\n" + \
            f"number of action reps {self._env_n_action_reps}\n" + \
            f"total policy updates to be performed {self._update_epochs * self._num_minibatches * self._iterations_n}\n" + \
            f"total_timesteps to be simulated {self._total_timesteps}\n"
        Journal.log(self.__class__.__name__,
            "_init_params",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._it_counter = 0

    def _init_buffers(self):

        self._obs = torch.full(size=(self._rollout_timesteps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._values = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
            
        self._actions = torch.full(size=(self._rollout_timesteps, self._num_envs, self._actions_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._logprobs = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)

        self._next_obs = torch.full(size=(self._rollout_timesteps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._next_values = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._next_terminations = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._next_dones = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)
        
        self._rewards = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        
        
        self._advantages = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._returns = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
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
