from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo
from lrhc_control.agents.sactor_critic.sac import SACAgent

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

class DummyTestAlgoBase():

    # base class for actor-critic RL algorithms
     
    def __init__(self,
            env, 
            debug = False,
            remote_db = False,
            anomaly_detect = False,
            seed: int = 1):

        self._env = env 
        self._seed = seed

        self._agent = None 
        
        self._debug = debug
        self._remote_db = remote_db

        self._anomaly_detect = anomaly_detect

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._dbinfo_drop_fname = None
        self._model_path = None
        
        self._policy_update_db_data_dict =  {}
        self._custom_env_data_db_dict = {}
        self._hyperparameters = {}
        
        self._episodic_reward_metrics = self._env.ep_rewards_metrics()
        
        tot_tsteps=50e6
        self._init_params(tot_tsteps=tot_tsteps)
        
        self._init_dbdata()

        self._setup_done = False

        self._verbose = False

        self._is_done = False
        
        self._shared_algo_data = None

        self._this_child_path = None
        self._this_basepath = os.path.abspath(__file__)
    
    def __del__(self):

        self.done()

    def eval(self):

        if not self._setup_done:
            self._should_have_called_setup()

        self._start_time = time.perf_counter()

        if not self._collect_transition():
            return False

        self._collection_t = time.perf_counter()
        
        self._post_step()

        return True
    
    @abstractmethod
    def _collect_transition(self)->bool:
        pass

    def setup(self,
            run_name: str,
            ns: str,
            custom_args: Dict = {},
            verbose: bool = False,
            drop_dir_name: str = None,
            eval: bool = False,
            model_path: str = None,
            n_eval_timesteps: int = None,
            comment: str = "",
            dump_checkpoints: bool = False,
            norm_obs: bool = True):

        self._verbose = verbose

        self._ns=ns # only used for shared mem stuff
        
        self._run_name = run_name
        from datetime import datetime
        self._time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')
        self._unique_id = self._time_id + "-" + self._run_name
        self._init_algo_shared_data(static_params=self._hyperparameters) # can only handle dicts with
        # numeric values

        data_names={}
        data_names["obs_names"]=self._env.obs_names()
        data_names["action_names"]=self._env.action_names()
        data_names["sub_reward_names"]=self._env.sub_rew_names()

        self._hyperparameters.update(custom_args)
        self._hyperparameters.update(data_names)

        self._torch_device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")

        self._agent = DummyAgent(obs_dim=self._env.obs_dim(),
                    actions_dim=self._env.actions_dim(),
                    actions_ub=self._env.get_actions_ub().flatten().tolist(),
                    actions_lb=self._env.get_actions_lb().flatten().tolist(),
                    device=self._torch_device,
                    dtype=self._dtype,
                    debug=self._debug)
                
        # create dump directory + copy important files for debug
        self._init_drop_dir(drop_dir_name)
            
        # seeding + deterministic behavior for reproducibility
        self._set_all_deterministic()
        torch.autograd.set_detect_anomaly(self._anomaly_detect)

        if (self._debug):
            if self._remote_db:
                job_type = "dummy"
                full_run_config={**self._hyperparameters,**self._env.custom_db_info}
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
                    config=full_run_config,
                    monitor_gym=True,
                    save_code=True,
                    dir=self._drop_dir
                )
                wandb.watch(self._agent, log="all")
        
        # self._env.reset()
        
        self._setup_done = True

        self._is_done = False

        self._start_time_tot = time.perf_counter()

        self._start_time = time.perf_counter()
    
    def is_done(self):

        return self._is_done 
    
    def model_path(self):

        return self._model_path

    def done(self):
        
        if not self._is_done:
            
            self._dump_dbinfo_to_file()
            
            if self._shared_algo_data is not None:
                self._shared_algo_data.write(dyn_info_name=["is_done"],
                    val=[1.0])
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
            hf.create_dataset('env_step_fps', data=self._env_step_fps.numpy())
            hf.create_dataset('env_step_rt_factor', data=self._env_step_rt_factor.numpy())
            hf.create_dataset('policy_update_dt', data=self._policy_update_dt.numpy())
            hf.create_dataset('policy_update_fps', data=self._policy_update_fps.numpy())
            hf.create_dataset('n_of_played_episodes', data=self._n_of_played_episodes.numpy())
            hf.create_dataset('n_timesteps_done', data=self._n_timesteps_done.numpy())
            hf.create_dataset('n_policy_updates', data=self._n_policy_updates.numpy())
            hf.create_dataset('n_qfun_updates', data=self._n_qfun_updates.numpy())
            hf.create_dataset('n_tqfun_updates', data=self._n_tqfun_updates.numpy())

            hf.create_dataset('elapsed_min', data=self._elapsed_min.numpy())

            # algo data 
            hf.create_dataset('qf1_vals', data=self._qf1_vals.numpy())
            hf.create_dataset('qf2_vals', data=self._qf2_vals.numpy())
            hf.create_dataset('qf1_loss', data=self._qf1_loss.numpy())
            hf.create_dataset('qf2_loss', data=self._qf2_loss.numpy())
            hf.create_dataset('qf_loss', data=self._qf_loss.numpy())
            hf.create_dataset('actor_loss', data=self._actor_loss.numpy())
            hf.create_dataset('alphas', data=self._alphas.numpy())
            hf.create_dataset('alpha_loss', data=self._alpha_loss.numpy())

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
            
            # other data
            hf.create_dataset('running_mean_obs', data=self._running_mean_obs.numpy())
            hf.create_dataset('running_std_obs', data=self._running_std_obs.numpy())
        
        info = f"done."
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)

    def _set_all_deterministic(self):

        random.seed(self._seed) # python seed
        torch.manual_seed(self._seed)
        torch.backends.cudnn.deterministic = self._torch_deterministic
        # torch.use_deterministic_algorithms(mode=True) # will throw excep. when trying to use non-det. algos
        import numpy as np
        np.random.seed(0)

    def drop_dir(self):
        return self._drop_dir
        
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
        self._model_path = self._drop_dir + "/" + self._unique_id + "_model"

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
                
        self._step_dt[self._log_it_counter] += \
            (self._collection_t-self._start_time) 
    
        if self._vec_transition_counter % self._db_vecstep_frequency== 0:
            # only log data every n timesteps 
        
            self._env_step_fps[self._log_it_counter] = (self._db_vecstep_frequency*self._num_envs)/ self._step_dt[self._log_it_counter]
            if "control_clust_dt" in self._hyperparameters:
                self._env_step_rt_factor[self._log_it_counter] = self._env_step_fps[self._log_it_counter]*self._env_n_action_reps*self._hyperparameters["control_clust_dt"]

            self._n_of_played_episodes[self._log_it_counter] = self._episodic_reward_metrics.get_n_played_episodes()
            self._n_timesteps_done[self._log_it_counter]=(self._vec_transition_counter+1)*self._num_envs

            self._elapsed_min[self._log_it_counter] = (time.perf_counter() - self._start_time_tot)/60.0
        
            # we get some episodic reward metrics
            self._episodic_rewards[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_avrg_over_eps() # total ep. rewards across envs
            self._episodic_rewards_env_avrg[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_avrg() # tot, avrg over envs
            self._episodic_sub_rewards[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_avrg_over_eps() # sub-episodic rewards across envs
            self._episodic_sub_rewards_env_avrg[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_env_avrg_over_eps() # avrg over envs

            # fill env custom db metrics
            db_data_names = list(self._env.custom_db_data.keys())
            for dbdatan in db_data_names:
                self._custom_env_data[dbdatan]["rollout_stat"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_sub_avrg_over_eps()
                self._custom_env_data[dbdatan]["rollout_stat_env_avrg"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_sub_env_avrg_over_eps()
                self._custom_env_data[dbdatan]["rollout_stat_comp"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_avrg_over_eps()
                self._custom_env_data[dbdatan]["rollout_stat_comp_env_avrg"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_tot_avrg()

            self._log_info()

            self._log_it_counter+=1 

        self._vec_transition_counter+=1
        if self._vec_transition_counter == self._total_timesteps_vec:
            self.done()            
            
    def _should_have_called_setup(self):

        exception = f"setup() was not called!"

        Journal.log(self.__class__.__name__,
            "_should_have_called_setup",
            exception,
            LogType.EXCEP,
            throw_when_excep = True)
    
    def _log_info(self):
        
        if self._debug or self._verbose:
            elapsed_h = self._elapsed_min[self._log_it_counter].item()/60.0
            est_remaining_time_h =  elapsed_h * 1/(self._vec_transition_counter+1) * (self._total_timesteps_vec-self._vec_transition_counter-1)
            is_done=self._vec_transition_counter==self._total_timesteps_vec
     
        if self._debug:
            # add custom env db data (dumped LOCALLY at the end of training)
            db_data_names = list(self._env.custom_db_data.keys())
            for dbdatan in db_data_names: 
                data = self._custom_env_data[dbdatan]
                data_names = self._env.custom_db_data[dbdatan].data_names()
                self._custom_env_data_db_dict.update({f"{dbdatan}" + "_rollout_stat_comp": 
                        wandb.Histogram(data["rollout_stat_comp"][self._log_it_counter-1, :, :].numpy())})
                self._custom_env_data_db_dict.update({f"{dbdatan}" + "_rollout_stat_comp_env_avrg": 
                        data["rollout_stat_comp_env_avrg"][self._log_it_counter-1, :, :].item()})
                self._custom_env_data_db_dict.update({f"sub_env_dbdata/{dbdatan}-{data_names[i]}" + "_rollout_stat_env_avrg": 
                       data["rollout_stat_env_avrg"][self._log_it_counter-1, :, i:i+1] for i in range(len(data_names))})
                self._custom_env_data_db_dict.update({f"sub_env_dbdata/{dbdatan}-{data_names[i]}" + "_rollout_stat": 
                        wandb.Histogram(data["rollout_stat"].numpy()[self._log_it_counter-1, :, i:i+1]) for i in range(len(data_names))})

            if self._remote_db: 
                # write general algo debug info to shared memory    
                info_names=self._shared_algo_data.dynamic_info.get()
                info_data = [
                    self._n_timesteps_done[self._log_it_counter].item(),
                    elapsed_h,
                    est_remaining_time_h,
                    self._env_step_fps[self._log_it_counter].item(),
                    self._env_step_rt_factor[self._log_it_counter].item(),
                    is_done
                    ]
                self._shared_algo_data.write(dyn_info_name=info_names,
                                        val=info_data)

                # write debug info to remote wandb server
                wandb_d = {'tot_episodic_reward': wandb.Histogram(self._episodic_rewards[self._log_it_counter-1, :, :].numpy()),
                    'tot_episodic_reward_env_avrg': self._episodic_rewards_env_avrg[self._log_it_counter-1, :, :].item(),
                    'log_iteration' : self._log_it_counter}
                wandb_d.update(dict(zip(info_names, info_data)))
                wandb_d.update({f"sub_reward/{self._reward_names[i]}_env_avrg":
                        self._episodic_sub_rewards_env_avrg[self._log_it_counter-1, :, i:i+1] for i in range(len(self._reward_names))})
                wandb_d.update({f"sub_reward/{self._reward_names[i]}":
                        wandb.Histogram(self._episodic_sub_rewards.numpy()[self._log_it_counter-1, :, i:i+1]) for i in range(len(self._reward_names))})
                
                wandb_d.update(self._policy_update_db_data_dict)
                wandb_d.update(self._custom_env_data_db_dict)

                wandb.log(wandb_d)

        if self._verbose:
                       
            info =f"\nTotal n. timesteps simulated: {self._n_timesteps_done[self._log_it_counter].item()}/{self._total_timesteps}\n" + \
                f"Elapsed time: {self._elapsed_min[self._log_it_counter].item()/60.0} h\n" + \
                f"Estimated remaining time: " + \
                f"{est_remaining_time_h} h\n" + \
                f"Average episodic return across all environments: {self._episodic_rewards_env_avrg[self._log_it_counter, :, :].item()}\n" + \
                f"Average episodic returns across all environments {self._reward_names_str}: {self._episodic_sub_rewards_env_avrg[self._log_it_counter, :]}\n" + \
                f"Current env. step sps: {self._env_step_fps[self._log_it_counter].item()}, time for experience collection {self._step_dt[self._log_it_counter].item()} s\n" + \
                f"Current env (sub-steping) rt factor: {self._env_step_rt_factor[self._log_it_counter].item()}\n"            
            
            Journal.log(self.__class__.__name__,
                "_post_step",
                info,
                LogType.INFO,
                throw_when_excep = True)

    def _init_dbdata(self):

        # initalize some debug data
        self._collection_t = -1.0
        
        self._n_of_played_episodes = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_timesteps_done = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._elapsed_min = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0, device="cpu")
        
        # reward db data
        tot_ep_rew_shape = self._episodic_reward_metrics.get_avrg_over_eps().shape
        tot_ep_rew_shape_env_avrg_shape = self._episodic_reward_metrics.get_tot_avrg().shape
        rollout_avrg_rew_shape = self._episodic_reward_metrics.get_sub_avrg_over_eps().shape
        rollout_avrg_rew_env_avrg_shape = self._episodic_reward_metrics.get_sub_env_avrg_over_eps().shape
        self._reward_names = self._episodic_reward_metrics.reward_names()
        self._reward_names_str = "[" + ', '.join(self._reward_names) + "]"
        self._episodic_rewards = torch.full((self._db_data_size, tot_ep_rew_shape[0], tot_ep_rew_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_rewards_env_avrg = torch.full((self._db_data_size, tot_ep_rew_shape_env_avrg_shape[0], tot_ep_rew_shape_env_avrg_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_sub_rewards = torch.full((self._db_data_size, rollout_avrg_rew_shape[0], rollout_avrg_rew_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_sub_rewards_env_avrg = torch.full((self._db_data_size, rollout_avrg_rew_env_avrg_shape[0], rollout_avrg_rew_env_avrg_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        
        # custom data from env
        self._custom_env_data = {}
        db_data_names = list(self._env.custom_db_data.keys())
        for dbdatan in db_data_names:
            self._custom_env_data[dbdatan] = {}
            rollout_stat=self._env.custom_db_data[dbdatan].get_sub_avrg_over_eps()
            self._custom_env_data[dbdatan]["rollout_stat"] = torch.full((self._db_data_size, 
                                                                rollout_stat.shape[0], 
                                                                rollout_stat.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_env_avrg=self._env.custom_db_data[dbdatan].get_sub_env_avrg_over_eps()
            self._custom_env_data[dbdatan]["rollout_stat_env_avrg"] = torch.full((self._db_data_size, 
                                                                        rollout_stat_env_avrg.shape[0], 
                                                                        rollout_stat_env_avrg.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_comp=self._env.custom_db_data[dbdatan].get_avrg_over_eps()
            self._custom_env_data[dbdatan]["rollout_stat_comp"] = torch.full((self._db_data_size, 
                                                                    rollout_stat_comp.shape[0], 
                                                                    rollout_stat_comp.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_comp_env_avrg=self._env.custom_db_data[dbdatan].get_tot_avrg()
            self._custom_env_data[dbdatan]["rollout_stat_comp_env_avrg"] = torch.full((self._db_data_size, rollout_stat_comp_env_avrg.shape[0], rollout_stat_comp_env_avrg.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")

        # other data
        self._running_mean_obs = torch.full((self._db_data_size, self._env.obs_dim()), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._running_std_obs = torch.full((self._db_data_size, self._env.obs_dim()), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
    def _init_params(self,
            tot_tsteps: int,
            run_name: str = "SACDefaultRunName"):

        self._dtype = self._env.dtype()

        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()

        self._run_name = run_name # default
        self._env_name = self._env.name()
        self._episode_timeout_lb, self._episode_timeout_ub = self._env.episode_timeout_bounds()
        self._task_rand_timeout_lb, self._task_rand_timeout_ub = self._env.task_rand_timeout_bounds()
        self._env_n_action_reps = self._env.n_action_reps()
        
        self._use_gpu = self._env.using_gpu()
        self._torch_device = torch.device("cpu") # defaults to cpu
        self._torch_deterministic = True

        # main algo settings
        self._total_timesteps = int(tot_tsteps)
        self._total_timesteps = self._total_timesteps//self._env_n_action_reps # correct with n of action reps
        self._total_timesteps_vec = self._total_timesteps // self._num_envs
        self._total_timesteps = self._total_timesteps_vec*self._num_envs # actual n transitions
        
        # debug
        self._db_vecstep_frequency = 128 # log db data every n (vectorized) timesteps
        
        self._db_data_size = round(self._total_timesteps_vec/self._db_vecstep_frequency)+self._db_vecstep_frequency
        # write them to hyperparam dictionary for debugging
        self._hyperparameters["n_envs"] = self._num_envs
        self._hyperparameters["obs_dim"] = self._obs_dim
        self._hyperparameters["actions_dim"] = self._actions_dim
        # self._hyperparameters["critic_size"] = self._critic_size
        # self._hyperparameters["actor_size"] = self._actor_size
        self._hyperparameters["seed"] = self._seed
        self._hyperparameters["using_gpu"] = self._use_gpu
        self._hyperparameters["total_timesteps_vec"] = self._total_timesteps_vec

        self._hyperparameters["episodes timeout lb"] = self._episode_timeout_lb
        self._hyperparameters["episodes timeout ub"] = self._episode_timeout_ub
        self._hyperparameters["task rand timeout lb"] = self._task_rand_timeout_lb
        self._hyperparameters["task rand timeout ub"] = self._task_rand_timeout_ub
        
        self._hyperparameters["_total_timesteps"] = self._total_timesteps
        self._hyperparameters["_db_vecstep_frequency"] = self._db_vecstep_frequency

        # small debug log
        info = f"\nUsing \n" + \
            f"total (vectorized) timesteps to be simulated {self._total_timesteps_vec}\n" + \
            f"total timesteps to be simulated {self._total_timesteps}\n" + \
            f"episode timeout max steps {self._episode_timeout_ub}\n" + \
            f"episode timeout min steps {self._episode_timeout_lb}\n" + \
            f"task rand. max n steps {self._task_rand_timeout_ub}\n" + \
            f"task rand. min n steps {self._task_rand_timeout_lb}\n" + \
            f"number of action reps {self._env_n_action_reps}\n" 
        
        Journal.log(self.__class__.__name__,
            "_init_params",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._vec_transition_counter = 0
        self._log_it_counter = 0

    def _init_replay_buffers(self):
        
        self._bpos = 0

        self._obs = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._obs_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._actions = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._actions_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._rewards = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, 1),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._next_obs = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._obs_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._next_terminal = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)

    def _add_experience(self, 
            obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
            next_obs: torch.Tensor, 
            next_terminal: torch.Tensor) -> None:
        
        self._obs[self._bpos] = obs
        self._next_obs[self._bpos] = next_obs
        self._actions[self._bpos] = actions
        self._rewards[self._bpos] = rewards
        self._next_terminal[self._bpos] = next_terminal

        self._bpos += 1
        if self._bpos == self._replay_buffer_size_vec:
            self._replay_bf_full = True
            self._bpos = 0
    
    def _init_algo_shared_data(self,
                static_params: Dict):

        self._shared_algo_data = SharedRLAlgorithmInfo(namespace=self._ns,
                is_server=True, 
                static_params=static_params,
                verbose=self._verbose, 
                vlevel=VLevel.V2, 
                safe=False,
                force_reconnection=True)

        self._shared_algo_data.run()

        # write some initializations
        self._shared_algo_data.write(dyn_info_name=["is_done"],
                val=[0.0])